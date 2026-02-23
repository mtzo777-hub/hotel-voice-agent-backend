# faq_retriever.py
"""
Sunshine Hotel Voice Agent - FAQ Retriever (BM25 + lightweight synonym/alias expansion)

Fixes included:
- Backend goodbye intent: returns "Thank you. Goodbye!" consistently (matches frontend behavior).
- Morphology normalization: plural/singular + noun/verb variants (hour/hours, smoke/smoking, etc.).
- Out-of-domain fallback reliability:
  * Fix score normalization bug (weak matches no longer appear as score=1).
  * Domain gating + OOD conversational patterns to reduce false positives.
- No external dependencies (safe for Cloud Run builds).
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _micro_stems(tok: str) -> List[str]:
    """
    Very small, controlled stemming to handle your reported failures:
    - hours -> hour
    - methods -> method
    - rates -> rate
    - smoking -> smoke
    - soundproofing -> soundproof
    Keep it conservative to avoid breaking working queries.
    """
    t = tok
    out = {t}

    if len(t) >= 4 and t.endswith("s") and not t.endswith("ss"):
        out.add(t[:-1])

    if len(t) >= 6 and t.endswith("ing"):
        base = t[:-3]
        out.add(base)
        # smoke <- smoking
        if base.endswith("k"):
            out.add(base + "e")

    if len(t) >= 8 and t.endswith("proofing"):
        out.add(t.replace("proofing", "proof"))

    return sorted(out)


def _tokens(s: str) -> List[str]:
    s = _norm(s)
    base = _WORD_RE.findall(s)
    expanded: List[str] = []
    for b in base:
        expanded.extend(_micro_stems(b))
    # De-dup while preserving order
    seen = set()
    out = []
    for t in expanded:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


class _BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = float(k1)
        self.b = float(b)
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 0.0

        df: Dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.df = df

        self.idf: Dict[str, float] = {}
        for t, dfi in df.items():
            self.idf[t] = math.log(1 + (self.n - dfi + 0.5) / (dfi + 0.5))

        self.tf: List[Dict[str, int]] = []
        for d in docs:
            m: Dict[str, int] = {}
            for t in d:
                m[t] = m.get(t, 0) + 1
            self.tf.append(m)

    def score(self, q_tokens: List[str]) -> List[float]:
        if not self.docs:
            return []
        if not q_tokens:
            return [0.0] * len(self.docs)

        scores = [0.0] * len(self.docs)
        for qt in q_tokens:
            idf = self.idf.get(qt, 0.0)
            if idf <= 0:
                continue
            for i in range(len(self.docs)):
                f = self.tf[i].get(qt, 0)
                if f <= 0:
                    continue
                dl = len(self.docs[i])
                denom = f + self.k1 * \
                    (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[i] += idf * (f * (self.k1 + 1.0)) / (denom or 1.0)
        return scores


@dataclass
class _FAQItem:
    faq_id: str
    text: str
    phrase: str
    doc_text: str


class FAQRetriever:
    DEFAULT_FALLBACK = "Sorry, I don't have that information yet. Please contact the hotel reception for assistance."
    GOODBYE_TEXT = "Thank you. Goodbye!"

    # High precision good-bye patterns
    _GOODBYE_RE = re.compile(
        r"\b(bye|goodbye|see you|see ya|thanks\s*(and|,)?\s*bye|thank you\s*(and|,)?\s*bye|okay\s*bye|ok\s*bye)\b",
        re.IGNORECASE,
    )

    # Strong OOD keywords (keep relatively high precision)
    _OOD_KEYWORDS = {
        "warranty", "watch", "bicycle", "bike", "book", "boyfriend", "girlfriend",
        "sell houses", "electrical bill", "university", "sleeping", "how old are you",
        # umbrella -> often used in "where is my umbrella"
        "teach me", "drive", "umbrella",
        "beer", "vodka", "whisky", "whiskey", "wine",
        "ski", "resort", "mountain", "snow",
    }

    # OOD conversational patterns that caused false matches in your evidence
    _OOD_PATTERNS = [
        # "Where is my umbrella?"
        re.compile(r"\bwhere\s+(is|are)\s+my\b", re.IGNORECASE),
        # "Have you finished dinner?"
        re.compile(r"\bhave you finished\b", re.IGNORECASE),
        re.compile(r"\bdo you have (a )?(boyfriend|girlfriend)\b",
                   re.IGNORECASE),
        re.compile(r"\bhow old are you\b", re.IGNORECASE),
        # We'll route these to hotel_identity instead of random
        re.compile(r"\bwho are you\b", re.IGNORECASE),
        re.compile(r"\bdo you know who i am\b", re.IGNORECASE),
    ]

    # Domain hints: if absent, we apply stricter threshold
    _DOMAIN_HINTS = {
        "hotel", "sunshine", "room", "rooms", "suite", "check", "checkin", "checkout",
        "breakfast", "restaurant", "dining", "wifi", "internet", "pool", "gym", "parking",
        "shuttle", "taxi", "luggage", "late", "early", "deposit", "payment",
        "reservation", "booking", "cancel", "cancellation", "smok", "smoking",
        "address", "location", "contact", "phone", "telephone", "email",
        "front", "desk",
    }

    def __init__(self, faq_path: str = "faq.json"):
        self.faq_path = str(faq_path)
        self.ready: bool = False
        self.error: Optional[str] = None

        self.items: List[_FAQItem] = []
        self._bm25: Optional[_BM25] = None

        self._load_and_index()

    @property
    def is_ready(self) -> bool:
        return bool(self.ready)

    @property
    def faq_count(self) -> int:
        return len(self.items)

    def _load_and_index(self) -> None:
        try:
            data = json.loads(Path(self.faq_path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("faq.json must be a list of {id,text}")

            items: List[_FAQItem] = []
            docs_tokens: List[List[str]] = []

            for obj in data:
                if not isinstance(obj, dict):
                    continue
                faq_id = str(obj.get("id", "")).strip()
                text = str(obj.get("text", "")).strip()
                if not faq_id or not text:
                    continue

                phrase = faq_id
                doc_text = self._build_doc_text(faq_id, text)
                items.append(_FAQItem(faq_id=faq_id, text=text,
                             phrase=phrase, doc_text=doc_text))
                docs_tokens.append(_tokens(doc_text))

            self.items = items
            self._bm25 = _BM25(docs_tokens)
            self.ready = True
            self.error = None

        except Exception as e:
            self.ready = False
            self.error = f"{type(e).__name__}: {e}"
            self.items = []
            self._bm25 = None

    # ----------------------------
    # Alias expansion (doc-side)
    # ----------------------------
    def _aliases_for_id(self, faq_id: str) -> List[str]:
        fid = _norm(faq_id)
        aliases = {fid}

        # common separator variants
        aliases.add(fid.replace("  ", " ").strip())

        # check-in/out variants
        if "checkin" in fid or "check in" in fid:
            aliases.update({"check in", "check-in", "checkin"})
        if "checkout" in fid or "check out" in fid:
            aliases.update({"check out", "check-out", "checkout"})

        # fee/price/cost variants
        if "fee" in fid or "price" in fid or "cost" in fid:
            aliases.update({"fee", "price", "cost", "charges", "charge"})

        # contact variants
        if "contact" in fid or "phone" in fid or "telephone" in fid or "email" in fid:
            aliases.update({"contact number", "phone number",
                           "telephone", "contact phone", "contact email"})

        # identity/name
        if faq_id == "hotel_identity":
            aliases.update({
                "hotel name", "name of the hotel", "hotel identity", "about the hotel",
                "what hotel is this", "property name", "sunshine hotel singapore",
                "who are you", "what are you", "why are you answering",
            })

        # address/location
        if "address" in fid or "location" in fid:
            aliases.update({
                "hotel address", "hotel location", "where is the hotel",
                "where is sunshine hotel", "where is sunshine hotel singapore",
                "how to get to the hotel", "directions",
            })

        # hours singular/plural reinforcement
        if "hours" in fid or "hour" in fid:
            aliases.update({"hour", "hours", "opening hours", "open hours"})

        return sorted(a for a in aliases if a)

    def _build_doc_text(self, faq_id: str, text: str) -> str:
        aliases = self._aliases_for_id(faq_id)
        # Index: original id + aliases + answer text
        return " | ".join([faq_id] + aliases + [text])

    # ----------------------------
    # Query expansion (query-side)
    # ----------------------------
    def _expand_query(self, query: str) -> List[str]:
        q0 = _norm(query)
        if not q0:
            return [""]

        ex = {q0}

        # check-in/out variants
        if "check in" in q0 or "checkin" in q0:
            ex.add(q0.replace("check in", "checkin"))
            ex.add(q0.replace("checkin", "check in"))
        if "check out" in q0 or "checkout" in q0:
            ex.add(q0.replace("check out", "checkout"))
            ex.add(q0.replace("checkout", "check out"))

        # contact/phone
        if "phone" in q0 or "contact" in q0 or "telephone" in q0:
            ex.add(q0.replace("phone", "contact number"))
            ex.add(q0.replace("contact", "phone"))
            ex.add(q0.replace("telephone", "phone"))

        # address/location
        if "address" in q0 or "location" in q0 or "located" in q0 or "where is" in q0:
            ex.add(q0.replace("address", "location"))
            ex.add(q0.replace("location", "address"))

        # identity/name
        if ("hotel" in q0) and ("name" in q0 or "identity" in q0):
            ex.add(q0.replace("name", "identity"))
            ex.add(q0.replace("identity", "name"))

        # plural/singular helpers (hours/hour, methods/method, rates/rate)
        ex.add(q0.replace("hours", "hour"))
        ex.add(q0.replace("hour", "hours"))
        ex.add(q0.replace("methods", "method"))
        ex.add(q0.replace("method", "methods"))
        ex.add(q0.replace("rates", "rate"))
        ex.add(q0.replace("rate", "rates"))

        # smoking/smoke
        ex.add(q0.replace("smoking", "smoke"))
        ex.add(q0.replace("smoke", "smoking"))

        # soundproofing/soundproof
        ex.add(q0.replace("soundproofing", "soundproof"))
        ex.add(q0.replace("soundproof", "soundproofing"))

        return sorted(s for s in ex if s)

    # ----------------------------
    # Guardrails / routing
    # ----------------------------
    def _is_goodbye(self, query: str) -> bool:
        q = _norm(query)
        if not q:
            return False
        return bool(self._GOODBYE_RE.search(q))

    def _is_too_short(self, query: str) -> bool:
        q = _norm(query)
        if len(q) < 3:
            return True
        # single token noises: "afggg"
        return len(_tokens(q)) < 2

    def _is_ood(self, query: str) -> bool:
        q = _norm(query)

        for pat in self._OOD_PATTERNS:
            if pat.search(q):
                # Note: "who are you" is not treated as OOD; we route to hotel_identity.
                if "who are you" in q or "do you know who i am" in q:
                    return False
                return True

        for kw in self._OOD_KEYWORDS:
            if kw in q:
                # again: allow "who are you" to route to hotel_identity
                if "who are you" in q:
                    return False
                return True

        return False

    def _has_domain_hint(self, query: str) -> bool:
        q = _norm(query)
        return any(h in q for h in self._DOMAIN_HINTS)

    def _route_special(self, query: str) -> Optional[str]:
        """
        Special routing:
        - "who are you" -> hotel_identity
        This prevents random matches like wifi_password for "who are you?" (seen in your evidence).
        """
        q = _norm(query)
        if "who are you" in q or "why are you answering" in q:
            return "hotel_identity"
        return None

    def _answer_by_id(self, faq_id: str, request_id: str, latency_ms: int) -> Dict[str, Any]:
        for it in self.items:
            if it.faq_id == faq_id:
                return {
                    "request_id": request_id,
                    "matched": True,
                    "answer": it.text,
                    "best_score": 1.0,
                    "best_id": it.faq_id,
                    "route": "rule_route",
                    "latency_ms": latency_ms,
                    "top": [{"id": it.faq_id, "question": it.phrase, "score": 1.0}],
                }
        # If not found, fallback
        return {
            "request_id": request_id,
            "matched": False,
            "answer": self.DEFAULT_FALLBACK,
            "best_score": 0.0,
            "best_id": "",
            "route": "rule_route_missing",
            "latency_ms": latency_ms,
            "top": [],
        }

    # ----------------------------
    # Public API
    # ----------------------------
    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35, request_id: str = "") -> Dict[str, Any]:
        t0 = time.time()
        rid = request_id or str(uuid.uuid4())

        if not self.ready or not self._bm25 or not self.items:
            return {
                "request_id": rid,
                "matched": False,
                "answer": "Sorry, the FAQ knowledge base is currently unavailable.",
                "best_score": 0.0,
                "best_id": "",
                "route": "degraded_no_index",
                "latency_ms": int((time.time() - t0) * 1000),
                "top": [],
            }

        # 1) Goodbye should behave consistently (frontend + backend)
        if self._is_goodbye(query):
            return {
                "request_id": rid,
                "matched": False,
                "answer": self.GOODBYE_TEXT,
                "best_score": 0.0,
                "best_id": "",
                "route": "goodbye",
                "latency_ms": int((time.time() - t0) * 1000),
                "top": [],
            }

        # 2) Special routing for "who are you" -> hotel_identity
        forced = self._route_special(query)
        if forced:
            return self._answer_by_id(forced, rid, int((time.time() - t0) * 1000))

        # 3) Guardrails
        if self._is_too_short(query):
            return {
                "request_id": rid,
                "matched": False,
                "answer": self.DEFAULT_FALLBACK,
                "best_score": 0.0,
                "best_id": "",
                "route": "guardrail_too_short",
                "latency_ms": int((time.time() - t0) * 1000),
                "top": [],
            }

        if self._is_ood(query):
            return {
                "request_id": rid,
                "matched": False,
                "answer": self.DEFAULT_FALLBACK,
                "best_score": 0.0,
                "best_id": "",
                "route": "guardrail_ood",
                "latency_ms": int((time.time() - t0) * 1000),
                "top": [],
            }

        # 4) BM25 over query expansions — keep best
        expansions = self._expand_query(query)
        best_raw_scores: Optional[List[float]] = None
        best_q: Optional[str] = None
        best_raw_max = -1.0

        for qx in expansions:
            q_tokens = _tokens(qx)
            raw_scores = self._bm25.score(q_tokens)
            raw_max = max(raw_scores) if raw_scores else 0.0
            if raw_max > best_raw_max:
                best_raw_max = raw_max
                best_raw_scores = raw_scores
                best_q = qx

        if not best_raw_scores or best_raw_max <= 0:
            return {
                "request_id": rid,
                "matched": False,
                "answer": self.DEFAULT_FALLBACK,
                "best_score": 0.0,
                "best_id": "",
                "route": "not_found",
                "latency_ms": int((time.time() - t0) * 1000),
                "top": [],
            }

        # 5) Proper confidence score (fixes “everything becomes 1.0” bug)
        # Use bounded transform + margin to 2nd best
        idx_sorted = sorted(range(len(best_raw_scores)),
                            key=lambda i: best_raw_scores[i], reverse=True)
        best_idx = idx_sorted[0]
        best_raw = best_raw_scores[best_idx]
        second_raw = best_raw_scores[idx_sorted[1]] if len(
            idx_sorted) > 1 else 0.0

        K = 6.0  # smoothing constant
        best_norm = best_raw / (best_raw + K) if best_raw > 0 else 0.0
        gap = (best_raw - second_raw) / \
            (best_raw + 1e-9) if best_raw > 0 else 0.0
        gap = max(0.0, min(1.0, gap))
        conf = 0.75 * best_norm + 0.25 * gap

        # 6) Domain gating: stricter threshold if query doesn't look hotel-related
        effective_min = float(min_score)
        if not self._has_domain_hint(query):
            effective_min = min(0.85, effective_min + 0.20)

        matched = conf >= effective_min

        # 7) Build top list with normalized scores (bounded)
        top_k = max(1, min(int(top_k or 5), 10))
        top_idxs = idx_sorted[:top_k]
        top = []
        for i in top_idxs:
            s = best_raw_scores[i]
            s_norm = s / (s + K) if s > 0 else 0.0
            it = self.items[i]
            top.append({"id": it.faq_id, "question": it.phrase,
                       "score": round(float(s_norm), 4)})

        best_item = self.items[best_idx]
        answer_text = best_item.text if matched else self.DEFAULT_FALLBACK
        route = "bm25" if matched else "not_found"

        return {
            "request_id": rid,
            "matched": bool(matched),
            "answer": answer_text,
            "best_score": round(float(conf), 4),
            "best_id": best_item.faq_id if matched else "",
            "route": route,
            "latency_ms": int((time.time() - t0) * 1000),
            "top": top,
        }
