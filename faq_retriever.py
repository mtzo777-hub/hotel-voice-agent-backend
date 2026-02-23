# faq_retriever.py
"""
Sunshine Hotel Voice Agent - FAQ Retriever (BM25 + lightweight synonym/alias expansion)

Goals (exam-focused):
- Keep *existing* in-domain queries working (IDs in faq.json).
- Improve Not-Found contract / safe fallback for out-of-domain queries.
- Avoid external BM25 dependencies (no rank_bm25) to keep Cloud Run builds stable.
- Provide a stable response contract used by both Swagger (/docs) and the GitHub Pages frontend.

Response contract returned by FAQRetriever.answer():
{
  "request_id": str,
  "matched": bool,
  "answer": str,
  "best_score": float,     # 0..1 normalized confidence-like score
  "best_id": str,
  "route": str,            # bm25 | not_found | guardrail_* | degraded_*
  "latency_ms": int,
  "top": [{"id": str, "question": str, "score": float}, ...]   # scores 0..1
}
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


# ----------------------------
# Text normalization utilities
# ----------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    # Keep "in"/"out" tokens (important for check-in vs check-out).
    # We do NOT aggressively remove stopwords because the FAQ corpus is small
    # and we want to preserve discriminative words.
    s = _norm(s)
    return _WORD_RE.findall(s)


# ----------------------------
# BM25 implementation (no deps)
# ----------------------------
class _BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = float(k1)
        self.b = float(b)

        self.doc_count = len(docs)
        self.avgdl = (sum(len(d) for d in docs) /
                      self.doc_count) if self.doc_count else 0.0

        # document frequencies
        df: Dict[str, int] = {}
        for d in docs:
            seen = set(d)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.df = df

        # idf with BM25+ style smoothing
        self.idf: Dict[str, float] = {}
        for t, n_q in df.items():
            self.idf[t] = math.log(
                1 + (self.doc_count - n_q + 0.5) / (n_q + 0.5))

        # term frequencies per doc
        self.tf: List[Dict[str, int]] = []
        for d in docs:
            m: Dict[str, int] = {}
            for t in d:
                m[t] = m.get(t, 0) + 1
            self.tf.append(m)

    def score(self, query_tokens: List[str]) -> List[float]:
        if not self.docs:
            return []
        if not query_tokens:
            return [0.0] * len(self.docs)

        scores = [0.0] * len(self.docs)
        for qi in query_tokens:
            idf = self.idf.get(qi, 0.0)
            if idf <= 0:
                continue
            for i, d in enumerate(self.docs):
                f = self.tf[i].get(qi, 0)
                if f == 0:
                    continue
                dl = len(d)
                denom = f + self.k1 * \
                    (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return scores


@dataclass
class _FAQItem:
    faq_id: str
    text: str
    phrase: str  # human-ish "question" label for top-k display
    doc_text: str  # indexed text used by BM25


# ----------------------------
# Retriever
# ----------------------------
class FAQRetriever:
    DEFAULT_FALLBACK = "Sorry, I don't have that information yet. Please contact the hotel reception for assistance."

    # Out-of-domain hints (keep small + high precision)
    _OOD_HINTS = {
        "warranty", "watch", "bicycle", "book", "beer", "alcohol", "wine", "vodka", "whisky",
        "iphone", "android", "laptop", "printer", "refund", "amazon", "lazada", "shopee",
        "baggage claim", "airline", "flight", "airport terminal", "passport renewal",
        "ski", "resort", "mountain", "snow",
        "car insurance", "motorcycle", "loan", "bank", "credit card bill",
    }

    # In-domain hints (hotel context). If absent, we apply a stricter threshold.
    _DOMAIN_HINTS = {
        "hotel", "sunshine", "room", "rooms", "suite", "checkin", "check-in", "check in",
        "checkout", "check-out", "check out",
        "breakfast", "restaurant", "dining", "wifi", "internet", "pool", "gym", "parking",
        "shuttle", "taxi", "luggage", "late", "early", "deposit", "payment",
        "reservation", "booking", "cancel", "cancellation", "smoking", "pets",
        "address", "location", "contact", "phone", "telephone", "email",
    }

    def __init__(self, faq_path: str = "faq.json"):
        self.faq_path = str(faq_path)
        self.ready: bool = False
        self.error: Optional[str] = None

        self.items: List[_FAQItem] = []
        self._bm25: Optional[_BM25] = None
        self._docs_tokens: List[List[str]] = []

        self._load_and_index()

    # Compatibility with older code / tests
    @property
    def is_ready(self) -> bool:
        return bool(self.ready)

    @property
    def faq_count(self) -> int:
        return len(self.items)

    def _load_and_index(self) -> None:
        try:
            p = Path(self.faq_path)
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(
                    "faq.json must be a list of {id,text} objects")

            items: List[_FAQItem] = []
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                faq_id = str(obj.get("id", "")).strip()
                text = str(obj.get("text", "")).strip()
                if not faq_id or not text:
                    continue

                # display label (can be improved later if you add 'question' field)
                phrase = faq_id
                doc_text = self._build_doc_text(faq_id, text)
                items.append(_FAQItem(faq_id=faq_id, text=text,
                             phrase=phrase, doc_text=doc_text))

            self.items = items
            self._docs_tokens = [_tokens(it.doc_text) for it in self.items]
            self._bm25 = _BM25(self._docs_tokens)
            self.ready = True
            self.error = None
        except Exception as e:
            self.ready = False
            self.error = f"{type(e).__name__}: {e}"
            self.items = []
            self._docs_tokens = []
            self._bm25 = None

    # ----------------------------
    # Alias expansion (doc side)
    # ----------------------------
    def _aliases_for_id(self, faq_id: str) -> List[str]:
        fid = _norm(faq_id)

        # base variants
        aliases = {fid, fid.replace("  ", " ").strip()}

        # check-in / check-out family
        if "checkin" in fid or "check in" in fid:
            aliases.update({"check in", "check-in", "checkin"})
        if "checkout" in fid or "check out" in fid:
            aliases.update({"check out", "check-out", "checkout"})

        # common business synonyms
        def add_pair(a: str, b: str) -> None:
            if a in fid or b in fid:
                aliases.update({a, b})

        add_pair("fee", "price")
        add_pair("fee", "cost")
        add_pair("contact number", "phone number")
        add_pair("telephone", "phone")
        add_pair("location", "address")
        add_pair("identity", "name")

        # High-impact special cases
        if faq_id == "hotel_identity":
            aliases.update({
                "hotel name", "name of the hotel", "hotel identity", "about the hotel",
                "what hotel is this", "property name", "sunshine hotel singapore",
            })

        # If you have address-like IDs, help match location/address phrasing
        if "address" in fid or "location" in fid:
            aliases.update({
                "hotel address", "hotel location", "where is the hotel", "where is sunshine hotel",
                "where is sunshine hotel singapore", "how to get to the hotel",
            })

        return sorted(a for a in aliases if a)

    def _build_doc_text(self, faq_id: str, text: str) -> str:
        # We index:
        # - ID tokens (underscores/hyphens)
        # - generated alias phrases
        # - the actual answer text
        aliases = self._aliases_for_id(faq_id)
        parts = [faq_id] + aliases + [text]
        return " | ".join(p for p in parts if p)

    # ----------------------------
    # Query expansion (query side)
    # ----------------------------
    def _expand_query(self, query: str) -> List[str]:
        q0 = _norm(query)

        # Fast guard for totally empty
        if not q0:
            return [q0]

        expansions = {q0}

        # normalize check in/out forms
        if "check in" in q0 or "check-in" in q0 or "checkin" in q0:
            expansions.update({q0.replace("check in", "checkin"), q0.replace(
                "check-in", "checkin"), q0.replace("checkin", "check in")})
        if "check out" in q0 or "check-out" in q0 or "checkout" in q0:
            expansions.update({q0.replace("check out", "checkout"), q0.replace(
                "check-out", "checkout"), q0.replace("checkout", "check out")})

        # phone/contact
        if "phone" in q0 or "telephone" in q0 or "contact" in q0:
            expansions.add(q0.replace("phone", "contact number"))
            expansions.add(q0.replace("telephone", "phone"))
            expansions.add(q0.replace("contact", "phone"))

        # fee/price/cost
        if "fee" in q0 or "price" in q0 or "cost" in q0 or "charge" in q0:
            expansions.add(q0.replace("price", "fee"))
            expansions.add(q0.replace("cost", "fee"))
            expansions.add(q0.replace("charge", "fee"))

        # name/identity
        if ("hotel" in q0) and ("name" in q0 or "identity" in q0 or "property" in q0):
            expansions.update({
                q0.replace("name", "identity"),
                q0.replace("identity", "name"),
                q0 + " sunshine hotel singapore",
                "hotel identity sunshine hotel singapore",
                "hotel name sunshine hotel singapore",
            })

        # address/location
        if ("hotel" in q0) and ("address" in q0 or "location" in q0 or "located" in q0 or "where" in q0):
            expansions.update({
                q0.replace("location", "address"),
                q0.replace("address", "location"),
                q0 + " sunshine hotel singapore",
                "hotel address sunshine hotel singapore",
                "where is sunshine hotel singapore located",
            })

        return sorted(expansions)

    # ----------------------------
    # Guardrails
    # ----------------------------
    def _is_too_short(self, query: str) -> bool:
        qt = _tokens(query)
        # one-token garbage like "afggg"
        return len(qt) < 2 or len(query.strip()) < 3

    def _looks_out_of_domain(self, query: str) -> bool:
        qn = _norm(query)
        # If it contains any strong OOD hints, treat as OOD.
        for w in self._OOD_HINTS:
            if w in qn:
                return True
        return False

    def _has_domain_hint(self, query: str) -> bool:
        qn = _norm(query)
        return any(h in qn for h in self._DOMAIN_HINTS)

    # ----------------------------
    # Public API
    # ----------------------------
    def answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
        request_id: str = "",
    ) -> Dict[str, Any]:
        t0 = time.time()
        rid = request_id or str(uuid.uuid4())

        if not self.ready or not self._bm25:
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

        if self._looks_out_of_domain(query):
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

        # Query expansions (lexical only)
        expanded = self._expand_query(query)

        # Evaluate each expansion; keep the best overall
        best_pack: Optional[Tuple[str, List[float]]] = None
        best_raw_max = -1.0

        for qx in expanded:
            q_tokens = _tokens(qx)
            raw_scores = self._bm25.score(q_tokens)
            raw_max = max(raw_scores) if raw_scores else 0.0
            if raw_max > best_raw_max:
                best_raw_max = raw_max
                best_pack = (qx, raw_scores)

        if not best_pack or best_raw_max <= 0:
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

        qx, raw_scores = best_pack

        # Normalize scores for reporting/thresholding to a 0..1 range
        raw_max = max(raw_scores) if raw_scores else 0.0
        norm_scores = [(s / raw_max) if raw_max >
                       0 else 0.0 for s in raw_scores]

        # Get top-k indices
        k = max(1, int(top_k))
        idxs = sorted(range(len(norm_scores)),
                      key=lambda i: norm_scores[i], reverse=True)[:k]

        best_idx = idxs[0]
        best_item = self.items[best_idx]
        best_score = float(norm_scores[best_idx])

        # Stricter threshold if query has no in-domain hints (reduces false positives)
        effective_min = float(min_score)
        if not self._has_domain_hint(query):
            effective_min = min(0.75, effective_min + 0.15)

        matched = best_score >= effective_min

        top_list = []
        for i in idxs:
            it = self.items[i]
            top_list.append({
                "id": it.faq_id,
                "question": it.phrase or it.faq_id,
                "score": round(float(norm_scores[i]), 4),
            })

        route = "bm25" if matched else "not_found"
        answer_text = best_item.text if matched else self.DEFAULT_FALLBACK

        return {
            "request_id": rid,
            "matched": bool(matched),
            "answer": answer_text,
            "best_score": round(float(best_score), 4),
            "best_id": best_item.faq_id if matched else "",
            "route": route,
            "latency_ms": int((time.time() - t0) * 1000),
            "top": top_list,
        }
