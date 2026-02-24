# faq_retriever.py
# Sunshine Hotel Voice Agent - FAQ Retriever (BM25 + lightweight guardrails)
# MERGED PRODUCTION VERSION (RESTORE OLD ACCURACY + KEEP CLOUD RUN STABILITY):
# - Keep OLD doc expansion, normalization, tie-break boosts (high accuracy) :contentReference[oaicite:2]{index=2}
# - Add Cloud Run safe loading (no crash if faq.json missing)
# - Add NumPy-safe BM25 score handling (tolist + len checks)

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------- Optional dependency: rank_bm25 ----------
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore


# ---------- Small fallback BM25 (if rank_bm25 is not installed) ----------
class _FallbackBM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(1, self.N)

        df = {}
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1

        self.idf = {}
        for t, n in df.items():
            self.idf[t] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))

        self.tf = []
        for doc in corpus_tokens:
            d = {}
            for t in doc:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for i, doc in enumerate(self.corpus_tokens):
            doc_len = len(doc)
            tf = self.tf[i]
            denom_norm = self.k1 * \
                (1 - self.b + self.b * (doc_len / max(1e-9, self.avgdl)))
            score = 0.0
            for t in query_tokens:
                if t not in tf:
                    continue
                f = tf[t]
                idf = self.idf.get(t, 0.0)
                score += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = score
        return scores


@dataclass
class FAQEntry:
    id: str
    text: str
    doc: str  # expanded doc string for BM25


class FAQRetriever:
    VERSION = "2026.02.25-merged-prod"

    def __init__(self, faq_path: str = "faq.json"):
        self.faq_path = faq_path
        self.entries: List[FAQEntry] = []
        self.faq_count: int = 0
        self.is_ready: bool = False

        self._id_to_index: Dict[str, int] = {}
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []

        self._load()

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35, request_id: str = "") -> Dict[str, Any]:
        """
        Contract:
        {
          request_id, matched, answer, best_score, best_id, route, latency_ms, top:[{id, question, score}]
        }
        best_score is normalized to [0,1].
        """
        t0 = time.time()
        rid = request_id or uuid.uuid4().hex

        q_raw = (query or "").strip()
        q_norm = self._normalize_query(q_raw)

        if not self.is_ready:
            return self._resp(
                rid, False,
                "Sorry, the FAQ knowledge base is currently unavailable.",
                0.0, "", "degraded_no_index",
                int((time.time() - t0) * 1000), []
            )

        # Guardrail: Goodbye
        if self._is_goodbye(q_norm):
            return self._resp(
                rid, True,
                "Thank you. Goodbye!",
                1.0, "goodbye", "goodbye",
                int((time.time() - t0) * 1000), []
            )

        # Direct ID match (stable for automated tests)
        direct_id = self._direct_id_match(q_raw)
        if direct_id:
            entry = self.entries[self._id_to_index[direct_id]]
            return self._resp(
                rid, True,
                entry.text,
                1.0, entry.id, "direct_id",
                int((time.time() - t0) * 1000),
                [{"id": entry.id, "question": entry.id, "score": 1.0}],
            )

        # Guardrails: possessive + obvious OOD topics
        forced = self._guardrail_out_of_domain(q_norm)
        if forced is not None:
            return self._resp(
                rid, False,
                forced,
                0.0, "", "not_found_guardrail",
                int((time.time() - t0) * 1000), []
            )

        # BM25 retrieval
        q_tokens = self._tokenize(q_norm)

        raw_scores = self._bm25.get_scores(
            q_tokens) if self._bm25 is not None else None

        if raw_scores is None:
            return self._resp(
                rid, False,
                "Sorry, the FAQ knowledge base is currently unavailable.",
                0.0, "", "degraded_no_index",
                int((time.time() - t0) * 1000), []
            )

        # ---- Cloud/NumPy safety: convert to a plain python list ----
        # rank_bm25 often returns a numpy array; never do `if not raw_scores` on numpy.
        try:
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            else:
                raw_scores = list(raw_scores)
        except Exception:
            raw_scores = []

        if len(raw_scores) == 0:
            return self._resp(
                rid, False,
                "Sorry, the FAQ knowledge base is currently unavailable.",
                0.0, "", "degraded_no_index",
                int((time.time() - t0) * 1000), []
            )

        # Normalize to [0,1]
        max_raw = max(raw_scores) if raw_scores else 0.0
        norm_scores = [(s / max_raw) if max_raw >
                       0 else 0.0 for s in raw_scores]

        # Apply boosts
        boosted_scores = self._apply_boosts(q_norm, norm_scores)

        ranked = sorted(range(len(boosted_scores)),
                        key=lambda i: boosted_scores[i], reverse=True)
        ranked = ranked[: max(1, int(top_k))]

        top = [{"id": self.entries[i].id, "question": self.entries[i].id,
                "score": round(boosted_scores[i], 4)} for i in ranked]

        best_i = ranked[0]
        best_id = self.entries[best_i].id
        best_score = float(boosted_scores[best_i])

        matched = best_score >= float(min_score)

        # WH-question OOD guardrail (only when not matched)
        if (not matched) and self._wh_ood_guardrail(q_norm):
            return self._resp(
                rid, False,
                "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                best_score, best_id, "not_found_wh_guardrail",
                int((time.time() - t0) * 1000), top
            )

        if not matched:
            return self._resp(
                rid, False,
                "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                best_score, best_id, "not_found",
                int((time.time() - t0) * 1000), top
            )

        return self._resp(
            rid, True,
            self.entries[best_i].text,
            best_score, best_id, "bm25",
            int((time.time() - t0) * 1000), top
        )

    def _load(self) -> None:
        # Cloud Run: do NOT crash if faq.json missing
        if not os.path.exists(self.faq_path):
            print(f"[FAQRetriever] faq.json not found at: {self.faq_path}")
            self.entries = []
            self.faq_count = 0
            self._id_to_index = {}
            self._bm25 = None
            self._corpus_tokens = []
            self.is_ready = False
            return

        try:
            with open(self.faq_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(
                    "faq.json must be a list of {id,text} objects")

            entries: List[FAQEntry] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                fid = str(item.get("id", "")).strip()
                txt = str(item.get("text", "")).strip()
                if not fid or not txt:
                    continue
                doc = self._build_doc(fid, txt)
                entries.append(FAQEntry(id=fid, text=txt, doc=doc))

            self.entries = entries
            self.faq_count = len(entries)
            self._id_to_index = {e.id: i for i, e in enumerate(entries)}
            self._corpus_tokens = [self._tokenize(e.doc) for e in entries]

            if BM25Okapi is not None:
                self._bm25 = BM25Okapi(self._corpus_tokens)
            else:
                self._bm25 = _FallbackBM25(self._corpus_tokens)

            self.is_ready = True
            print(f"[FAQRetriever] FAQ loaded: {self.faq_count} entries")

        except Exception as e:
            print("[FAQRetriever] Failed loading FAQ:", repr(e))
            self.entries = []
            self.faq_count = 0
            self._id_to_index = {}
            self._bm25 = None
            self._corpus_tokens = []
            self.is_ready = False

    def _build_doc(self, fid: str, txt: str) -> str:
        base = fid.lower().strip()
        tokens = base.split("_")
        id_space = " ".join(tokens)
        id_hyphen = "-".join(tokens)

        extra = []

        if "policy" in tokens:
            extra += ["policy", "allowed", "can", "may",
                      "rules", "is it possible", "permitted"]
        if any(x in tokens for x in ["fee", "price", "charge", "cost"]):
            extra += ["fee", "price", "cost", "charge", "how much", "rates"]
        if "hours" in tokens or "time" in tokens:
            extra += ["hours", "time", "when", "open", "close", "what time"]
        if "phone" in tokens:
            extra += ["phone", "call", "telephone",
                      "contact number", "hotline"]
        if "email" in tokens:
            extra += ["email", "mail", "contact email", "@"]

        # Identity/name/location helpers
        if fid in {"hotel_identity", "hotel_name", "hotel"}:
            extra += ["hotel name", "hotel identity", "which hotel",
                      "what hotel", "sunshine hotel singapore"]
        if fid in {"address", "location", "hotel_address"}:
            extra += ["address", "location",
                      "where is the hotel", "how to get there"]

        doc = f"{base} {id_space} {id_hyphen} " + \
            " ".join(extra) + " " + txt.lower()
        return doc

    def _normalize_query(self, q: str) -> str:
        q = (q or "").strip().lower()
        q = q.replace("’", "'")
        q = re.sub(r"[\u200b\u200c\u200d]", "", q)
        q = re.sub(r"[^a-z0-9@\+\-\s'_]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()

        q = re.sub(r"\bcheck\s*-\s*in\b", "checkin", q)
        q = re.sub(r"\bcheck\s+in\b", "checkin", q)
        q = re.sub(r"\bcheck\s*-\s*out\b", "checkout", q)
        q = re.sub(r"\bcheck\s+out\b", "checkout", q)

        return q

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9@\+_ ]", " ", text)
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        toks = [t for t in text.split(" ") if t]

        out = []
        for t in toks:
            if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                out.append(t[:-1])
            out.append(t)
        return out

    def _direct_id_match(self, q_raw: str) -> Optional[str]:
        q = (q_raw or "").strip().lower()
        q = q.replace("?", "").replace(".", "").strip()
        q = q.replace("-", "_").replace(" ", "_")
        q = re.sub(r"_+", "_", q)
        if q in self._id_to_index:
            return q
        return None

    def _is_goodbye(self, q_norm: str) -> bool:
        return bool(re.search(r"\b(bye|goodbye|stop|end session|exit)\b", q_norm or ""))

    def _guardrail_out_of_domain(self, q_norm: str) -> Optional[str]:
        if not q_norm:
            return "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."

        # possessive "my ___" personal-property pattern
        if re.search(r"\bmy\s+(book|bicycle|bike|umbrella|watch|phone|wallet|passport|laptop|bag|camera|keys?)\b", q_norm):
            return "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."

        # product/warranty topics
        if re.search(r"\b(warranty|this watch|refund for this|this product)\b", q_norm):
            return "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."

        # off-topic geography
        if re.search(r"\b(ski resort)\b", q_norm):
            return "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."

        # alcohol questions unless explicitly about minibar
        if re.search(r"\b(beer|alcohol|wine|whisky|vodka|champagne)\b", q_norm) and "minibar" not in q_norm:
            return "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."

        return None

    def _wh_ood_guardrail(self, q_norm: str) -> bool:
        if not re.match(r"^(what|where|when|how|why|which)\b", q_norm or ""):
            return False
        return bool(re.search(r"\b(warranty|watch|book|bicycle|bike|ski resort)\b", q_norm or ""))

    def _apply_boosts(self, q_norm: str, scores: List[float]) -> List[float]:
        q = q_norm

        want_fee = bool(
            re.search(r"\b(fee|price|cost|charge|how much|rates?)\b", q))
        want_policy = bool(
            re.search(r"\b(policy|allowed|can i|is it possible|permit|rule)\b", q))
        want_time = bool(
            re.search(r"\b(when|time|hour|open|close|starts?)\b", q))
        want_phone = bool(
            re.search(r"\b(phone|call|telephone|hotline|number)\b", q)) or ("+" in q)
        want_email = ("@" in q) or bool(re.search(r"\b(email|mail)\b", q))

        boosted = scores[:]

        def bump(fid: str, delta: float):
            i = self._id_to_index.get(fid)
            if i is not None:
                boosted[i] = max(0.0, min(1.0, boosted[i] + delta))

        # early check-in: fee vs policy vs time
        if "early" in q and "checkin" in q:
            if want_fee:
                bump("early_checkin_fee", 0.12)
                bump("early_checkin_policy", 0.06)
                bump("checkin_time", -0.05)
            elif want_policy:
                bump("early_checkin_policy", 0.12)
                bump("early_checkin_fee", 0.06)
                bump("checkin_time", -0.05)
            elif want_time:
                bump("checkin_time", 0.12)
                bump("early_checkin_policy", 0.04)

        # late checkout: fee vs policy
        if "late" in q and "checkout" in q:
            if want_fee:
                bump("late_checkout_fee", 0.12)
                bump("late_checkout_policy", 0.06)
            elif want_policy:
                bump("late_checkout_policy", 0.12)
                bump("late_checkout_fee", 0.06)

        # contact info tie-break
        if want_phone:
            bump("contact_phone", 0.12)
            bump("contact_email", -0.05)
        if want_email:
            bump("contact_email", 0.12)
            bump("contact_phone", -0.05)

        # hotel name/identity
        if re.search(r"\b(hotel name|name of the hotel|identity of the hotel|which hotel)\b", q):
            bump("hotel_identity", 0.10)

        # address/location
        if re.search(r"\b(address|location|where is the hotel)\b", q):
            bump("address", 0.10)

        return boosted

    def _resp(
        self,
        request_id: str,
        matched: bool,
        answer: str,
        best_score: float,
        best_id: str,
        route: str,
        latency_ms: int,
        top: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "matched": bool(matched),
            "answer": answer,
            "best_score": float(round(best_score, 4)),
            "best_id": best_id,
            "route": route,
            "latency_ms": int(latency_ms),
            "top": top,
        }
