from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+")

# IMPORTANT: do NOT include "hotel" in stopwords — it matters for intent.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "at", "by",
    "is", "are", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "you", "your", "we", "our",
    "with", "as", "from", "into", "over", "under", "up", "down",
    "please", "kindly", "may", "might", "can", "could", "should", "would",
    "subject", "depends", "available", "availability",
    "contact", "front", "desk",  # common in many answers -> noisy
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _simple_stem(tok: str) -> str:
    # Very small stemmer to handle smoke/smoking, checkins, etc.
    if len(tok) > 4 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 3 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> List[str]:
    toks = _WORD_RE.findall(_norm(text))
    out = []
    for t in toks:
        if not t or t in _STOPWORDS:
            continue
        out.append(_simple_stem(t))
    return out


def _id_to_phrase(faq_id: str) -> str:
    s = (faq_id or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _query_normalize(q: str) -> str:
    qn = _norm(q)
    qn = qn.replace("wi-fi", "wifi")
    qn = qn.replace("check-in", "checkin").replace("check in", "checkin")
    qn = qn.replace("check-out", "checkout").replace("check out", "checkout")
    qn = qn.replace("e-mail", "email")
    # common typo from your log: "message service" -> "massage service"
    qn = re.sub(r"\bmessage service\b", "massage service", qn)
    return qn


# ---- INTENT ROUTER (the big fix) ----
# Map common user phrasing -> faq.id
_INTENT_RULES: List[Tuple[re.Pattern, str]] = [
    # hotel identity / name
    (re.compile(r"\b(hotel name|name of (the )?hotel|what'?s the hotel called)\b"), "hotel_identity"),
    # phone / contact number
    (re.compile(r"\b(contact number|phone number|contact phone|telephone|call you|contact by phone)\b"), "contact_phone"),
    # email
    (re.compile(r"\b(email|contact email|email address)\b"), "contact_email"),
    # address / location / where is the hotel
    (re.compile(r"\b(address|where is (the )?hotel|hotel location|how to get to (the )?hotel|directions)\b"), "address"),
    # smoking
    (re.compile(r"\b(smok|cigarette|vape)\b"), "smoking_policy"),
    # room capacity / occupancy
    (re.compile(r"\b(how many (people|person|persons)|max(imum)? occupancy|room capacity|how many guests)\b"), "room_capacity"),
    # age policy
    (re.compile(r"\b(age policy|minimum age|under 18|child policy)\b"), "age_policy"),
    # massage service
    (re.compile(r"\b(massage service|in-room massage|massage)\b"), "service_massage"),
]


@dataclass
class FAQItem:
    faq_id: str
    text: str
    phrase: str
    doc_tokens: List[str]


class BM25:
    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = (sum(len(d)
                      for d in tokenized_docs) / self.N) if self.N else 0.0

        df: Dict[str, int] = {}
        for doc in tokenized_docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1

        self.idf: Dict[str, float] = {}
        for t, freq in df.items():
            self.idf[t] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

        self.tf: List[Dict[str, int]] = []
        self.dl: List[int] = []
        for doc in tokenized_docs:
            m: Dict[str, int] = {}
            for t in doc:
                m[t] = m.get(t, 0) + 1
            self.tf.append(m)
            self.dl.append(len(doc))

    def score(self, query_tokens: List[str]) -> List[float]:
        if not self.N:
            return []
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.dl[i]
            denom_norm = self.k1 * \
                (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            tf_i = self.tf[i]
            s = 0.0
            for t in query_tokens:
                if t not in tf_i:
                    continue
                f = tf_i[t]
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        return scores


class FAQRetriever:
    def __init__(self, faq_json_path: Optional[str] = None) -> None:
        self.faq_json_path = faq_json_path or os.getenv(
            "FAQ_JSON_PATH", "/app/faq.json")
        self.items: List[FAQItem] = []
        self.by_id: Dict[str, FAQItem] = {}
        self._bm25: Optional[BM25] = None
        self.ready: bool = False
        self.error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            if not os.path.exists(self.faq_json_path):
                raise FileNotFoundError(
                    f"faq.json not found at {self.faq_json_path}")
            with open(self.faq_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("faq.json must be a JSON list of objects")

            items: List[FAQItem] = []
            by_id: Dict[str, FAQItem] = {}
            for obj in raw:
                if not isinstance(obj, dict):
                    continue
                faq_id = str(obj.get("id", "")).strip()
                text = str(obj.get("text", "")).strip()
                if not faq_id or not text:
                    continue
                phrase = _id_to_phrase(faq_id)

                # Weight ID/phrase heavily + include text
                doc = f"{faq_id} {phrase} {phrase} {phrase} {text}"
                doc_tokens = _tokens(doc)

                it = FAQItem(faq_id=faq_id, text=text,
                             phrase=phrase, doc_tokens=doc_tokens)
                items.append(it)
                by_id[faq_id.lower()] = it

            if not items:
                raise ValueError(
                    "faq.json loaded but contained 0 valid FAQ items")

            self.items = items
            self.by_id = by_id
            self._bm25 = BM25([it.doc_tokens for it in items])
            self.ready = True
            self.error = None
        except Exception as e:
            self.items = []
            self.by_id = {}
            self._bm25 = None
            self.ready = False
            self.error = str(e)

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": "intent_router_plus_bm25",
            "faq_json_path": self.faq_json_path,
            "faq_count": len(self.items),
            "error": self.error,
        }

    def _direct_id_match(self, query: str) -> Optional[FAQItem]:
        q = _norm(query)
        q_id_like = re.sub(r"[^a-z0-9_]+", "_", q).strip("_")
        return self.by_id.get(q) or self.by_id.get(q_id_like)

    def _intent_route(self, query: str) -> Optional[FAQItem]:
        qn = _query_normalize(query)
        for pat, faq_id in _INTENT_RULES:
            if pat.search(qn):
                return self.by_id.get(faq_id.lower())
        return None

    def _overlap(self, q_tokens: List[str], d_tokens: List[str]) -> float:
        qs = set(q_tokens)
        ds = set(d_tokens)
        if not qs or not ds:
            return 0.0
        return len(qs & ds) / (len(qs) + 0.5)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[FAQItem, float, float]]:
        if not self.ready or not self._bm25:
            return []
        if not query or not query.strip():
            return []

        # 1) Direct ID match
        direct = self._direct_id_match(query)
        if direct:
            return [(direct, 999.0, 999.0)]

        # 2) Intent routing (fixes “contact number vs contact phone”, “where is the hotel”, etc.)
        routed = self._intent_route(query)
        if routed:
            return [(routed, 998.0, 998.0)]

        # 3) BM25 fallback
        qn = _query_normalize(query)
        q_tokens = _tokens(qn)
        if not q_tokens:
            return []

        bm25_scores = self._bm25.score(q_tokens)
        if not bm25_scores:
            return []

        pool_k = max(12, top_k * 6)
        idxs = sorted(range(len(bm25_scores)),
                      key=lambda i: bm25_scores[i], reverse=True)[:pool_k]

        scored: List[Tuple[FAQItem, float, float]] = []
        for i in idxs:
            it = self.items[i]
            ov = self._overlap(q_tokens, it.doc_tokens)
            blended = bm25_scores[i] + (2.5 * ov)
            scored.append((it, float(bm25_scores[i]), float(blended)))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[: max(1, top_k)]

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
        """
        IMPORTANT CHANGE:
        - Remove the “ambiguous top1 vs top2 margin reject”.
          Your log shows correct top1 getting rejected as ambiguous. :contentReference[oaicite:2]{index=2}
        - Use overlap + min_score gate instead.
        """
        if not self.ready:
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": self.error or "not_ready",
            }

        results = self.search(query, top_k=top_k)
        if not results:
            return {
                "answer": "No match found.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": "no_results",
            }

        blended_scores = [r[2] for r in results]
        max_b = max(blended_scores) if blended_scores else 0.0
        norm = [(s / max_b) if max_b > 0 else 0.0 for s in blended_scores]

        best_item = results[0][0]
        best_score = float(norm[0])

        # Stricter threshold for super short queries
        q_len = len(_tokens(_query_normalize(query)))
        eff_min_score = min_score + (0.12 if q_len <= 2 else 0.0)

        matched = best_score >= eff_min_score

        top_list = []
        for (it, bm25_s, blended_s), ns in zip(results, norm):
            top_list.append({
                "id": it.faq_id,
                "question": it.phrase or it.faq_id,
                "score": round(float(ns), 4),
            })

        return {
            "answer": best_item.text if matched else "No match found.",
            "matched": matched,
            "best_score": round(best_score, 4),
            "top": top_list,
            "error": None if matched else f"best_score {best_score:.4f} < min_score {eff_min_score:.2f}",
        }
