# faq_retriever.py
"""
FAQ Retriever (BM25 lexical)
- Loads faq.json (list of {"id","question","answer"} OR {"id","text"}).
- Builds a lightweight BM25 index (no external dependency).
- Returns a stable JSON contract used by the frontend + Swagger UI.

Key goals:
1) Keep in-domain queries (present in faq.json) working reliably.
2) Strengthen "Not found" guardrails so out-of-domain random queries do NOT get a wrong hotel answer.
3) Provide debug-friendly metadata (route, best_score, best_id, top candidates).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -----------------------------
# Text utilities
# -----------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "is", "are", "am", "was", "were", "be", "been", "being",
    "i", "me", "my", "mine", "we", "our", "you", "your", "yours", "he", "she", "they", "them", "their",
    "do", "does", "did", "can", "could", "should", "would", "may", "might", "must",
    "to", "of", "in", "on", "at", "for", "from", "with", "as", "by", "about", "into", "over", "under",
    "what", "when", "where", "why", "how",
    "please", "thanks", "thank", "hi", "hello", "hey",
}

# Terms commonly associated with clearly out-of-domain queries that caused wrong answers.
# We use these to apply a stricter acceptance threshold.
_OUT_OF_DOMAIN_TERMS = {
    "beer", "wine", "alcohol",
    "warranty", "watch",
    "book", "bicycle", "bike",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase + normalize common variants."""
    s = (text or "").strip().lower()

    # unify check-in/check in, check-out/check out
    s = s.replace("check in", "check-in")
    s = s.replace("check out", "check-out")

    # unify WiFi variants
    s = s.replace("wi fi", "wifi")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    """Tokenize to alphanum words + remove stopwords."""
    s = normalize_text(text)
    toks = _WORD_RE.findall(s)
    toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def looks_like_gibberish(text: str) -> bool:
    """Heuristic for obvious nonsense strings."""
    s = (text or "").strip()
    if not s:
        return True
    # too short after trimming
    if len(s) <= 2:
        return True
    # very low alphabetic ratio
    alpha = sum(ch.isalpha() for ch in s)
    if alpha / max(1, len(s)) < 0.4:
        return True
    # long consonant runs (e.g., "afggg", "asdfghjkl")
    lower = s.lower()
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", lower):
        return True
    return False


# -----------------------------
# BM25 implementation
# -----------------------------

@dataclass
class FAQItem:
    id: str
    question: str
    answer: str


class FAQRetriever:
    """
    Lightweight BM25 retriever.

    Score normalization:
    - BM25 scores are unbounded.
    - For API stability, we normalize scores to [0, 1] per-query using:
        normalized = score / max_score  (if max_score > 0)
    """

    def __init__(self, faq_path: str = "faq.json"):
        self.faq_path = faq_path
        self.items: List[FAQItem] = []

        # Index
        self._docs_terms: List[List[str]] = []
        self._docs_term_sets: List[set[str]] = []
        self._df: Dict[str, int] = {}
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._N: int = 0

        self.is_ready: bool = False
        self.faq_count: int = 0
        self.load()

    def load(self) -> None:
        """Load faq.json and build index."""
        # Resolve relative path for Cloud Run: keep it next to the app working directory.
        path = self.faq_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items: List[FAQItem] = []
        for row in raw:
            _id = str(row.get("id", "")).strip() or "unknown"
            q = (row.get("question") or row.get("q") or "").strip()
            a = (row.get("answer") or row.get("text") or "").strip()
            if not q and "id" in row:
                # some datasets store the canonical question under id; keep a best-effort fallback
                q = str(row["id"]).strip()
            if not a:
                # ensure non-empty answer to avoid blank TTS
                a = "Sorry, I don't have that information yet. Please contact the hotel reception for assistance."
            items.append(FAQItem(id=_id, question=q, answer=a))

        self.items = items
        self.faq_count = len(items)

        # Build BM25 structures
        self._docs_terms = [tokenize(it.question) for it in items]
        self._docs_term_sets = [set(terms) for terms in self._docs_terms]
        self._doc_len = [len(t) for t in self._docs_terms]
        self._N = len(self._docs_terms)
        self._avgdl = (sum(self._doc_len) / self._N) if self._N else 0.0

        self._df = {}
        for terms in self._docs_term_sets:
            for t in terms:
                self._df[t] = self._df.get(t, 0) + 1

        self.is_ready = True

    def _idf(self, term: str) -> float:
        # Standard BM25 idf with +1 smoothing
        df = self._df.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_scores(self, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        if not self._N:
            return []
        scores = [0.0] * self._N

        # term frequency per doc is computed on the fly (docs are short)
        for term in query_terms:
            idf = self._idf(term)
            for i, doc_terms in enumerate(self._docs_terms):
                tf = doc_terms.count(term)
                if tf == 0:
                    continue
                dl = self._doc_len[i] or 1
                denom = tf + k1 * (1 - b + b * (dl / (self._avgdl or 1.0)))
                scores[i] += idf * (tf * (k1 + 1)) / denom

        return scores

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return ranked candidates with *raw* scores."""
        q_terms = tokenize(query)
        scores = self._bm25_scores(q_terms)

        if not scores:
            return []

        idx_scores = list(enumerate(scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for i, s in idx_scores[: max(1, int(top_k))]:
            it = self.items[i]
            results.append(
                {
                    "id": it.id,
                    "question": it.question,
                    "answer": it.answer,
                    "score": float(s),
                }
            )
        return results

    def answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main retrieval method. Returns a stable JSON contract:
        {
          request_id, matched, answer, best_score, best_id, route, latency_ms, top:[...]
        }
        """
        t0 = time.perf_counter()

        # Fast "gibberish" guardrail to avoid random matches.
        if looks_like_gibberish(query):
            return {
                "request_id": request_id or "",
                "matched": False,
                "answer": "Sorry, I didn’t catch that. Please ask a hotel question.",
                "best_score": 0.0,
                "best_id": "",
                "route": "guardrail_gibberish",
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "top": [],
            }

        candidates = self.search(query=query, top_k=top_k)
        if not candidates:
            return {
                "request_id": request_id or "",
                "matched": False,
                "answer": "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                "best_score": 0.0,
                "best_id": "",
                "route": "not_found_empty",
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "top": [],
            }

        max_raw = max(c["score"] for c in candidates) or 0.0
        # Normalize to [0,1] for contract stability
        for c in candidates:
            c["score_norm"] = (c["score"] / max_raw) if max_raw > 0 else 0.0

        best = candidates[0]
        best_norm = float(best.get("score_norm", 0.0))
        best_id = str(best.get("id") or "")
        best_answer = str(best.get("answer") or "")

        # Dynamic min_score: if the query is very short, require stronger confidence
        q_terms = tokenize(query)
        eff_min = float(min_score)
        if len(q_terms) <= 2:
            eff_min = max(eff_min, min_score + 0.10)

        # Token overlap guardrail:
        # If the best candidate shares almost no meaningful terms with the query,
        # treat as not-found unless confidence is extremely high.
        best_doc_terms = self._docs_term_sets[self._doc_index(best_id)]
        overlap = len(set(q_terms) & best_doc_terms)
        overlap_ratio = overlap / max(1, len(set(q_terms)))

        # Out-of-domain strict mode: only accept if very confident + overlaps.
        q_lower = normalize_text(query)
        out_of_domain_hit = any(
            term in q_lower for term in _OUT_OF_DOMAIN_TERMS)

        matched = best_norm >= eff_min
        route = "bm25"

        if matched and out_of_domain_hit:
            # Stricter requirements for known out-of-domain terms
            if not (best_norm >= max(eff_min, 0.75) and overlap_ratio >= 0.34):
                matched = False
                route = "not_found_guardrail_out_of_domain"

        if matched and overlap_ratio < 0.20 and best_norm < 0.80:
            matched = False
            route = "not_found_guardrail_low_overlap"

        if not matched:
            return {
                "request_id": request_id or "",
                "matched": False,
                "answer": "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                "best_score": round(best_norm, 4),
                "best_id": best_id,
                "route": route,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "top": [
                    {"id": c["id"], "question": c["question"],
                        "score": round(float(c["score_norm"]), 4)}
                    for c in candidates
                ],
            }

        return {
            "request_id": request_id or "",
            "matched": True,
            "answer": best_answer,
            "best_score": round(best_norm, 4),
            "best_id": best_id,
            "route": route,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "top": [
                {"id": c["id"], "question": c["question"],
                    "score": round(float(c["score_norm"]), 4)}
                for c in candidates
            ],
        }

    def _doc_index(self, faq_id: str) -> int:
        """Find doc index for a faq_id; fallback to 0 if not found."""
        for i, it in enumerate(self.items):
            if it.id == faq_id:
                return i
        return 0
