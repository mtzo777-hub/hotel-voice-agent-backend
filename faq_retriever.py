"""
faq_retriever.py
----------------
Lightweight FAQ retriever designed to work well on Cloud Run without extra
dependencies (no FAISS required).

It loads `faq.json` (list of objects with at least: {"id": "...", "text": "..."} )
and performs a BM25-style lexical search.

Why this design (Option B / best practice for maintainability)
- No external index files like /app/rag_store/* to keep in sync
- Only one artifact to maintain: faq.json
- Clear health/status information for debugging
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Lightweight stopword list to avoid common verbs dominating matches
_STOPWORDS = set(
    """
a an the and or but if then else when what how where who whom which
is are was were be been being
do does did have has had
can could should would may might will shall
i you we they he she it my your our their me us them
please tell show give provide
""".split()
)


def _normalize_query(s: str) -> str:
    # Normalize common hotel phrases (helps matching ids like "checkin_time")
    s = (s or "").lower()
    s = s.replace("check-in", "check in")
    s = s.replace("check-out", "checkout")
    s = s.replace("check out", "checkout")
    s = s.replace("wi-fi", "wifi").replace("wi fi", "wifi")
    return s


def _tokenize(s: str) -> List[str]:
    toks = _TOKEN_RE.findall((s or "").lower())
    return [t for t in toks if t not in _STOPWORDS]


@dataclass
class RetrieverStatus:
    ready: bool
    mode: str
    faq_json_path: str
    faq_count: int
    error: Optional[str] = None


class FAQRetriever:
    """
    BM25-like lexical retriever over faq.json entries.
    """

    def __init__(self, faq_json_path: str = "/app/faq.json") -> None:
        self.faq_json_path = faq_json_path
        self.items: List[Dict[str, Any]] = []
        self._docs_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._ready: bool = False
        self._error: Optional[str] = None

        self.reload()

    def status(self) -> RetrieverStatus:
        return RetrieverStatus(
            ready=self._ready,
            mode="bm25_lexical",
            faq_json_path=self.faq_json_path,
            faq_count=len(self.items),
            error=self._error,
        )

    def reload(self) -> None:
        self._ready = False
        self._error = None
        self.items = []
        self._docs_tokens = []
        self._doc_len = []
        self._avgdl = 0.0
        self._idf = {}

        try:
            with open(self.faq_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("faq.json must be a list of objects")

            cleaned: List[Dict[str, Any]] = []
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                _id = str(obj.get("id", "")).strip()
                text = str(obj.get("text", "")).strip()
                if not _id and not text:
                    continue
                cleaned.append({"id": _id, "text": text})

            if not cleaned:
                raise ValueError(
                    "faq.json loaded but contained 0 valid entries")

            self.items = cleaned

            # Build BM25 statistics
            docs_tokens: List[List[str]] = []
            doc_lens: List[int] = []
            df: Dict[str, int] = {}

            for it in self.items:
                # Include id tokens to help match short queries like "check-in time"
                doc = f"{it.get('id', '')} {it.get('text', '')}"
                toks = _tokenize(_normalize_query(doc))
                if not toks:
                    toks = ["_empty_"]
                docs_tokens.append(toks)
                doc_lens.append(len(toks))

                for t in set(toks):
                    df[t] = df.get(t, 0) + 1

            N = len(docs_tokens)
            avgdl = sum(doc_lens) / max(1, N)

            # IDF with BM25 smoothing
            idf: Dict[str, float] = {}
            for t, dfi in df.items():
                idf[t] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

            self._docs_tokens = docs_tokens
            self._doc_len = doc_lens
            self._avgdl = avgdl
            self._idf = idf
            self._ready = True

        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._ready = False

    def _bm25_score(self, q_tokens: List[str], doc_tokens: List[str], doc_len: int) -> float:
        # BM25 parameters
        k1 = 1.5
        b = 0.75

        tf: Dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        denom_norm = k1 * (1.0 - b + b * (doc_len / (self._avgdl or 1.0)))

        for t in q_tokens:
            if t not in tf:
                continue
            idf = self._idf.get(t, 0.0)
            f = tf[t]
            score += idf * (f * (k1 + 1.0)) / (f + denom_norm)

        return score

    def search(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
        """
        Returns:
          {
            "answer": str,
            "matched": bool,
            "best_score": float,
            "top": [{"id":..., "text":..., "score":...}, ...],
            "error": Optional[str]
          }
        """
        if not self._ready:
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": self._error or "not_ready",
            }

        q_norm = _normalize_query(query)
        q_tokens = _tokenize(q_norm)

        if not q_tokens:
            return {
                "answer": "Please provide a non-empty query.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": "empty_query",
            }

        scored: List[Tuple[float, int]] = []
        for i, doc_tokens in enumerate(self._docs_tokens):
            s = self._bm25_score(q_tokens, doc_tokens, self._doc_len[i])

            # Strong boost if query tokens appear in the FAQ id (your ids are highly informative)
            faq_id = (self.items[i].get("id") or "").lower()
            if faq_id:
                id_tokens = set(_tokenize(faq_id.replace("_", " ")))
                overlap = sum(1 for t in q_tokens if t in id_tokens)
                if overlap:
                    s += 2.0 * overlap  # heavy boost for id overlap

                # common intent boosts
                if "check" in q_tokens and ("in" in q_tokens or "checkin" in q_tokens) and (
                    "check_in" in faq_id or "checkin" in faq_id
                ):
                    s += 2.5
                if "checkout" in q_tokens and ("check_out" in faq_id or "checkout" in faq_id):
                    s += 2.5

            scored.append((s, i))

        scored.sort(reverse=True, key=lambda x: x[0])
        top_k = max(1, min(int(top_k), 20))

        top = []
        for s, i in scored[:top_k]:
            it = self.items[i]
            top.append({"id": it.get("id", ""), "text": it.get(
                "text", ""), "score": round(float(s), 6)})

        best = top[0] if top else None
        best_score = float(best["score"]) if best else 0.0
        matched = bool(best) and best_score >= float(min_score)

        answer = best["text"] if matched and best else "No confident match found."
        return {
            "answer": answer,
            "matched": matched,
            "best_score": best_score,
            "top": top,
            "error": None if matched else (None if best else "no_results"),
        }
