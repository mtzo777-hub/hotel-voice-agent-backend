"""
faq_retriever.py

Goal:
- Work with ONLY faq.json (no rag_store, no embeddings, no external deps).
- Robust lexical retrieval (BM25 + a few safe heuristics).
- Deterministic and easy to maintain: update faq.json and redeploy.

faq.json expected format: a JSON list like:
[
  {"id": "checkin_time", "text": "Check-in is from 3 PM..."},
  ...
]
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(_norm(text))


def _id_to_phrase(faq_id: str) -> str:
    # "checkin_time" -> "checkin time"
    s = (faq_id or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _query_expansions(q: str) -> str:
    """
    Expand common paraphrases without changing meaning.
    Keep it small and safe (avoid surprising matches).
    """
    qn = _norm(q)

    # canonicalize common variants
    qn = qn.replace("check in", "checkin")
    qn = qn.replace("check-out", "checkout").replace("check out", "checkout")
    qn = qn.replace("wi-fi", "wifi")

    # light synonyms (add, don't replace)
    extra = []
    if "address" in qn or ("where" in qn and "hotel" in qn):
        extra += ["location", "located", "directions", "map"]
    if "wifi" in qn or "internet" in qn:
        extra += ["wireless", "network", "password"]
    if "breakfast" in qn:
        extra += ["buffet"]
    if "parking" in qn:
        extra += ["car park"]
    if "pool" in qn:
        extra += ["swimming"]
    if "gym" in qn or "fitness" in qn:
        extra += ["workout"]
    if "laundry" in qn:
        extra += ["washing", "dry cleaning", "drycleaning"]
    if "room types" in qn or ("room" in qn and "types" in qn):
        extra += ["available rooms", "categories"]
    if "late" in qn and "checkout" in qn:
        extra += ["late check out", "checkout time"]

    if extra:
        qn = qn + " " + " ".join(extra)
    return qn


@dataclass
class FAQItem:
    faq_id: str
    text: str
    phrase: str
    doc: str  # searchable document text


class BM25:
    """
    Minimal BM25 implementation (Okapi BM25).
    No third-party dependency.
    """

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = (sum(len(d)
                      for d in tokenized_docs) / self.N) if self.N else 0.0

        # document frequencies
        df: Dict[str, int] = {}
        for doc in tokenized_docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.df = df

        # idf
        self.idf: Dict[str, float] = {}
        for t, freq in df.items():
            self.idf[t] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

        # term frequencies per doc
        self.tf: List[Dict[str, int]] = []
        for doc in tokenized_docs:
            m: Dict[str, int] = {}
            for t in doc:
                m[t] = m.get(t, 0) + 1
            self.tf.append(m)

        self.dl = [len(d) for d in tokenized_docs]

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
            for obj in raw:
                if not isinstance(obj, dict):
                    continue
                faq_id = str(obj.get("id", "")).strip()
                text = str(obj.get("text", "")).strip()
                if not faq_id or not text:
                    continue

                phrase = _id_to_phrase(faq_id)
                # searchable doc = id + id phrase + answer text
                doc = f"{faq_id} {phrase} {text}"
                items.append(
                    FAQItem(faq_id=faq_id, text=text, phrase=phrase, doc=doc))

            if not items:
                raise ValueError(
                    "faq.json loaded but contained 0 valid FAQ items (need id and text)")

            self.items = items
            tokenized_docs = [_tokens(it.doc) for it in items]
            self._bm25 = BM25(tokenized_docs)
            self.ready = True
            self.error = None
        except Exception as e:
            self.items = []
            self._bm25 = None
            self.ready = False
            self.error = str(e)

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": "bm25_lexical",
            "faq_json_path": self.faq_json_path,
            "faq_count": len(self.items),
            "error": self.error,
        }

    def _direct_id_match(self, query: str) -> Optional[FAQItem]:
        q = _norm(query)
        q_id_like = re.sub(r"[^a-z0-9_]+", "_", q).strip("_")
        for it in self.items:
            if q == it.faq_id.lower() or q_id_like == it.faq_id.lower():
                return it
        return None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[FAQItem, float]]:
        if not self.ready or not self._bm25:
            return []
        if not query or not query.strip():
            return []

        direct = self._direct_id_match(query)
        if direct:
            return [(direct, 1.0)]

        expanded = _query_expansions(query)
        q_tokens = _tokens(expanded)
        if not q_tokens:
            return []

        scores = self._bm25.score(q_tokens)
        if not scores:
            return []

        max_s = max(scores) if scores else 0.0
        norm_scores = [(s / max_s) if max_s > 0 else 0.0 for s in scores]

        idxs = sorted(range(len(norm_scores)),
                      key=lambda i: norm_scores[i], reverse=True)[: max(1, top_k)]
        return [(self.items[i], float(norm_scores[i])) for i in idxs]

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
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

        best_item, best_score = results[0]
        matched = best_score >= float(min_score)

        top_list = []
        for it, s in results:
            top_list.append(
                {"id": it.faq_id, "question": it.phrase or it.faq_id,
                    "score": round(float(s), 4)}
            )

        return {
            "answer": best_item.text if matched else "No match found.",
            "matched": matched,
            "best_score": round(float(best_score), 4),
            "top": top_list,
            "error": None if matched else f"best_score {best_score:.4f} < min_score {min_score:.2f}",
        }
