from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+")

# Small, safe stopword list to reduce “front desk / please / available” noise.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "at", "by",
    "is", "are", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "you", "your", "we", "our",
    "with", "as", "from", "into", "over", "under", "up", "down",
    "please", "kindly", "may", "might", "can", "could", "should", "would",
    "subject", "depends", "available", "availability",
    "hotel", "sunshine", "singapore",  # prevent identity words from dominating
    "contact", "front", "desk",        # super common across many answers
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> List[str]:
    toks = _WORD_RE.findall(_norm(text))
    return [t for t in toks if t and t not in _STOPWORDS]


def _id_to_phrase(faq_id: str) -> str:
    s = (faq_id or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _query_expansions(q: str) -> str:
    qn = _norm(q)
    qn = qn.replace("check in", "checkin")
    qn = qn.replace("check-out", "checkout").replace("check out", "checkout")
    qn = qn.replace("wi-fi", "wifi")

    extra = []
    # Keep expansions small and safe.
    if "wifi" in qn or "internet" in qn:
        extra += ["wireless", "password", "network"]
    if "phone" in qn or "contact number" in qn:
        extra += ["telephone", "call", "contact phone"]
    if "email" in qn:
        extra += ["contact email"]
    if "late" in qn and "checkout" in qn:
        extra += ["late check out", "checkout time", "late checkout fee"]
    if "early" in qn and "checkin" in qn:
        extra += ["early checkin fee", "checkin time"]
    if "visitor" in qn or "visitors" in qn or "guest" in qn and "room" in qn:
        extra += ["visitor policy", "registration"]

    if extra:
        qn = qn + " " + " ".join(extra)
    return qn


@dataclass
class FAQItem:
    faq_id: str
    text: str
    phrase: str
    doc: str   # searchable doc
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
        self.df = df

        self.idf: Dict[str, float] = {}
        for t, freq in df.items():
            self.idf[t] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

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

                # Keep doc = (id + phrase) heavily weighted + text.
                # This improves matching for short queries like "contact phone".
                doc = f"{faq_id} {phrase} {phrase} {text}"
                doc_tokens = _tokens(doc)

                items.append(FAQItem(faq_id=faq_id, text=text,
                             phrase=phrase, doc=doc, doc_tokens=doc_tokens))

            if not items:
                raise ValueError(
                    "faq.json loaded but contained 0 valid FAQ items (need id and text)")

            self.items = items
            self._bm25 = BM25([it.doc_tokens for it in items])
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
            "mode": "bm25_lexical_guarded",
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

    def _overlap_score(self, q_tokens: List[str], d_tokens: List[str]) -> float:
        # Jaccard-ish overlap (cheap rerank signal)
        qs = set(q_tokens)
        ds = set(d_tokens)
        if not qs or not ds:
            return 0.0
        inter = len(qs & ds)
        return inter / (len(qs) + 0.5)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[FAQItem, float, float]]:
        """
        Returns list of (item, bm25_score, blended_score).
        blended_score used for ranking; bm25_score kept for debug.
        """
        if not self.ready or not self._bm25:
            return []
        if not query or not query.strip():
            return []

        direct = self._direct_id_match(query)
        if direct:
            return [(direct, 999.0, 999.0)]

        expanded = _query_expansions(query)
        q_tokens = _tokens(expanded)
        if not q_tokens:
            return []

        bm25_scores = self._bm25.score(q_tokens)
        if not bm25_scores:
            return []

        # Candidate pool: take larger pool then rerank (stabilizes results)
        pool_k = max(10, top_k * 5)
        idxs = sorted(range(len(bm25_scores)),
                      key=lambda i: bm25_scores[i], reverse=True)[:pool_k]

        # Blend BM25 with overlap; overlap prevents bizarre matches.
        scored: List[Tuple[FAQItem, float, float]] = []
        for i in idxs:
            it = self.items[i]
            ov = self._overlap_score(q_tokens, it.doc_tokens)
            blended = bm25_scores[i] + (2.0 * ov)  # overlap boost
            scored.append((it, float(bm25_scores[i]), float(blended)))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[: max(1, top_k)]

    def answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
        min_margin: float = 0.12,
    ) -> Dict[str, Any]:
        """
        min_score: now applied on a normalized blended score (0..1) for consistency.
        min_margin: reject if top1 not sufficiently better than top2 (ambiguity).
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

        # Convert blended scores to 0..1 for stable thresholding
        blended_scores = [r[2] for r in results]
        max_b = max(blended_scores) if blended_scores else 0.0
        norm = [(s / max_b) if max_b > 0 else 0.0 for s in blended_scores]

        best_item = results[0][0]
        best_score = norm[0]

        # Stricter for very short queries
        q_len = len(_tokens(_query_expansions(query)))
        short_query = q_len <= 2
        eff_min_score = min_score + (0.15 if short_query else 0.0)

        # Ambiguity margin check
        margin_ok = True
        if len(norm) >= 2:
            margin_ok = (norm[0] - norm[1]) >= min_margin

        matched = (best_score >= eff_min_score) and margin_ok

        top_list = []
        for (it, bm25_s, blended_s), ns in zip(results, norm):
            top_list.append({
                "id": it.faq_id,
                "question": it.phrase or it.faq_id,
                "score": round(float(ns), 4),
            })

        err = None
        if not matched:
            if best_score < eff_min_score:
                err = f"best_score {best_score:.4f} < min_score {eff_min_score:.2f}"
            elif not margin_ok:
                err = "ambiguous_match (top1 too close to top2)"
            else:
                err = "no_match"

        return {
            "answer": best_item.text if matched else "No match found.",
            "matched": matched,
            "best_score": round(float(best_score), 4),
            "top": top_list,
            "error": err,
        }
