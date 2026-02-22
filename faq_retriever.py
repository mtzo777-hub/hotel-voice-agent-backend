"""
FAQ Retriever (BM25 lexical scoring) with stronger Not-Found guardrails.

Goal:
- Never return random wrong answers for nonsense queries.
- matched=false unless there is enough lexical evidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from difflib import SequenceMatcher


SAFE_FALLBACK = "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    # normalize hyphens and spaces
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    """
    Conservative tokenizer:
    - Keeps alphanumerics and hyphen words
    - Drops very short tokens that often cause noise (except 'pm', 'am')
    """
    s = _normalize_text(s)
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s)
    cleaned = []
    for t in tokens:
        if t in ("am", "pm"):
            cleaned.append(t)
            continue
        if len(t) <= 1:
            continue
        cleaned.append(t)
    return cleaned


def _is_gibberish(query: str) -> bool:
    """
    Detect obvious nonsense:
    - very short and no meaningful tokens
    - mostly non-alphanumeric
    - single token with low vowel ratio and long random letters
    """
    q = (query or "").strip()
    if not q:
        return True

    tokens = _tokenize(q)
    if len(tokens) == 0:
        return True

    # single very short token like "ok", "hmm"
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return True

    # If the query is one long token and looks random
    if len(tokens) == 1 and len(tokens[0]) >= 5:
        t = tokens[0]
        vowels = sum(1 for c in t if c in "aeiou")
        # very low vowel ratio often indicates random string: "afggg", "asdfghjkl"
        if vowels / max(len(t), 1) < 0.2:
            return True

    return False


@dataclass
class RetrievalResult:
    matched: bool
    answer: str
    best_id: str
    best_score: float
    route: str
    top: List[Dict[str, Any]]


class FAQRetriever:
    def __init__(self, faq_path: str = "faq.json") -> None:
        self.faq_path = faq_path
        self.is_ready: bool = False
        self.last_error: Optional[str] = None

        self.entries: List[Dict[str, Any]] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.ids: List[str] = []

        # BM25 statistics
        self.doc_tokens: List[List[str]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.doc_len: List[int] = []

        # parameters
        self.k1 = 1.5
        self.b = 0.75

        self._load_and_build()

    @property
    def faq_count(self) -> int:
        return len(self.entries)

    def _load_and_build(self) -> None:
        try:
            p = Path(self.faq_path)
            if not p.exists():
                # try relative to current file
                p2 = Path(__file__).parent / self.faq_path
                if p2.exists():
                    p = p2

            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("faq.json must be a list of entries")

            self.entries = raw
            self.ids = [str(x.get("id", "")).strip() for x in raw]
            self.questions = [str(x.get("question", "")).strip() for x in raw]
            self.answers = [str(x.get("answer", "")).strip() for x in raw]

            # Build BM25 index over questions (and optionally ids)
            self.doc_tokens = []
            for q, _id in zip(self.questions, self.ids):
                # index both question and id to help short keyword queries
                doc = f"{q} {_id}".strip()
                toks = _tokenize(doc)
                self.doc_tokens.append(toks)

            self.doc_len = [len(toks) for toks in self.doc_tokens]
            self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)

            # DF
            self.df = {}
            for toks in self.doc_tokens:
                for t in set(toks):
                    self.df[t] = self.df.get(t, 0) + 1

            # IDF (BM25)
            N = len(self.doc_tokens)
            self.idf = {}
            for t, df in self.df.items():
                # classic BM25 idf
                self.idf[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))

            self.is_ready = True
            self.last_error = None
        except Exception as e:
            self.is_ready = False
            self.last_error = repr(e)

    def _bm25_scores(self, query_tokens: List[str]) -> List[float]:
        if not query_tokens:
            return [0.0 for _ in self.doc_tokens]

        scores: List[float] = []
        for toks, dl in zip(self.doc_tokens, self.doc_len):
            if dl == 0:
                scores.append(0.0)
                continue

            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            score = 0.0
            for q in query_tokens:
                if q not in tf:
                    continue
                idf = self.idf.get(q, 0.0)
                freq = tf[q]
                denom = freq + self.k1 * \
                    (1 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                score += idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)

            scores.append(float(score))
        return scores

    def _secondary_similarity(self, query: str, candidate_question: str) -> float:
        """
        Secondary guardrail: character similarity between query and the candidate question.
        Helps prevent random strings from matching due to tokenization artifacts.
        """
        qn = _normalize_text(query)
        cn = _normalize_text(candidate_question)
        if not qn or not cn:
            return 0.0
        return float(SequenceMatcher(None, qn, cn).ratio())

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> RetrievalResult:
        # Basic readiness
        if not self.is_ready:
            return RetrievalResult(
                matched=False,
                answer="Sorry, the FAQ knowledge base is currently unavailable.",
                best_id="",
                best_score=0.0,
                route="degraded_no_index",
                top=[],
            )

        # Strong not-found gate for obvious nonsense
        if _is_gibberish(query):
            return RetrievalResult(
                matched=False,
                answer=SAFE_FALLBACK,
                best_id="",
                best_score=0.0,
                route="guardrail_gibberish",
                top=[],
            )

        q_tokens = _tokenize(query)
        if not q_tokens:
            return RetrievalResult(
                matched=False,
                answer=SAFE_FALLBACK,
                best_id="",
                best_score=0.0,
                route="guardrail_empty",
                top=[],
            )

        scores = self._bm25_scores(q_tokens)

        # Take top_k candidates
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_idx = indexed[: max(1, min(top_k, 20))]

        # Build top list
        top_list: List[Dict[str, Any]] = []
        for i, sc in top_idx:
            top_list.append(
                {"id": self.ids[i], "question": self.questions[i], "score": float(sc)})

        best_i, best_score = top_idx[0]
        best_id = self.ids[best_i]
        best_q = self.questions[best_i]
        best_a = self.answers[best_i] if self.answers[best_i] else SAFE_FALLBACK

        # Secondary similarity guardrail (prevents random strings matching)
        sim = self._secondary_similarity(query, best_q)

        # Token overlap guardrail
        best_tokens = set(_tokenize(best_q))
        overlap = len(set(q_tokens) & best_tokens) / max(len(set(q_tokens)), 1)

        # Decision:
        # Must pass BM25 threshold + overlap OR similarity.
        passed = (best_score >= float(min_score)) and (
            overlap >= 0.5 or sim >= 0.42)

        if not passed:
            return RetrievalResult(
                matched=False,
                answer=SAFE_FALLBACK,
                best_id=best_id,
                best_score=float(best_score),
                route="not_found",
                top=top_list,
            )

        return RetrievalResult(
            matched=True,
            answer=best_a,
            best_id=best_id,
            best_score=float(best_score),
            route="bm25",
            top=top_list,
        )
