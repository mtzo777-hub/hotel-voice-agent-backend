import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def _tokenize(text: str) -> List[str]:
    # Basic lexical tokenizer (fast, dependency-free)
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return [t for t in text.split() if t]


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    # Character ngrams improve tolerance to minor typos/spacing
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not text:
        return []
    if len(text) < n:
        return [text]
    return [text[i: i + n] for i in range(0, len(text) - n + 1)]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


class BM25OkapiLite:
    """
    Lightweight BM25 implementation (no external dependency).
    Good enough for small corpora like FAQ (~150 entries).
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens

        self.N = len(corpus_tokens)
        self.doc_lens = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0

        # Document frequency per term
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.df = df

        # Smoothed IDF
        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                1.0 + (self.N - freq + 0.5) / (freq + 0.5))

        # Sparse TF per doc
        self.tfs: List[Dict[str, int]] = []
        for doc in corpus_tokens:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if not self.N:
            return []
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_lens[i] or 1
            denom_norm = self.k1 * \
                (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            tf = self.tfs[i]
            s = 0.0
            for t in query_tokens:
                if t not in tf:
                    continue
                f = tf[t]
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + denom_norm)
            scores[i] = s
        return scores


@dataclass
class RetrievalCandidate:
    id: str
    question: str
    answer: str
    score_raw: float
    score_conf: float


@dataclass
class RetrievalResult:
    query: str
    matched: bool
    answer: str
    best_score: float
    best_id: str
    route: str
    candidates: List[RetrievalCandidate]
    latency_ms: int
    reason: Optional[str] = None


class FAQRetriever:
    """
    Hybrid lexical retriever:
    - BM25 raw scoring
    - Char trigram Jaccard (typos/spaces)
    Produces confidence score in [0..1] used for matched decision.
    """

    DEFAULT_FAQ_PATHS = [
        "faq.json",
        "/app/faq.json",
        "/app/data/faq.json",
        "/workspace/faq.json",
    ]

    # Inputs that should NEVER match an FAQ (avoid random matches)
    FILLER_QUERIES = {
        "ok",
        "okay",
        "hmm",
        "erm",
        "uh",
        "um",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "thankyou",
        "bye",
        "goodbye",
    }

    def __init__(self, faq_path: Optional[str] = None):
        self.faq_path = faq_path or os.getenv(
            "FAQ_PATH") or self._find_faq_path()
        self.faq = self._load_faq(self.faq_path)

        self._docs: List[Dict[str, str]] = []
        corpus_tokens: List[List[str]] = []

        for item in self.faq:
            q = (item.get("question") or "").strip()
            a = (item.get("text") or item.get("answer") or "").strip()
            if not q or not a:
                continue
            doc_text = f"{q} {item.get('id', '')}"
            toks = _tokenize(doc_text)
            self._docs.append(
                {"id": item.get("id", ""), "question": q, "answer": a, "doc_text": doc_text})
            corpus_tokens.append(toks)

        self.bm25 = BM25OkapiLite(corpus_tokens) if corpus_tokens else None

    def _find_faq_path(self) -> str:
        for p in self.DEFAULT_FAQ_PATHS:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            "faq.json not found. Set FAQ_PATH env var or include faq.json in the container.")

    def _load_faq(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "faq" in data and isinstance(data["faq"], list):
            return data["faq"]
        if isinstance(data, list):
            return data
        raise ValueError(
            "Unsupported FAQ JSON format. Expected a list or {faq:[...]}.")

    def _confidence(self, query: str, doc_text: str, bm25_raw: float) -> float:
        # Normalize BM25 raw score into [0..1] with saturation
        bm25_norm = bm25_raw / (bm25_raw + 5.0) if bm25_raw > 0 else 0.0

        # Char trigram similarity
        char_sim = _jaccard(_char_ngrams(query, 3), _char_ngrams(doc_text, 3))

        # Token overlap ratio (guard against random matches)
        q_toks = _tokenize(query)
        d_toks = _tokenize(doc_text)
        tok_overlap = (len(set(q_toks) & set(d_toks)) /
                       float(len(set(q_toks)) or 1.0)) if q_toks else 0.0

        blended = 0.55 * bm25_norm + 0.30 * char_sim + 0.15 * tok_overlap
        return float(max(0.0, min(1.0, blended)))

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> RetrievalResult:
        t0 = _now_ms()
        query = (query or "").strip()

        if not query:
            return RetrievalResult(
                query=query,
                matched=False,
                answer="Sorry, I didn’t catch that. Please ask a hotel question.",
                best_score=0.0,
                best_id="",
                route="guardrail_empty",
                candidates=[],
                latency_ms=_now_ms() - t0,
                reason="empty_query",
            )

        q_norm = re.sub(r"\s+", " ", query.lower()).strip()
        if q_norm in self.FILLER_QUERIES or len(q_norm) <= 2:
            return RetrievalResult(
                query=query,
                matched=False,
                answer="Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                best_score=0.0,
                best_id="",
                route="guardrail_filler_or_short",
                candidates=[],
                latency_ms=_now_ms() - t0,
                reason="filler_or_short",
            )

        if not self.bm25 or not self._docs:
            return RetrievalResult(
                query=query,
                matched=False,
                answer="Sorry, the FAQ knowledge base is currently unavailable.",
                best_score=0.0,
                best_id="",
                route="degraded_no_index",
                candidates=[],
                latency_ms=_now_ms() - t0,
                reason="no_index",
            )

        q_tokens = _tokenize(query)
        if not q_tokens:
            return RetrievalResult(
                query=query,
                matched=False,
                answer="Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                best_score=0.0,
                best_id="",
                route="guardrail_no_tokens",
                candidates=[],
                latency_ms=_now_ms() - t0,
                reason="no_tokens",
            )

        raw_scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(raw_scores)),
                        key=lambda i: raw_scores[i], reverse=True)
        top_idx = ranked[: max(1, min(int(top_k or 5), 20))]

        candidates: List[RetrievalCandidate] = []
        for i in top_idx:
            doc = self._docs[i]
            raw = float(raw_scores[i])
            conf = self._confidence(query, doc["doc_text"], raw)
            candidates.append(
                RetrievalCandidate(
                    id=doc["id"],
                    question=doc["question"],
                    answer=doc["answer"],
                    score_raw=raw,
                    score_conf=conf,
                )
            )

        candidates_sorted = sorted(candidates, key=lambda c: (
            c.score_conf, c.score_raw), reverse=True)
        best = candidates_sorted[0] if candidates_sorted else None

        best_conf = float(best.score_conf) if best else 0.0
        best_raw = float(best.score_raw) if best else 0.0
        second_conf = float(candidates_sorted[1].score_conf) if len(
            candidates_sorted) > 1 else 0.0
        margin = best_conf - second_conf

        # Match decision:
        # - must be above min_score
        # - must have a margin, unless confidence is clearly high
        matched = (best_conf >= float(min_score)) and (
            margin >= 0.05 or best_conf >= (float(min_score) + 0.10))

        if matched and best:
            answer = best.answer
            route = "bm25_hybrid_match"
            best_id = best.id
        else:
            answer = "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."
            route = "safe_fallback"
            best_id = best.id if best else ""

        return RetrievalResult(
            query=query,
            matched=bool(matched),
            answer=answer,
            best_score=best_conf,
            best_id=best_id,
            route=route,
            candidates=candidates_sorted,
            latency_ms=_now_ms() - t0,
            reason=None if matched else f"below_threshold_or_low_margin (conf={best_conf:.3f}, raw={best_raw:.3f}, margin={margin:.3f})",
        )
