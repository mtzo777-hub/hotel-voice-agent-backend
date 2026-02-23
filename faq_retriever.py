"""
faq_retriever.py

Lexical FAQ retrieval for the Sunshine Hotel Voice Agent.
- Loads faq.json (list of {id, text})
- Builds a BM25 index over (id phrase variants + answer text)
- Applies guardrails so out-of-domain queries reliably fall back
No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import math
import re
import time
import uuid

_WORD_RE = re.compile(r"[a-z0-9]+")

# Keep stopwords conservative (don't remove domain words like "late", "check", "out", etc.)
_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "can",
    "could",
    "may",
    "might",
    "will",
    "would",
    "should",
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "we",
    "our",
    "ours",
    "they",
    "their",
    "it",
    "this",
    "that",
    "these",
    "those",
    "to",
    "for",
    "of",
    "on",
    "in",
    "at",
    "by",
    "with",
    "from",
    "as",
    "and",
    "or",
    "but",
    "please",
    "kindly",
    "thanks",
    "thank",
    "hi",
    "hello",
    "hey",
    "tell",
    "show",
    "give",
    "need",
    "want",
    "like",
    "any",
    "some",
    # question words
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    # fillers
    "ok",
    "okay",
    "hmm",
    "um",
    "uh",
}

# Simple out-of-domain keyword guardrail list (extend as needed).
# These came from your failing examples; keep it tight to avoid blocking legitimate hotel topics.
_OOD_KEYWORDS = {
    "warranty",
    "watch",
    "bicycle",
    "bike",
    "book",
    "beer",
    "wine",
    "vodka",
    "whisky",
    "whiskey",
    "refund",
    "return",
    "exchange",
    "ski",
    "resort",  # treat as OOD unless you actually support ski-resort info
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _strip_punct(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _expand_check_variants(s: str) -> str:
    """
    Important: avoid collapsing 'check in' -> 'checkin' only.
    We include BOTH forms so ids like 'checkin_time' and user phrases like 'check-in' match.
    """
    s = s.lower()

    # unify separators to space
    s = s.replace("-", " ")
    s = s.replace("_", " ")

    # normalize common join/split variants by DUPLICATING variants into text
    # Example: "check in" -> "... check in checkin ..."
    #          "checkout" -> "... checkout check out ..."
    def add_both(text: str, joined: str, split: str) -> str:
        if joined in text and split not in text:
            text = text.replace(joined, f"{joined} {split}")
        if split in text and joined not in text:
            text = text.replace(split, f"{split} {joined}")
        return text

    s = add_both(s, "checkin", "check in")
    s = add_both(s, "checkout", "check out")
    s = add_both(s, "late checkout", "late check out")
    s = add_both(s, "early checkin", "early check in")

    # phone/contact synonyms (duplicate)
    s = add_both(s, "phone", "contact")
    s = add_both(s, "telephone", "contact")
    s = add_both(s, "fee", "price")
    return re.sub(r"\s+", " ", s).strip()


def _normalize_query(q: str) -> str:
    q = _strip_punct(q)
    q = _expand_check_variants(q)
    return q


def _tokens(text: str) -> List[str]:
    text = _normalize_query(text)
    toks = _WORD_RE.findall(text)
    toks = [t for t in toks if t and t not in _STOPWORDS]
    return toks


def _has_ood_keyword(query: str) -> Optional[str]:
    q = _normalize_query(query)
    for kw in _OOD_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", q):
            return kw
    return None


@dataclass
class _BM25Index:
    docs_tokens: List[List[str]]
    doc_lens: List[int]
    avgdl: float
    idf: Dict[str, float]
    tf: List[Dict[str, int]]
    k1: float = 1.5
    b: float = 0.75

    @staticmethod
    def build(docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> "_BM25Index":
        tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        doc_lens: List[int] = []

        for dt in docs_tokens:
            counts: Dict[str, int] = {}
            for t in dt:
                counts[t] = counts.get(t, 0) + 1
            tf.append(counts)
            doc_lens.append(len(dt))
            for t in counts.keys():
                df[t] = df.get(t, 0) + 1

        n_docs = len(docs_tokens)
        avgdl = sum(doc_lens) / n_docs if n_docs else 0.0

        # BM25 IDF (Okapi-style)
        idf: Dict[str, float] = {}
        for term, dfi in df.items():
            idf[term] = math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5))

        return _BM25Index(
            docs_tokens=docs_tokens,
            doc_lens=doc_lens,
            avgdl=avgdl,
            idf=idf,
            tf=tf,
            k1=k1,
            b=b,
        )

    def scores(self, query_tokens: List[str]) -> List[float]:
        if not query_tokens or not self.docs_tokens:
            return [0.0] * len(self.docs_tokens)

        scores = [0.0] * len(self.docs_tokens)
        for qi in query_tokens:
            if qi not in self.idf:
                continue
            idf = self.idf[qi]
            for i, doc_tf in enumerate(self.tf):
                f = doc_tf.get(qi, 0)
                if f <= 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * \
                    (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores


class FAQRetriever:
    """
    Public API expected by main.py
    """

    def __init__(self, faq_path: str = "faq.json"):
        self.faq_path = faq_path
        self.is_ready: bool = False
        self.faq_count: int = 0

        self._entries: List[Dict[str, str]] = []
        self._doc_ids: List[str] = []
        self._bm25: Optional[_BM25Index] = None

        self._load_and_index()

    def _load_and_index(self) -> None:
        try:
            with open(self.faq_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                raise ValueError(
                    "faq.json must be a list of {id, text} objects")

            cleaned = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                _id = str(e.get("id", "")).strip()
                _text = str(e.get("text", "")).strip()
                if _id and _text:
                    cleaned.append({"id": _id, "text": _text})

            self._entries = cleaned
            self.faq_count = len(cleaned)

            docs_tokens: List[List[str]] = []
            self._doc_ids = []

            for e in cleaned:
                faq_id = e["id"]
                text = e["text"]

                # Build multiple searchable variants for the id, then append answer text.
                id_phrase = faq_id.replace("_", " ")
                id_phrase = _expand_check_variants(id_phrase)

                aliases = [
                    id_phrase,
                    id_phrase.replace("policy", ""),
                    id_phrase.replace("fee", "price"),
                    id_phrase.replace("price", "fee"),
                ]
                alias_blob = " ".join(a.strip() for a in aliases if a.strip())

                # Key idea: index aliases + REAL answer text (so natural language queries match)
                doc = f"{alias_blob} {text}"
                docs_tokens.append(_tokens(doc))
                self._doc_ids.append(faq_id)

            self._bm25 = _BM25Index.build(docs_tokens)
            self.is_ready = True

        except Exception:
            self.is_ready = False
            self._entries = []
            self.faq_count = 0
            self._doc_ids = []
            self._bm25 = None

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35, request_id: str = "") -> Dict[str, Any]:
        """
        Return contract:
          {
            request_id, matched, answer, best_score, best_id, route, latency_ms,
            top: [{id, question, score}, ...]
          }
        best_score is a 0-1 confidence score (not raw BM25).
        """
        t0 = _now_ms()
        rid = request_id or ""

        fallback = "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."

        if not self.is_ready or not self._bm25:
            return {
                "request_id": rid,
                "matched": False,
                "answer": "Sorry, the FAQ knowledge base is currently unavailable.",
                "best_score": 0.0,
                "best_id": "",
                "route": "degraded_no_index",
                "latency_ms": _now_ms() - t0,
                "top": [],
            }

        if not isinstance(query, str) or not query.strip():
            return {
                "request_id": rid,
                "matched": False,
                "answer": fallback,
                "best_score": 0.0,
                "best_id": "",
                "route": "guardrail_empty",
                "latency_ms": _now_ms() - t0,
                "top": [],
            }

        # Guardrail: obvious out-of-domain keywords
        hit = _has_ood_keyword(query)
        if hit:
            return {
                "request_id": rid,
                "matched": False,
                "answer": fallback,
                "best_score": 0.0,
                "best_id": "",
                "route": f"guardrail_ood_keyword:{hit}",
                "latency_ms": _now_ms() - t0,
                "top": [],
            }

        q_tokens = _tokens(query)
        if len(q_tokens) < 2:
            # Avoid accidental matches on ultra-short noise like "ok", "hmm", "afggg"
            return {
                "request_id": rid,
                "matched": False,
                "answer": fallback,
                "best_score": 0.0,
                "best_id": "",
                "route": "guardrail_too_short",
                "latency_ms": _now_ms() - t0,
                "top": [],
            }

        raw_scores = self._bm25.scores(q_tokens)

        idx_scores = list(enumerate(raw_scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = max(1, min(int(top_k or 5), 10))
        top = idx_scores[:top_k]

        best_idx, best_raw = top[0]
        second_raw = top[1][1] if len(top) > 1 else 0.0

        # Convert raw BM25 to a 0-1 confidence-like score
        K = 6.0
        best_norm = best_raw / (best_raw + K) if best_raw > 0 else 0.0
        gap = (best_raw - second_raw) / \
            (best_raw + 1e-9) if best_raw > 0 else 0.0
        gap = max(0.0, min(1.0, gap))

        conf = 0.75 * best_norm + 0.25 * gap
        matched = conf >= float(min_score or 0.35)

        best_id = self._doc_ids[best_idx] if best_raw > 0 else ""
        answer_text = self._entries[best_idx]["text"] if matched else fallback

        top_list: List[Dict[str, Any]] = []
        for i, s in top:
            s_norm = s / (s + K) if s > 0 else 0.0
            top_list.append(
                {
                    "id": self._doc_ids[i],
                    "question": self._doc_ids[i],
                    "score": round(float(s_norm), 4),
                }
            )

        route = "bm25" if matched else "not_found"

        return {
            "request_id": rid,
            "matched": bool(matched),
            "answer": answer_text,
            "best_score": round(float(conf), 4),
            "best_id": best_id if matched else "",
            "route": route,
            "latency_ms": _now_ms() - t0,
            "top": top_list,
        }
