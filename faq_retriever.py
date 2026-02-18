import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _normalize(text: str) -> str:
    text = text.lower().strip()
    # keep letters/numbers/spaces
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    t = _normalize(text)
    if not t:
        return []
    return t.split(" ")


def _cosine_binary_score(q_tokens: List[str], d_tokens: List[str]) -> float:
    # cosine similarity on binary token sets
    qs = set(q_tokens)
    ds = set(d_tokens)
    if not qs or not ds:
        return 0.0
    inter = len(qs.intersection(ds))
    denom = (len(qs) * len(ds)) ** 0.5
    if denom == 0:
        return 0.0
    return inter / denom


@dataclass
class Match:
    question: str
    answer: str
    score: float


class FAQRetriever:
    """
    Option B (best practice for Cloud Run):
    - Use faq.json as the source of truth.
    - Build an in-memory lexical matcher at startup.
    - No dependency on /app/rag_store files.
    """

    def __init__(self, faq_json_path: str = "/app/faq.json"):
        self.faq_json_path = faq_json_path
        self._ready: bool = False
        self._error: Optional[str] = None
        self._items: List[Tuple[str, str, List[str]]] = []  # (q, a, tokens)

        self._load_faq()

    def _load_faq(self) -> None:
        try:
            if not os.path.exists(self.faq_json_path):
                self._ready = False
                self._error = f"faq.json not found at {self.faq_json_path}"
                self._items = []
                return

            with open(self.faq_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Accept either:
            # - [{"question": "...", "answer": "..."}, ...]
            # - {"faqs": [{"q": "...", "a": "..."}, ...]}
            faqs = None
            if isinstance(data, list):
                faqs = data
            elif isinstance(data, dict):
                faqs = data.get("faqs") or data.get(
                    "items") or data.get("data")
            if not faqs or not isinstance(faqs, list):
                self._ready = False
                self._error = "faq.json format not recognized (expected list of Q/A objects)"
                self._items = []
                return

            items: List[Tuple[str, str, List[str]]] = []
            for row in faqs:
                if not isinstance(row, dict):
                    continue
                q = row.get("question") or row.get(
                    "q") or row.get("Query") or row.get("Question")
                a = row.get("answer") or row.get("a") or row.get(
                    "Response") or row.get("Answer")
                if not q or not a:
                    continue
                qt = _tokens(str(q))
                # include both question and answer tokens to improve matching a bit
                at = _tokens(str(a))
                items.append((str(q).strip(), str(
                    a).strip(), list(set(qt + at))))

            self._items = items
            self._ready = len(self._items) > 0
            self._error = None if self._ready else "faq.json loaded but contained 0 valid Q/A pairs"

        except Exception as e:
            self._ready = False
            self._error = f"Failed to load faq.json: {type(e).__name__}: {e}"
            self._items = []

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "faq_json_path": self.faq_json_path,
            "faq_count": len(self._items),
            "error": self._error,
            "mode": "lexical_in_memory",
        }

    def search(self, query: str, top_k: int = 5) -> List[Match]:
        q = str(query or "").strip()
        if not q:
            return []

        q_tokens = _tokens(q)
        scored: List[Match] = []
        for (qq, aa, dtokens) in self._items:
            score = _cosine_binary_score(q_tokens, dtokens)

            # small boost if query is substring of the FAQ question
            nq = _normalize(q)
            nqq = _normalize(qq)
            if nq and nqq and nq in nqq:
                score = min(1.0, score + 0.15)

            if score > 0:
                scored.append(Match(question=qq, answer=aa, score=score))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: max(1, int(top_k or 5))]

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
        """
        Returns:
          answer: str
          matched: bool
          best_score: float
          top: list[{question, answer, score}]
          error: optional str
        """
        if not self._ready:
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0,
                "top": [],
                "error": self._error or "Unknown retriever error",
            }

        hits = self.search(query=query, top_k=top_k)
        if not hits:
            return {
                "answer": "Sorry, I couldn't find a relevant answer.",
                "matched": False,
                "best_score": 0,
                "top": [],
                "error": None,
            }

        best = hits[0]
        matched = best.score >= float(min_score or 0.0)

        return {
            "answer": best.answer if matched else "Sorry, I couldn't find a confident match for that question.",
            "matched": matched,
            "best_score": round(float(best.score), 4),
            "top": [
                {"question": h.question, "answer": h.answer,
                    "score": round(float(h.score), 4)}
                for h in hits
            ],
            "error": None,
        }
