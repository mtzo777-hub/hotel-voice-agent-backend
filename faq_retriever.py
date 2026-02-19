import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer: lowercase words/numbers only
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _overlap_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    # Lightweight lexical similarity (no extra dependencies)
    if not query_tokens or not doc_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    inter = len(q & d)
    if inter == 0:
        return 0.0
    # normalize by query size so score stays ~0..1
    return inter / max(1, len(q))


@dataclass
class FAQEntry:
    id: Optional[str]
    question: str
    answer: str
    # Cached tokens for fast scoring
    tokens: List[str]


class FAQRetriever:
    """
    Option B (best-practice for your case):
    - Always works even without rag_store (FAISS files).
    - Uses faq.json as the single source of truth (easy to maintain).
    - Fast lexical retrieval, no extra libraries.
    """

    def __init__(self, base_dir: str = "/app", faq_json_filename: str = "faq.json") -> None:
        self.base_dir = base_dir
        self.faq_json_path = os.path.join(base_dir, faq_json_filename)

        self.entries: List[FAQEntry] = []
        self.last_error: Optional[str] = None
        self.mode: str = "lexical_in_memory"

        self._load_faq_json()

    def _load_faq_json(self) -> None:
        self.entries = []
        self.last_error = None

        if not os.path.exists(self.faq_json_path):
            self.last_error = f"faq.json not found at {self.faq_json_path}"
            return

        try:
            with open(self.faq_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Your file is a list of {"id": "...", "text": "..."} (based on what you uploaded)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        _id = str(item.get("id")) if item.get(
                            "id") is not None else None

                        # Accept multiple possible shapes
                        q = item.get("question") or item.get("q")
                        a = item.get("answer") or item.get("a")
                        text = item.get("text")

                        if q and a:
                            question = str(q).strip()
                            answer = str(a).strip()
                        elif text:
                            # Treat "text" as an FAQ chunk that can answer directly.
                            # We index by the same text to match keywords.
                            question = str(text).strip()
                            answer = str(text).strip()
                        else:
                            continue

                        tokens = _tokenize(question + " " + answer)
                        if tokens:
                            self.entries.append(
                                FAQEntry(id=_id, question=question,
                                         answer=answer, tokens=tokens)
                            )

            # Also support dict format if you ever switch later
            elif isinstance(data, dict):
                items = data.get("faqs") or data.get(
                    "items") or data.get("data") or []
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        _id = str(item.get("id")) if item.get(
                            "id") is not None else None
                        q = item.get("question") or item.get(
                            "q") or item.get("text")
                        a = item.get("answer") or item.get(
                            "a") or item.get("text")
                        if not q or not a:
                            continue
                        question = str(q).strip()
                        answer = str(a).strip()
                        tokens = _tokenize(question + " " + answer)
                        if tokens:
                            self.entries.append(
                                FAQEntry(id=_id, question=question,
                                         answer=answer, tokens=tokens)
                            )
            else:
                self.last_error = f"Unsupported faq.json format: {type(data)}"
                return

            if not self.entries:
                self.last_error = "faq.json loaded but contained 0 valid entries"

        except Exception as e:
            self.last_error = f"Failed to load faq.json: {e}"

    def is_ready(self) -> bool:
        return len(self.entries) > 0

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "mode": self.mode,
            "faq_json_path": self.faq_json_path,
            "faq_count": len(self.entries),
            "error": self.last_error,
        }

    def search(self, query: str, top_k: int = 5) -> List[Tuple[FAQEntry, float]]:
        q_tokens = _tokenize(query)
        scored: List[Tuple[FAQEntry, float]] = []
        for e in self.entries:
            s = _overlap_score(q_tokens, e.tokens)
            scored.append((e, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
        if not self.is_ready():
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": self.last_error or "Retriever not initialized",
            }

        top = self.search(query=query, top_k=top_k)
        if not top:
            return {
                "answer": "No results.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": None,
            }

        best_entry, best_score = top[0]
        matched = best_score >= float(min_score)

        # Return the best answer; for your data, answer==text, so this is correct.
        return {
            "answer": best_entry.answer if matched else "No confident match.",
            "matched": matched,
            "best_score": round(float(best_score), 4),
            "top": [
                {
                    "id": t[0].id,
                    "score": round(float(t[1]), 4),
                    "text": t[0].answer,
                }
                for t in top
            ],
            "error": None if matched else f"Best score {best_score:.4f} < min_score {min_score}",
        }
