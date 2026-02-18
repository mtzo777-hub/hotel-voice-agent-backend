"""
FAQ Retriever

Option B (best-practice for ops/maintenance):
- Prefer a prebuilt local retrieval store (FAISS artifacts under ./rag_store) if it exists.
- Fall back to a lightweight lexical retriever using faq.json when the store is missing.

This makes deployments resilient: the service still answers even if vector artifacts are absent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- Data loading helpers ----------

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_faq_items(raw: Any) -> List[Dict[str, str]]:
    """Try to normalize various faq.json shapes into a list of {question, answer}."""
    if raw is None:
        return []

    # common shapes:
    # 1) [ {"question":..., "answer":...}, ... ]
    if isinstance(raw, list):
        items = raw
    # 2) {"faqs": [...]}
    elif isinstance(raw, dict) and isinstance(raw.get("faqs"), list):
        items = raw["faqs"]
    # 3) {"data": [...]}
    elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
        items = raw["data"]
    else:
        return []

    out: List[Dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or it.get("q")
             or it.get("title") or "").strip()
        a = (it.get("answer") or it.get("a") or it.get("text") or "").strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _lexical_score(query: str, question: str) -> float:
    """Simple, dependency-free score in [0, 1]."""
    q_tokens = set(_tokenize(query))
    s_tokens = set(_tokenize(question))
    if not q_tokens or not s_tokens:
        return 0.0

    overlap = len(q_tokens & s_tokens)
    jacc = overlap / max(1, len(q_tokens | s_tokens))

    # small bonus for substring match
    q_norm = " ".join(_tokenize(query))
    s_norm = " ".join(_tokenize(question))
    substr = 1.0 if q_norm and q_norm in s_norm else 0.0

    # weighted blend
    score = 0.85 * jacc + 0.15 * substr
    return max(0.0, min(1.0, score))


# ---------- Retriever ----------

@dataclass
class RetrievalResult:
    answer: str
    matched: bool
    best_score: float
    top: List[Dict[str, Any]]
    error: Optional[str] = None


class FAQRetriever:
    """Unified retriever with graceful fallback."""

    def __init__(
        self,
        faq_json_path: str = "faq.json",
        rag_store_dir: str = "rag_store",
    ) -> None:
        self.faq_json_path = Path(faq_json_path)
        self.rag_store_dir = Path(rag_store_dir)

        self._faqs: List[Dict[str, str]] = []
        self._faiss_ready = False
        self._faiss_error: Optional[str] = None

        # load lexical FAQ data (always)
        self._load_faq_json()

        # try loading vector store if present
        self._try_load_faiss_store()

    @property
    def ready(self) -> bool:
        # "ready" means we can answer using *either* method
        return bool(self._faqs) or self._faiss_ready

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "lexical_loaded": bool(self._faqs),
            "faiss_loaded": self._faiss_ready,
            "faiss_error": self._faiss_error,
            "faq_count": len(self._faqs),
            "rag_store_dir": str(self.rag_store_dir.resolve()),
            "faq_json_path": str(self.faq_json_path.resolve()),
        }

    def _load_faq_json(self) -> None:
        try:
            # allow relative paths (repo root)
            if not self.faq_json_path.exists():
                # also try /app/faq.json in container
                alt = Path("/app") / self.faq_json_path.name
                if alt.exists():
                    self.faq_json_path = alt

            raw = _read_json(self.faq_json_path)
            self._faqs = _normalize_faq_items(raw)
        except Exception as e:
            self._faqs = []
            # don't fail hard; fallback might still work via FAISS
            self._faiss_error = f"Failed to read faq.json: {e}"

    def _try_load_faiss_store(self) -> None:
        """Optional: load FAISS artifacts if they exist."""
        try:
            expected = [
                self.rag_store_dir / "faq.index",
                self.rag_store_dir / "faq_texts.json",
                self.rag_store_dir / "faq_meta.json",
            ]
            if not all(p.exists() for p in expected):
                missing = [str(p) for p in expected if not p.exists()]
                self._faiss_ready = False
                self._faiss_error = (
                    "RAG store missing. Using lexical fallback. Missing:\n- " +
                    "\n- ".join(missing)
                )
                return

            # Lazy import so deployments work even if faiss isn't installed
            import faiss  # type: ignore

            with (self.rag_store_dir / "faq_texts.json").open("r", encoding="utf-8") as f:
                self._texts = json.load(f)  # list[str]
            with (self.rag_store_dir / "faq_meta.json").open("r", encoding="utf-8") as f:
                self._meta = json.load(f)  # list[dict]
            self._index = faiss.read_index(
                str(self.rag_store_dir / "faq.index"))

            # embeddings must match whatever built the index; we only use the store if embeddings are available
            from openai import OpenAI  # type: ignore

            self._openai = OpenAI()
            self._faiss_ready = True
            self._faiss_error = None
        except Exception as e:
            self._faiss_ready = False
            self._faiss_error = f"FAISS load failed. Using lexical fallback. Reason: {e}"

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> RetrievalResult:
        query = (query or "").strip()
        if not query:
            return RetrievalResult(
                answer="Please provide a question.",
                matched=False,
                best_score=0.0,
                top=[],
                error="Empty query",
            )

        # 1) Try FAISS if available
        if self._faiss_ready:
            try:
                emb = self._openai.embeddings.create(
                    model=os.getenv("OPENAI_EMBED_MODEL",
                                    "text-embedding-3-small"),
                    input=query,
                )
                vec = emb.data[0].embedding

                import numpy as np  # type: ignore

                xq = np.array([vec], dtype="float32")
                D, I = self._index.search(xq, top_k)

                # Convert distances to a similarity-ish score (monotonic)
                scored: List[Tuple[int, float]] = []
                for idx, dist in zip(I[0].tolist(), D[0].tolist()):
                    if idx < 0:
                        continue
                    score = 1.0 / (1.0 + float(dist))
                    scored.append((idx, score))

                top: List[Dict[str, Any]] = []
                best_score = 0.0
                best_answer = ""

                for idx, score in scored:
                    meta = self._meta[idx] if idx < len(self._meta) else {}
                    text = self._texts[idx] if idx < len(self._texts) else ""
                    top.append({"score": score, "text": text, "meta": meta})
                    if score > best_score:
                        best_score = score
                        best_answer = meta.get("answer") or text or ""

                matched = best_score >= float(min_score)
                return RetrievalResult(
                    answer=best_answer if matched else "No confident match found.",
                    matched=matched,
                    best_score=best_score,
                    top=top,
                    error=None,
                )
            except Exception as e:
                # fall through to lexical
                faiss_err = f"Vector retrieval error; falling back to lexical. Reason: {e}"
        else:
            faiss_err = None

        # 2) Lexical fallback
        if not self._faqs:
            return RetrievalResult(
                answer="Retriever is not ready.",
                matched=False,
                best_score=0.0,
                top=[],
                error=faiss_err or self._faiss_error or "FAQ data not available",
            )

        scored = []
        for item in self._faqs:
            score = _lexical_score(query, item["question"])
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[: max(1, top_k)]

        best_score = float(top_items[0][0]) if top_items else 0.0
        matched = best_score >= float(min_score)

        top: List[Dict[str, Any]] = [
            {"score": float(
                s), "question": it["question"], "answer": it["answer"]}
            for s, it in top_items
        ]

        answer = top_items[0][1]["answer"] if (
            matched and top_items) else "No confident match found."

        return RetrievalResult(
            answer=answer,
            matched=matched,
            best_score=best_score,
            top=top,
            error=faiss_err or self._faiss_error,
        )
