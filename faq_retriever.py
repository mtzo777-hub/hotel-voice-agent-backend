import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL",
                        "text-embedding-3-small").strip()

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "rag_store"
INDEX_PATH = STORE_DIR / "faq.index"
TEXTS_PATH = STORE_DIR / "faq_texts.json"
META_PATH = STORE_DIR / "faq_meta.json"

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.35"))


# ----------------------------
# Helpers
# ----------------------------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", " ")

    # normalize "check in"/"check-out" variants
    s = re.sub(r"\bcheck\s+in_", "checkin_", s)
    s = re.sub(r"\bcheck\s+out_", "checkout_", s)
    s = re.sub(r"\bcheck\s+in\b", "checkin", s)
    s = re.sub(r"\bcheck\s+out\b", "checkout", s)

    # remove punctuation, keep underscore
    s = re.sub(r"[^\w\s_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_id_like(s: str) -> str:
    s = _norm_text(s).replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _token_set(s: str) -> set[str]:
    s = _norm_text(s)
    return {t for t in s.split(" ") if t}


# ----------------------------
# Retriever
# ----------------------------
class FAQRetriever:
    """
    Cloud Run-safe retriever:
    - Never crashes the container at import time.
    - If dependencies are missing (rag_store / OPENAI_API_KEY), it returns clean errors at request time.
    """

    def __init__(self) -> None:
        self.ready: bool = True
        self.init_error: Optional[str] = None

        self.index = None
        self.texts: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.id_map: Dict[str, Tuple[str, int]] = {}
        self._text_tokens: List[set[str]] = []

        missing = [p for p in [INDEX_PATH,
                               TEXTS_PATH, META_PATH] if not p.exists()]
        if missing:
            self.ready = False
            self.init_error = (
                "RAG store missing. Ensure rag_store is packaged in the image.\n"
                "Missing:\n" + "\n".join([f"- {p}" for p in missing])
            )
            return

        try:
            self.index = faiss.read_index(str(INDEX_PATH))

            with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                self.texts = json.load(f)

            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

            for i, m in enumerate(self.meta):
                faq_id = str(m.get("id", "")).strip()
                if faq_id:
                    self.id_map[faq_id.lower()] = (self.texts[i], i)

            self._text_tokens = [_token_set(t) for t in self.texts]

        except Exception as e:
            self.ready = False
            self.init_error = f"Failed to load RAG store: {str(e)}"

    def _get_openai_client(self) -> Optional["OpenAI"]:
        if not OPENAI_API_KEY:
            return None
        if OpenAI is None:
            return None
        return OpenAI(api_key=OPENAI_API_KEY)

    def _embed_query(self, q: str) -> np.ndarray:
        client = self._get_openai_client()
        if client is None:
            raise RuntimeError(
                "OPENAI_API_KEY missing or OpenAI SDK unavailable; cannot embed query.")
        emb = client.embeddings.create(model=EMBED_MODEL, input=q)
        vec = np.array(emb.data[0].embedding, dtype=np.float32)
        return _l2_normalize(vec)

    def _faiss_search(self, qvec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None
        scores, idxs = self.index.search(qvec.astype(np.float32), top_k)
        return scores[0], idxs[0]

    def _simple_overlap_score(self, q_tokens: set[str], doc_tokens: set[str]) -> float:
        if not q_tokens or not doc_tokens:
            return 0.0
        inter = len(q_tokens.intersection(doc_tokens))
        return inter / (len(q_tokens) + 1e-9)

    def _exact_id_match(self, q: str) -> Tuple[bool, Dict[str, Any]]:
        qn = _norm_text(q)
        id_like = _to_id_like(q)

        tokens = set(re.findall(r"[a-z0-9_]+", qn))
        for t in tokens:
            if t in self.id_map:
                text, _ = self.id_map[t]
                return True, {
                    "answer": text,
                    "matched": True,
                    "best_score": 1.0,
                    "top": [{"score": 1.0, "id": t, "text": text}],
                }

        if id_like in self.id_map:
            text, _ = self.id_map[id_like]
            return True, {
                "answer": text,
                "matched": True,
                "best_score": 1.0,
                "top": [{"score": 1.0, "id": id_like, "text": text}],
            }

        return False, {}

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"answer": "Please ask a question about the hotel.", "matched": False, "best_score": 0.0, "top": []}

        if not self.ready:
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": self.init_error or "Unknown initialization error",
            }

        top_k = int(top_k if top_k is not None else DEFAULT_TOP_K)
        top_k = max(1, min(top_k, 10))
        min_score = float(
            min_score if min_score is not None else DEFAULT_MIN_SCORE)

        # 1) exact id match first
        matched, payload = self._exact_id_match(q)
        if matched:
            return payload

        # 2) vector search
        qvec = self._embed_query(q)
        scores, idxs = self._faiss_search(qvec, top_k=top_k)

        q_tokens = _token_set(q)
        candidates: List[Dict[str, Any]] = []

        for score, idx in zip(scores.tolist(), idxs.tolist()):
            if idx < 0 or idx >= len(self.texts):
                continue
            meta = self.meta[idx] if idx < len(self.meta) else {}
            faq_id = str(meta.get("id", "")).strip() or f"row_{idx}"
            text = self.texts[idx]

            overlap = self._simple_overlap_score(
                q_tokens, self._text_tokens[idx])
            blended = float(score) * 0.85 + float(overlap) * 0.15

            candidates.append({"score": blended, "id": faq_id, "text": text})

        candidates.sort(key=lambda x: x["score"], reverse=True)

        best = candidates[0] if candidates else None
        if not best or best["score"] < min_score:
            return {
                "answer": "Sorry — I can’t find that information in the hotel FAQ.",
                "matched": False,
                "best_score": float(best["score"]) if best else 0.0,
                "top": candidates,
            }

        return {
            "answer": best["text"],
            "matched": True,
            "best_score": float(best["score"]),
            "top": candidates,
        }
