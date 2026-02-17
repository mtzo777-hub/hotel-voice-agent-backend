from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL",
                        "text-embedding-3-small").strip()
FAQ_JSON_PATH_ENV = os.getenv("FAQ_JSON_PATH", "faq.json").strip()

BASE_DIR = Path(__file__).resolve().parent
FAQ_JSON_PATH = (BASE_DIR / FAQ_JSON_PATH_ENV).resolve()

STORE_DIR = BASE_DIR / "rag_store"
INDEX_PATH = STORE_DIR / "faq.index"
TEXTS_PATH = STORE_DIR / "faq_texts.json"
META_PATH = STORE_DIR / "faq_meta.json"

BATCH_SIZE = 128


def load_faq_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"FAQ JSON not found: {path}")

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    # Accept list OR {"items": [...]}
    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    if not isinstance(data, list):
        raise ValueError("faq.json must be a LIST or {'items': LIST}")

    items: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue

        # Your restored format: { "id": "...", "text": "..." }
        _id = str(obj.get("id", "")).strip()
        _text = str(obj.get("text", "")).strip()

        if _id and _text:
            items.append({"id": _id, "text": _text})

    if not items:
        raise ValueError(
            "No valid FAQ items found in faq.json (need fields: id + text)")

    return items


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # OpenAI embeddings call
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)

    # Normalize for cosine similarity (IndexFlatIP on normalized vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in backend/.env")

    print(f"[build] Read FAQ from: {FAQ_JSON_PATH}")
    items = load_faq_items(FAQ_JSON_PATH)
    print(f"[build] Raw items count: {len(items)}")
    print(f"[build] Usable items (non-empty text): {len(items)}")

    for s in items[:3]:
        preview = (s["text"][:70] + "...") if len(s["text"]
                                                  ) > 70 else s["text"]
        print(f"[build] sample id={s['id']} text='{preview}'")

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"[build] Embedding model: {EMBED_MODEL}")

    # Batch embeddings
    all_vecs: List[np.ndarray] = []
    texts = [it["text"] for it in items]

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        vecs = embed_texts(client, batch)
        all_vecs.append(vecs)

    vectors = np.vstack(all_vecs)
    n, d = vectors.shape
    print(f"[build] Got embeddings: N={n}, D={d}")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # FAISS index (cosine via inner product on normalized vectors)
    index = faiss.IndexFlatIP(d)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    TEXTS_PATH.write_text(json.dumps(
        texts, ensure_ascii=False, indent=2), encoding="utf-8")
    META_PATH.write_text(json.dumps(
        items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[build] Wrote FAISS index -> {INDEX_PATH}")
    print(f"[build] Wrote texts      -> {TEXTS_PATH}")
    print(f"[build] Wrote meta       -> {META_PATH}")
    print("[build] DONE")


if __name__ == "__main__":
    main()
