from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_NAME = "Hotel Voice Agent Backend"

app = FastAPI(title=APP_NAME, version="1.0.0")

# Allow your GitHub Pages frontend (and local dev)
allowed_origins = [
    "https://mtzo777-hub.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
extra = os.getenv("CORS_ORIGINS", "").strip()
if extra:
    allowed_origins.extend([o.strip() for o in extra.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever(
    faq_json_path=os.getenv("FAQ_JSON_PATH", "faq.json"),
    rag_store_dir=os.getenv("RAG_STORE_DIR", "rag_store"),
)


class FAQRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.35, ge=0.0, le=1.0)


@app.get("/")
def root():
    return {"service": APP_NAME, "docs": "/docs", "health": "/healthz"}


@app.get("/health")
@app.get("/healthz")
def healthz():
    return {"ok": True, "retriever": retriever.status}


@app.post("/faq/answer")
def faq_answer(payload: FAQRequest):
    result = retriever.answer(
        payload.query, top_k=payload.top_k, min_score=payload.min_score)
    return {
        "answer": result.answer,
        "matched": result.matched,
        "best_score": result.best_score,
        "top": result.top,
        "error": result.error,
    }
