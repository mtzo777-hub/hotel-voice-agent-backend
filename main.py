"""
main.py - Hotel Voice Agent Backend (Cloud Run)

Endpoints:
- GET  /health        -> JSON status (use this for health checks)
- POST /faq/answer    -> retrieval endpoint
- Swagger: /docs

Cloud Run:
- listens on PORT env var (default 8080)
"""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_TITLE = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# GitHub Pages frontend + local dev
allowed_origins = [
    "https://mtzo777-hub.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever()  # reads FAQ_JSON_PATH or /app/faq.json


class FAQRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Top K results")
    min_score: float = Field(0.35, ge=0.0, le=1.0,
                             description="Minimum score to accept match")


@app.get("/")
def root():
    return {"service": APP_TITLE, "version": APP_VERSION}


@app.get("/health")
def health():
    port = int(os.getenv("PORT", "8080"))
    return {"ok": True, "port": port, "retriever": retriever.status()}


@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    return retriever.answer(query=req.query, top_k=req.top_k, min_score=req.min_score)
