"""
main.py - FastAPI backend for Hotel Voice Agent

Endpoints:
- GET  /           : root
- GET  /health     : health + retriever status
- POST /faq/answer : query FAQ JSON via FAQRetriever

Notes:
- Cloud Run expects the container to listen on $PORT (default 8080).
- We only keep /health (remove /healthz as requested).
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_NAME = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# CORS: allow your GitHub Pages frontend + local dev
allowed_origins = [
    "https://mtzo777-hub.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever()


class FAQRequest(BaseModel):
    query: str = Field(..., title="Query", description="User question")
    top_k: int = Field(5, ge=1, le=20, title="Top K",
                       description="Top K results")
    min_score: float = Field(0.35, ge=0.0, le=1.0, title="Min Score",
                             description="Minimum score to accept match")


@app.get("/")
def root():
    return {"message": APP_NAME, "version": APP_VERSION}


@app.get("/health")
def health():
    port = int(os.getenv("PORT", "8080"))
    return {"ok": True, "port": port, "retriever": retriever.status()}


@app.post("/faq/answer")
def faq_answer(payload: FAQRequest):
    return retriever.search(payload.query, top_k=payload.top_k, min_score=payload.min_score)
