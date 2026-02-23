# main.py
"""
Sunshine Hotel Voice Agent Backend (FastAPI on Cloud Run)

Endpoints:
- GET  /        : simple root for sanity check
- GET  /health  : health + retriever status
- POST /faq/answer : FAQ retrieval (BM25 + lightweight synonym/alias expansion)

Design notes:
- Stateless backend; FAQ KB is bundled in the container as faq.json and loaded at startup.
- Request/response contract is stable for both Swagger (/docs) and GitHub Pages frontend.
- Structured logs include a request_id to correlate Cloud Run logs.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever


APP_VERSION = os.getenv("APP_VERSION", "2026.02.23")
FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=APP_VERSION,
    description="FastAPI backend for Sunshine Hotel Voice Agent (BM25 FAQ retrieval).",
)

# CORS: GitHub Pages frontend + Swagger testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for exam/demo; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load retriever once on startup
retriever = FAQRetriever(FAQ_PATH)


class FAQRequest(BaseModel):
    query: str = Field(...,
                       description="User query text (from STT or typed input).")
    top_k: int = Field(
        5, ge=1, le=20, description="Number of top candidates to return.")
    min_score: float = Field(0.35, ge=0.0, le=1.0,
                             description="Match threshold (0..1).")

    class Config:
        json_schema_extra = {
            "examples": [
                {"query": "When is check-in time?", "top_k": 5, "min_score": 0.35},
                {"query": "What is the name of the hotel?",
                    "top_k": 5, "min_score": 0.35},
            ]
        }


class FAQResponse(BaseModel):
    request_id: str
    matched: bool
    answer: str
    best_score: float
    best_id: str
    route: str
    latency_ms: int
    top: list

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "request_id": "uuid",
                    "matched": True,
                    "answer": "At Sunshine Hotel Singapore, check-in starts at 2:00 PM.",
                    "best_score": 1.0,
                    "best_id": "checkin_time",
                    "route": "bm25",
                    "latency_ms": 12,
                    "top": [
                        {"id": "checkin_time", "question": "checkin_time", "score": 1.0},
                        {"id": "early_checkin_policy",
                            "question": "early_checkin_policy", "score": 0.62},
                    ],
                }
            ]
        }


def _log(event: str, payload: Dict[str, Any]) -> None:
    # Cloud Run ingests stdout; JSON lines are easiest to filter in Logs Explorer.
    obj = {"event": event, **payload}
    print(json.dumps(obj, ensure_ascii=False))


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "hotel-voice-agent-backend", "version": APP_VERSION}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if retriever.is_ready else "degraded",
        "retriever": "ok" if retriever.is_ready else "unavailable",
        "faq_count": retriever.faq_count,
        "version": APP_VERSION,
        "error": retriever.error,
    }


@app.post("/faq/answer", response_model=FAQResponse)
def faq_answer(req: FAQRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    t0 = time.time()

    _log("faq_answer_request", {
        "request_id": rid,
        "query": req.query,
        "query_len": len(req.query or ""),
        "top_k": req.top_k,
        "min_score": req.min_score,
    })

    result = retriever.answer(
        req.query, top_k=req.top_k, min_score=req.min_score, request_id=rid)
    # overwrite latency based on server wall-clock (keeps consistent)
    result["latency_ms"] = int((time.time() - t0) * 1000)

    _log("faq_answer_response", {
        "request_id": rid,
        "matched": result.get("matched"),
        "best_score": result.get("best_score"),
        "best_id": result.get("best_id"),
        "route": result.get("route"),
        "latency_ms": result.get("latency_ms"),
    })

    return result
