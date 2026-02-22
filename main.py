"""
Hotel Voice Agent Backend (FastAPI on Cloud Run)

Endpoints:
- GET /           : simple banner (kept for Swagger "Root")
- GET /health     : service + retriever readiness
- POST /faq/answer: lexical retrieval over faq.json with not-found contract

Logging:
- JSON-ish structured logs with request_id for Cloud Logging correlation
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever, RetrievalResult


APP_NAME = "hotel_voice_agent"

# --- FastAPI app ---
app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=os.getenv("APP_VERSION", "2026.02.22"),
    description="FastAPI backend for the Sunshine Hotel Voice Agent (BM25 lexical retrieval over faq.json).",
)

# CORS: allow GitHub Pages + local dev (safe default: allow all; tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever (initialized at startup)
retriever: Optional[FAQRetriever] = None


# --- Request / Response models (ensures Swagger shows query/top_k/min_score) ---
class FAQAnswerRequest(BaseModel):
    query: str = Field(..., examples=["When is check-in time?"])
    top_k: int = Field(5, ge=1, le=20, examples=[5])
    min_score: float = Field(0.35, ge=0.0, le=10.0, examples=[0.35])


class FAQTopItem(BaseModel):
    id: str
    question: str
    score: float


class FAQAnswerResponse(BaseModel):
    request_id: str
    matched: bool
    answer: str
    best_score: float
    best_id: str
    route: str
    latency_ms: int
    top: list[FAQTopItem] = []


def log_event(event: Dict[str, Any]) -> None:
    """
    Print structured logs. Cloud Run captures stdout to Cloud Logging.
    Keep it one-line JSON for easy filtering.
    """
    try:
        print(json.dumps(event, ensure_ascii=False))
    except Exception:
        # Fallback: never crash due to logging
        print(str(event))


@app.on_event("startup")
def startup() -> None:
    global retriever
    # expect faq.json in repo root (same folder as main.py)
    faq_path = os.getenv("FAQ_PATH", "faq.json")
    try:
        retriever = FAQRetriever(faq_path=faq_path)
        log_event(
            {
                "app": APP_NAME,
                "event": "startup",
                "status": "ok",
                "faq_path": faq_path,
                "faq_count": retriever.faq_count,
            }
        )
    except Exception as e:
        retriever = None
        log_event(
            {
                "app": APP_NAME,
                "event": "startup",
                "status": "error",
                "faq_path": faq_path,
                "error": repr(e),
            }
        )


@app.get("/")
def root() -> Dict[str, str]:
    # Keep this so Swagger shows GET / Root (your “old version”)
    return {"service": "hotel-voice-agent-backend", "status": "ok"}


@app.get("/health")
def health() -> Dict[str, Any]:
    global retriever
    if retriever is None:
        return {"status": "degraded", "retriever": "unavailable", "error": "retriever_not_initialized"}

    if not retriever.is_ready:
        return {"status": "degraded", "retriever": "unavailable", "error": retriever.last_error or "unknown"}

    return {
        "status": "ok",
        "retriever": "ok",
        "faq_count": retriever.faq_count,
        "version": app.version,
    }


@app.post("/faq/answer", response_model=FAQAnswerResponse)
async def faq_answer(payload: FAQAnswerRequest, request: Request) -> FAQAnswerResponse:
    """
    Not-found contract:
    - If retrieval confidence is low, return matched=false with a safe fallback answer.
    - Never hallucinate.
    """
    global retriever

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    t0 = time.perf_counter()

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    log_event(
        {
            "app": APP_NAME,
            "event": "faq_answer_request",
            "request_id": request_id,
            "query": payload.query,
            "query_len": len(payload.query or ""),
            "top_k": payload.top_k,
            "min_score": payload.min_score,
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
    )

    # If retriever is not ready, do NOT attempt retrieval; return safe degraded response
    if retriever is None or not retriever.is_ready:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp = FAQAnswerResponse(
            request_id=request_id,
            matched=False,
            answer="Sorry, the FAQ knowledge base is currently unavailable.",
            best_score=0.0,
            best_id="",
            route="degraded_no_index",
            latency_ms=latency_ms,
            top=[],
        )
        log_event(
            {
                "app": APP_NAME,
                "event": "faq_answer_response",
                "request_id": request_id,
                "matched": resp.matched,
                "best_score": resp.best_score,
                "best_id": resp.best_id,
                "route": resp.route,
                "latency_ms": resp.latency_ms,
                "error": getattr(retriever, "last_error", None),
            }
        )
        return resp

    # Normal retrieval
    result: RetrievalResult = retriever.answer(
        query=payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    resp = FAQAnswerResponse(
        request_id=request_id,
        matched=result.matched,
        answer=result.answer,
        best_score=result.best_score,
        best_id=result.best_id,
        route=result.route,
        latency_ms=latency_ms,
        top=[FAQTopItem(id=t["id"], question=t["question"],
                        score=float(t["score"])) for t in result.top],
    )

    log_event(
        {
            "app": APP_NAME,
            "event": "faq_answer_response",
            "request_id": request_id,
            "matched": resp.matched,
            "best_score": resp.best_score,
            "best_id": resp.best_id,
            "route": resp.route,
            "latency_ms": resp.latency_ms,
        }
    )

    return resp
