# main.py
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever


APP_VERSION = os.getenv("APP_VERSION", "2026.02.22")

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=APP_VERSION,
)

# IMPORTANT: faq.json should be bundled into the container image (same folder as main.py) OR copied in Dockerfile.
FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")

retriever: Optional[FAQRetriever] = None
retriever_error: Optional[str] = None


class FaqRequest(BaseModel):
    query: str = Field(...,
                       description="User question (typed or recognized speech)")
    top_k: int = Field(
        5, ge=1, le=20, description="Number of candidates to return (debug)")
    min_score: float = Field(0.35, ge=0.0, le=1.0,
                             description="Confidence threshold (0..1)")


class FaqResponse(BaseModel):
    request_id: str
    matched: bool
    answer: str
    best_score: float
    best_id: str
    route: str
    latency_ms: int
    top: list


@app.on_event("startup")
def _startup() -> None:
    global retriever, retriever_error
    try:
        retriever = FAQRetriever(FAQ_PATH)
        retriever_error = None
    except Exception as e:
        retriever = None
        retriever_error = f"{type(e).__name__}: {e}"


@app.get("/", tags=["default"])
def root():
    return {
        "service": "hotel-voice-agent-backend",
        "version": APP_VERSION,
        "endpoints": ["/health", "/faq/answer"],
    }


@app.get("/health", tags=["default"])
def health():
    if retriever is None:
        return {"status": "degraded", "retriever": "unavailable", "error": retriever_error, "version": APP_VERSION}
    return {"status": "ok", "retriever": "ok", "faq_count": retriever.faq_count, "version": APP_VERSION}


@app.post("/faq/answer", response_model=FaqResponse, tags=["default"])
async def faq_answer(payload: FaqRequest, request: Request):
    """
    Main API used by frontend.
    Returns a stable contract and includes request_id for log correlation.
    """
    t0 = time.perf_counter()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    if retriever is None:
        # backend degraded, but still return a safe response
        return {
            "request_id": request_id,
            "matched": False,
            "answer": "Sorry, the FAQ knowledge base is currently unavailable.",
            "best_score": 0.0,
            "best_id": "",
            "route": "degraded_no_index",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "top": [],
        }

    result = retriever.answer(
        query=payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        request_id=request_id,
    )
    # Ensure latency is always filled even if retriever returned quickly
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)

    # Structured log line for Cloud Run / Cloud Logging
    # (Cloud Run will capture stdout/stderr automatically)
    client_ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")
    print(
        {
            "app": "hotel_voice_agent",
            "event": "faq_answer",
            "request_id": request_id,
            "query": payload.query,
            "top_k": payload.top_k,
            "min_score": payload.min_score,
            "matched": result.get("matched", False),
            "best_id": result.get("best_id", ""),
            "best_score": result.get("best_score", 0.0),
            "route": result.get("route", ""),
            "latency_ms": result.get("latency_ms", 0),
            "client_ip": client_ip,
            "user_agent": ua,
        }
    )
    return result
