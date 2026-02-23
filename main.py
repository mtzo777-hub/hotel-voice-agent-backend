"""
main.py

FastAPI backend for Sunshine Hotel Voice Agent (FAQ retrieval).
Designed for Cloud Run deployment with structured logs (Cloud Logging).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_VERSION = os.getenv("APP_VERSION", "2026.02.23")

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=APP_VERSION,
    description="BM25 FAQ retrieval backend for Sunshine Hotel Voice Agent.",
)

# CORS: allow GitHub Pages frontend and local dev. (Safe here because no auth / no secrets.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")
retriever = FAQRetriever(FAQ_PATH)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _request_id(req: Request) -> str:
    rid = req.headers.get(
        "x-request-id") or req.headers.get("x-cloud-trace-context")
    if rid:
        return rid.split("/")[0]
    return str(uuid.uuid4())


def _log(event: str, payload: Dict[str, Any]) -> None:
    # Cloud Run captures stdout/stderr into Cloud Logging.
    record = {"event": event, **payload}
    print(json.dumps(record, ensure_ascii=False))


class FAQRequest(BaseModel):
    query: str = Field(..., examples=["When is check-in time?"])
    top_k: int = Field(5, ge=1, le=10, examples=[5])
    min_score: float = Field(0.35, ge=0.0, le=1.0, examples=[0.35])

    class Config:
        # This restores the Swagger "Edit Value" defaults nicely.
        json_schema_extra = {
            "example": {"query": "When is check-in time?", "top_k": 5, "min_score": 0.35}
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


@app.get("/")
def root() -> Dict[str, Any]:
    # Restore GET / Root (you said you prefer the old Swagger layout)
    return {
        "service": "hotel-voice-agent-backend",
        "version": APP_VERSION,
        "endpoints": ["/health", "/faq/answer"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    if retriever.is_ready:
        return {"status": "ok", "retriever": "ok", "faq_count": retriever.faq_count, "version": APP_VERSION}
    return {"status": "degraded", "retriever": "unavailable", "error": "retriever_not_ready", "version": APP_VERSION}


@app.post("/faq/answer", response_model=FAQResponse)
async def faq_answer(req: Request, body: FAQRequest) -> Dict[str, Any]:
    rid = _request_id(req)

    t0 = _now_ms()
    _log(
        "faq_answer_request",
        {
            "request_id": rid,
            "query": body.query,
            "query_len": len(body.query or ""),
            "top_k": body.top_k,
            "min_score": body.min_score,
            "client_ip": req.client.host if req.client else None,
            "user_agent": req.headers.get("user-agent"),
        },
    )

    result = retriever.answer(
        body.query, top_k=body.top_k, min_score=body.min_score, request_id=rid)
    result["latency_ms"] = _now_ms() - t0  # end-to-end time

    _log(
        "faq_answer_response",
        {
            "request_id": rid,
            "matched": result.get("matched"),
            "best_score": result.get("best_score"),
            "best_id": result.get("best_id"),
            "route": result.get("route"),
            "latency_ms": result.get("latency_ms"),
        },
    )

    return result
