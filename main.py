# main.py
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
FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=APP_VERSION,
    description="FastAPI backend for Sunshine Hotel Voice Agent (BM25 FAQ retrieval).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # exam/demo friendly
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever(FAQ_PATH)


def _log(event: str, payload: Dict[str, Any]) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False))


def _rid(req: Request) -> str:
    # try to reuse upstream request id when available
    hdr = req.headers.get(
        "x-request-id") or req.headers.get("x-cloud-trace-context")
    if hdr:
        return hdr.split("/")[0]
    return str(uuid.uuid4())


class FAQRequest(BaseModel):
    query: str = Field(..., examples=["When is check-in time?"])
    top_k: int = Field(5, ge=1, le=10, examples=[5])
    min_score: float = Field(0.35, ge=0.0, le=1.0, examples=[0.35])

    class Config:
        json_schema_extra = {"example": {
            "query": "When is check-in time?", "top_k": 5, "min_score": 0.35}}


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
    return {"service": "hotel-voice-agent-backend", "version": APP_VERSION, "endpoints": ["/health", "/faq/answer"]}


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
async def faq_answer(request: Request, body: FAQRequest) -> Dict[str, Any]:
    rid = _rid(request)
    t0 = time.time()

    _log("faq_answer_request", {
        "request_id": rid,
        "query": body.query,
        "top_k": body.top_k,
        "min_score": body.min_score,
        "user_agent": request.headers.get("user-agent"),
    })

    result = retriever.answer(
        body.query, top_k=body.top_k, min_score=body.min_score, request_id=rid)
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
