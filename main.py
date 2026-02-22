"""
main.py
FastAPI backend for Sunshine Hotel Voice Agent (Cloud Run).

- POST /faq/answer: lexical retrieval over bundled faq.json (BM25 + overlap confidence)
- GET  /health: health probe

Observability:
- Emits structured JSON logs with a per-request request_id.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FaqRetriever


# --- Logging --------------------------------------------------------------------

logger = logging.getLogger("hotel_voice_agent")
logger.setLevel(logging.INFO)

# Ensure log output goes to stdout (Cloud Run / Cloud Logging)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_json(payload: Dict[str, Any]) -> None:
    """
    Cloud Logging parses JSON when printed as a single line string.
    We log as INFO by default.
    """
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info(str(payload))


# --- App ------------------------------------------------------------------------

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version="1.0.0",
)

# Allow GitHub Pages / local testing; for production you can lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load FAQ dataset -----------------------------------------------------------

FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")
retriever = FaqRetriever(FAQ_PATH)


# --- Schemas --------------------------------------------------------------------

class FaqAnswerRequest(BaseModel):
    query: str = Field(..., description="User query (typed or STT text).")
    top_k: int = Field(
        5, ge=1, le=20, description="How many candidates to evaluate.")
    min_score: float = Field(
        0.35, ge=0.0, le=1.0, description="Minimum confidence to accept a match.")


class FaqAnswerResponse(BaseModel):
    request_id: str
    matched: bool
    answer: str
    best_id: Optional[str] = None
    best_score: float = 0.0
    best_raw: float = 0.0
    overlap: float = 0.0
    route: str = "bm25"
    error: Optional[str] = None


# --- Endpoints ------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/faq/answer", response_model=FaqAnswerResponse)
async def faq_answer(req: FaqAnswerRequest, request: Request) -> FaqAnswerResponse:
    t0 = time.perf_counter()

    # Request id: allow client to provide one, else generate.
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    log_json({
        "event": "faq_answer_request",
        "request_id": rid,
        "route": "bm25",
        "query": (req.query or "")[:300],
        "query_len": len(req.query or ""),
        "top_k": req.top_k,
        "min_score": req.min_score,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "path": str(request.url.path),
        "method": request.method,
    })

    try:
        result = retriever.answer(
            req.query, top_k=req.top_k, min_score=req.min_score)
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        log_json({
            "event": "faq_answer_error",
            "request_id": rid,
            "route": "bm25",
            "latency_ms": latency_ms,
            "error": repr(e),
        })
        # Safe fallback response
        return FaqAnswerResponse(
            request_id=rid,
            matched=False,
            answer="Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
            best_id=None,
            best_score=0.0,
            best_raw=0.0,
            overlap=0.0,
            route="bm25",
            error="exception",
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    resp = FaqAnswerResponse(
        request_id=rid,
        matched=bool(result.get("matched")),
        answer=str(result.get("answer") or ""),
        best_id=result.get("best_id"),
        best_score=float(result.get("best_score") or 0.0),
        best_raw=float(result.get("best_raw") or 0.0),
        overlap=float(result.get("overlap") or 0.0),
        route="bm25",
        error=result.get("error"),
    )

    log_json({
        "event": "faq_answer_response",
        "request_id": rid,
        "route": "bm25",
        "matched": resp.matched,
        "best_id": resp.best_id,
        "best_score": resp.best_score,
        "best_raw": resp.best_raw,
        "overlap": resp.overlap,
        "latency_ms": latency_ms,
        "error": resp.error,
    })

    return resp
