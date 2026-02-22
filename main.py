"""
main.py
FastAPI backend for Sunshine Hotel Voice Agent (Cloud Run).

Robust startup:
- Never crashes the container if faq.json or dependencies are missing.
- /health reports degraded status with error details.
- /faq/answer returns safe fallback when retriever is unavailable.

Observability:
- Structured JSON logs with a per-request request_id.
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

# Import retriever module, but do NOT initialize at import time.
retriever = None
retriever_load_error: Optional[str] = None


# --- Logging --------------------------------------------------------------------

logger = logging.getLogger("hotel_voice_agent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_json(payload: Dict[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info(str(payload))


# --- App ------------------------------------------------------------------------

app = FastAPI(title="Hotel Voice Agent Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")


@app.on_event("startup")
def _startup_load_retriever() -> None:
    """
    Cloud Run must be able to start listening even if data is missing.
    We load retriever here and handle any failure gracefully.
    """
    global retriever, retriever_load_error
    try:
        from faq_retriever import FaqRetriever  # local import
        retriever = FaqRetriever(FAQ_PATH)
        retriever_load_error = None
        log_json({
            "event": "startup_retriever_loaded",
            "faq_path": FAQ_PATH,
            "status": "ok",
        })
    except Exception as e:
        retriever = None
        retriever_load_error = repr(e)
        log_json({
            "event": "startup_retriever_failed",
            "faq_path": FAQ_PATH,
            "status": "degraded",
            "error": retriever_load_error,
        })


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


SAFE_FALLBACK = "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."


# --- Endpoints ------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Cloud Run health. If retriever failed, still return 200 but indicate degraded.
    """
    if retriever is None:
        return {"status": "degraded", "retriever": "unavailable", "error": retriever_load_error}
    return {"status": "ok", "retriever": "ready", "faq_path": FAQ_PATH}


@app.post("/faq/answer", response_model=FaqAnswerResponse)
async def faq_answer(req: FaqAnswerRequest, request: Request) -> FaqAnswerResponse:
    t0 = time.perf_counter()
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

    # If retriever isn't available, return safe fallback but keep service alive.
    if retriever is None:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        log_json({
            "event": "faq_answer_response",
            "request_id": rid,
            "route": "bm25",
            "matched": False,
            "best_id": None,
            "best_score": 0.0,
            "best_raw": 0.0,
            "overlap": 0.0,
            "latency_ms": latency_ms,
            "error": "retriever_unavailable",
            "startup_error": retriever_load_error,
        })
        return FaqAnswerResponse(
            request_id=rid,
            matched=False,
            answer=SAFE_FALLBACK,
            route="bm25",
            error="retriever_unavailable",
        )

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
        return FaqAnswerResponse(
            request_id=rid,
            matched=False,
            answer=SAFE_FALLBACK,
            route="bm25",
            error="exception",
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    resp = FaqAnswerResponse(
        request_id=rid,
        matched=bool(result.get("matched")),
        answer=str(result.get("answer") or SAFE_FALLBACK),
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
