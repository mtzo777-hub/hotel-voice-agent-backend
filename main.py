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

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_TITLE = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ------------------------------------------------------------
# Logging (Cloud Run friendly structured logs)
# ------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("hotel_voice_agent")


def log_event(event: str, **fields: Any) -> None:
    """
    Emit one structured JSON log line.
    Cloud Run/Cloud Logging will parse JSON fields automatically.
    """
    payload = {"event": event, **fields}
    # Use ensure_ascii=False so smart quotes etc. remain readable in logs
    logger.info(json.dumps(payload, ensure_ascii=False))


# ------------------------------------------------------------
# 1️⃣ API production hardening
# Step 1.1: Strict CORS (GitHub Pages frontend + specific local dev ports)
# ------------------------------------------------------------

allowed_origins = [
    "https://mtzo777-hub.github.io",   # GitHub Pages
    "http://localhost:5500",           # common static server port
    "http://127.0.0.1:5500",
    "http://localhost:3000",           # if you run a local dev server
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # safer; GitHub Pages doesn't need cookies
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

retriever = FAQRetriever()  # reads FAQ_JSON_PATH or /app/faq.json

# Step 1.4: Not-found contract + stable shape
FALLBACK_ANSWER = (
    "Sorry, I don’t have that information yet. "
    "Please contact the hotel reception for assistance."
)
TIMEOUT_SECONDS = float(os.getenv("FAQ_TIMEOUT_SECONDS", "8.0"))


# ------------------------------------------------------------
# Request/Response models
# Step 1.2: Validation / clamping
# ------------------------------------------------------------

class FAQRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300,
                       description="User question")
    top_k: int = Field(5, ge=1, le=10, description="Top K results (1..10)")
    min_score: float = Field(
        0.35, ge=0.0, le=1.0, description="Minimum score to accept match (0..1)"
    )


def build_response(
    *,
    request_id: str,
    answer: str,
    matched: bool,
    best_score: float,
    top: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
    latency_ms: int,
) -> Dict[str, Any]:
    """Stable response contract for frontend (always the same keys)."""
    return {
        "request_id": request_id,
        "answer": answer,
        "matched": matched,
        "best_score": round(float(best_score), 4),
        "top": top or [],
        "error": error,
        "latency_ms": int(latency_ms),
    }


@app.get("/")
def root():
    return {"service": APP_TITLE, "version": APP_VERSION}


@app.get("/health")
def health(request: Request):
    port = int(os.getenv("PORT", "8080"))
    st = retriever.status()

    # Small health log (low frequency; okay to keep)
    log_event(
        "health",
        path=str(request.url.path),
        port=port,
        retriever_ready=bool(st.get("ready")),
        faq_count=int(st.get("faq_count", 0) or 0),
        mode=st.get("mode"),
        error=st.get("error"),
    )
    return {"ok": True, "port": port, "retriever": st}


# ------------------------------------------------------------
# Step 1.3: Timeout protection
# Step 1.4: Stable response + fallback on not-found
# + Observability: structured logs with request_id correlation
# ------------------------------------------------------------

@app.post("/faq/answer")
async def faq_answer(req: FAQRequest, request: Request):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    query = (req.query or "").strip()

    # Client IP: best-effort (Cloud Run may be behind proxies; X-Forwarded-For exists)
    xff = request.headers.get("x-forwarded-for", "")
    client_ip = (xff.split(",")[0].strip() if xff else (
        request.client.host if request.client else None))

    log_event(
        "faq_answer_request",
        request_id=request_id,
        query=query[:300],
        query_len=len(query),
        top_k=req.top_k,
        min_score=req.min_score,
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent"),
    )

    if not query:
        latency_ms = int((time.time() - t0) * 1000)
        resp = build_response(
            request_id=request_id,
            answer=FALLBACK_ANSWER,
            matched=False,
            best_score=0.0,
            top=[],
            error="empty_query",
            latency_ms=latency_ms,
        )
        log_event(
            "faq_answer_response",
            request_id=request_id,
            matched=False,
            best_score=0.0,
            best_id=None,
            route="none",
            latency_ms=latency_ms,
            error="empty_query",
        )
        return resp

    async def run_retrieval():
        # retriever.answer is sync; run in a thread so we can time out safely
        return await asyncio.to_thread(
            retriever.answer, query=query, top_k=req.top_k, min_score=req.min_score
        )

    try:
        result = await asyncio.wait_for(run_retrieval(), timeout=TIMEOUT_SECONDS)
        latency_ms = int((time.time() - t0) * 1000)

        matched = bool(result.get("matched", False))
        best_score = float(result.get("best_score", 0.0) or 0.0)
        top = result.get("top", []) if isinstance(
            result.get("top", []), list) else []
        err = result.get("error", None)

        # Extra metadata (if provided by retriever)
        best_id = result.get("best_id")
        route = result.get("route")
        eff_min_score = result.get("eff_min_score")

        if not matched:
            resp = build_response(
                request_id=request_id,
                answer=FALLBACK_ANSWER,
                matched=False,
                best_score=best_score,
                top=top,
                error=err or "no_match",
                latency_ms=latency_ms,
            )
            log_event(
                "faq_answer_response",
                request_id=request_id,
                matched=False,
                best_score=best_score,
                best_id=best_id,
                route=route,
                eff_min_score=eff_min_score,
                latency_ms=latency_ms,
                error=err or "no_match",
                top1=(top[0] if top else None),
            )
            return resp

        resp = build_response(
            request_id=request_id,
            answer=str(result.get("answer", FALLBACK_ANSWER)),
            matched=True,
            best_score=best_score,
            top=top,
            error=None,
            latency_ms=latency_ms,
        )
        log_event(
            "faq_answer_response",
            request_id=request_id,
            matched=True,
            best_score=best_score,
            best_id=best_id,
            route=route,
            eff_min_score=eff_min_score,
            latency_ms=latency_ms,
            error=None,
            top1=(top[0] if top else None),
        )
        return resp

    except asyncio.TimeoutError:
        latency_ms = int((time.time() - t0) * 1000)
        resp = build_response(
            request_id=request_id,
            answer="Sorry, the system is taking too long. Please try again.",
            matched=False,
            best_score=0.0,
            top=[],
            error="timeout",
            latency_ms=latency_ms,
        )
        log_event(
            "faq_answer_response",
            request_id=request_id,
            matched=False,
            best_score=0.0,
            best_id=None,
            route="timeout",
            latency_ms=latency_ms,
            error="timeout",
        )
        return resp

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        resp = build_response(
            request_id=request_id,
            answer=FALLBACK_ANSWER,
            matched=False,
            best_score=0.0,
            top=[],
            error=f"server_error: {e}",
            latency_ms=latency_ms,
        )
        log_event(
            "faq_answer_response",
            request_id=request_id,
            matched=False,
            best_score=0.0,
            best_id=None,
            route="server_error",
            latency_ms=latency_ms,
            error=str(e),
        )
        return resp
