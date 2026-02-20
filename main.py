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
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_TITLE = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

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
TIMEOUT_SECONDS = 8.0


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
def health():
    port = int(os.getenv("PORT", "8080"))
    return {"ok": True, "port": port, "retriever": retriever.status()}


# ------------------------------------------------------------
# Step 1.3: Timeout protection
# Step 1.4: Stable response + fallback on not-found
# ------------------------------------------------------------

@app.post("/faq/answer")
async def faq_answer(req: FAQRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # Extra guard (even though pydantic min_length=1 exists)
    query = (req.query or "").strip()
    if not query:
        latency_ms = int((time.time() - t0) * 1000)
        return build_response(
            request_id=request_id,
            answer=FALLBACK_ANSWER,
            matched=False,
            best_score=0.0,
            top=[],
            error="empty_query",
            latency_ms=latency_ms,
        )

    async def run_retrieval():
        # retriever.answer is sync; run it in a thread so we can time out safely
        return await asyncio.to_thread(
            retriever.answer, query=query, top_k=req.top_k, min_score=req.min_score
        )

    try:
        result = await asyncio.wait_for(run_retrieval(), timeout=TIMEOUT_SECONDS)
        latency_ms = int((time.time() - t0) * 1000)

        # Enforce stable contract regardless of retriever output shape
        matched = bool(result.get("matched", False))
        best_score = float(result.get("best_score", 0.0) or 0.0)
        top = result.get("top", []) if isinstance(
            result.get("top", []), list) else []
        err = result.get("error", None)

        if not matched:
            return build_response(
                request_id=request_id,
                answer=FALLBACK_ANSWER,
                matched=False,
                best_score=best_score,
                top=top,
                error=err or "no_match",
                latency_ms=latency_ms,
            )

        return build_response(
            request_id=request_id,
            answer=str(result.get("answer", FALLBACK_ANSWER)),
            matched=True,
            best_score=best_score,
            top=top,
            error=None,
            latency_ms=latency_ms,
        )

    except asyncio.TimeoutError:
        latency_ms = int((time.time() - t0) * 1000)
        return build_response(
            request_id=request_id,
            answer="Sorry, the system is taking too long. Please try again.",
            matched=False,
            best_score=0.0,
            top=[],
            error="timeout",
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return build_response(
            request_id=request_id,
            answer=FALLBACK_ANSWER,
            matched=False,
            best_score=0.0,
            top=[],
            error=f"server_error: {e}",
            latency_ms=latency_ms,
        )
