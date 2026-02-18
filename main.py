from __future__ import annotations

import io
import os
import time
import uuid
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from faq_retriever import FAQRetriever

try:
    # If OPENAI_API_KEY is missing, we will run in FAQ-only mode.
    from openai import OpenAI
except Exception:  # extremely defensive; app should still boot
    OpenAI = None  # type: ignore


# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hotel-voice-agent")


# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy").strip()


# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Sunshine Hotel Voice Agent Backend",
    version="2.1 (Cloud Run resilient boot)",
)

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://mtzo777-hub.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Safe initialization (DO NOT crash on boot)
# --------------------------------------------------
retriever: Optional[FAQRetriever] = None

try:
    retriever = FAQRetriever()
    logger.info("FAQRetriever initialized successfully.")
except Exception as e:
    # This is the key fix: Cloud Run must still start even if rag_store is missing.
    retriever = None
    logger.exception(
        "FAQRetriever failed to initialize. Running in SAFE FAQ-only mode. "
        "Likely missing rag_store artifacts inside the container. Error: %s",
        e,
    )

tts_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        tts_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI TTS client initialized.")
    except Exception as e:
        tts_client = None
        logger.exception(
            "Failed to init OpenAI client. TTS disabled. Error: %s", e)
else:
    logger.warning(
        "OPENAI_API_KEY missing (or OpenAI SDK unavailable). TTS disabled.")


# --------------------------------------------------
# Models
# --------------------------------------------------
class FAQRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.35


class TTSRequest(BaseModel):
    text: str


# --------------------------------------------------
# Startup log (helps confirm Cloud Run boot)
# --------------------------------------------------
@app.on_event("startup")
def _startup_log():
    port = os.getenv("PORT", "8080")
    logger.info("App startup complete. Expecting to listen on PORT=%s", port)


# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/version")
def version():
    return {"version": "CI/CD test working 🎉", "tts_enabled": bool(tts_client), "retriever_enabled": bool(retriever)}


# --------------------------------------------------
# FAQ Retrieval Endpoint
# --------------------------------------------------
@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    top_k = int(req.top_k if req.top_k is not None else 5)
    min_score = float(req.min_score if req.min_score is not None else 0.35)

    # If retriever failed to init, return safe fallback (still 200 OK)
    if retriever is None:
        backend_ms = int((time.time() - t0) * 1000)
        logger.warning(
            f"[FAQ] id={request_id} retriever_disabled latency_ms={backend_ms}")
        return {
            "matched": False,
            "best_score": 0.0,
            "answer": (
                "I'm an AI assistant for Sunshine Hotel. The FAQ search index is not available right now. "
                "For bookings or special requests, please contact our front desk directly."
            ),
            "request_id": request_id,
            "backend_total_ms": backend_ms,
            "min_score": min_score,
            "top_k": top_k,
        }

    try:
        result = retriever.search(
            query,
            top_k=top_k,
            min_score=min_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {e}")

    backend_ms = int((time.time() - t0) * 1000)

    matched = bool(result.get("matched", False))
    best_score = float(result.get("best_score", 0.0))

    # Safe refusal + human handoff
    if not matched:
        result["answer"] = (
            "I'm an AI assistant for Sunshine Hotel and can only provide "
            "information from the official hotel FAQ. "
            "For bookings or special requests, please contact our front desk directly."
        )

    # Attach metadata
    result["request_id"] = request_id
    result["backend_total_ms"] = backend_ms
    result["min_score"] = min_score
    result["top_k"] = top_k

    # Privacy-friendly structured log (do NOT log full query text)
    logger.info(
        f"[FAQ] id={request_id} matched={matched} score={best_score:.3f} latency_ms={backend_ms}"
    )

    return result


# --------------------------------------------------
# TTS Endpoint
# --------------------------------------------------
@app.post("/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    if tts_client is None:
        # Don’t crash; return a clear error.
        raise HTTPException(
            status_code=503,
            detail="TTS is disabled (OPENAI_API_KEY missing or OpenAI client failed to initialize)."
        )

    try:
        audio = tts_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
        )

        return StreamingResponse(
            io.BytesIO(audio.content),
            media_type="audio/mpeg"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")
