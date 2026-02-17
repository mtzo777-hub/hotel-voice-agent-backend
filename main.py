from __future__ import annotations
from faq_retriever import FAQRetriever
from openai import OpenAI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from typing import Optional

import io
import os
import time
import uuid
import logging
logger = logging.getLogger("hotel-voice-agent")


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy").strip()

if not OPENAI_API_KEY:
    logger.warning(
        "OPENAI_API_KEY is missing. Running in FAQ-only mode (no LLM).")
tts_client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="Sunshine Hotel Voice Agent Backend",
    version="2.0 (Responsible AI Enhanced)",
)

# --------------------------------------------------
# Logging (Responsible AI: minimal + audit friendly)
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hotel-voice-agent")

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

retriever = FAQRetriever()

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
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


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

    try:
        result = retriever.search(
            query,
            top_k=top_k,
            min_score=min_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {e}")

    backend_ms = int((time.time() - t0) * 1000)

    matched = result.get("matched", False)
    best_score = result.get("best_score", 0.0)

    # --------------------------------------------------
    # Responsible AI: Safe refusal + human handoff
    # --------------------------------------------------

    if not matched:
        result["answer"] = (
            "I'm an AI assistant for Sunshine Hotel and can only provide "
            "information from the official hotel FAQ. "
            "For bookings or special requests, please contact our front desk directly."
        )

    # --------------------------------------------------
    # Attach metadata (Transparency + Auditability)
    # --------------------------------------------------

    result["request_id"] = request_id
    result["backend_total_ms"] = backend_ms
    result["min_score"] = min_score
    result["top_k"] = top_k

    # --------------------------------------------------
    # Privacy-friendly structured log
    # (Do NOT log full user query text)
    # --------------------------------------------------

    logger.info(
        f"[FAQ] id={request_id} "
        f"matched={matched} "
        f"score={best_score:.3f} "
        f"latency_ms={backend_ms}"
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
