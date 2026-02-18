import os
import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional OpenAI (only needed for /tts)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from faq_retriever import FAQRetriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hotel-voice-agent-backend")

app = FastAPI(title="Hotel Voice Agent Backend", version="1.0.0")

# ----------------------------
# CORS
# ----------------------------
# In production, it's better to set this explicitly.
# Example: ALLOWED_ORIGINS="https://your-frontend-domain.com,https://another.com"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Keep your previous local defaults; you can widen later via ALLOWED_ORIGINS
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Models
# ----------------------------


class FAQRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.35


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


# ----------------------------
# Lazy singletons (IMPORTANT)
# ----------------------------
@lru_cache(maxsize=1)
def get_retriever() -> FAQRetriever:
    """
    Lazy init so the app can start even if:
    - rag_store files are missing
    - OPENAI_API_KEY is missing
    The endpoint will return a clean error instead of crashing the container.
    """
    return FAQRetriever()


def get_tts_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


# ----------------------------
# Health check
# ----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


# ----------------------------
# FAQ endpoint
# ----------------------------
@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    try:
        retriever = get_retriever()
        result = retriever.search(
            req.query, top_k=req.top_k, min_score=req.min_score)
        return result
    except Exception as e:
        logger.exception("FAQ retriever error")
        raise HTTPException(
            status_code=500, detail=f"Retriever error: {str(e)}")


# ----------------------------
# Optional TTS endpoint
# ----------------------------
@app.post("/tts")
def tts(req: TTSRequest):
    client = get_tts_client()
    if client is None:
        # Do NOT crash startup if key is missing.
        raise HTTPException(
            status_code=501,
            detail="TTS is not configured. Set OPENAI_API_KEY to enable /tts.",
        )

    try:
        # Example TTS call (adjust if your project uses a different OpenAI API)
        audio = client.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip(),
            voice=req.voice,
            input=req.text,
        )
        # Return base64 or bytes is up to your frontend; here we return raw bytes.
        # type: ignore
        return {"ok": True, "message": "TTS generated", "bytes": audio.read()}
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


# ----------------------------
# Local dev runner
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
