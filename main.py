import time
import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faq_retriever import FAQRetriever

APP_VERSION = "2026.02.25"

# Cloud Run / stdout logging (Logs Explorer picks this up)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hotel_voice_agent")

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=APP_VERSION
)

# ✅ CORS: allow your GitHub Pages frontend to call Cloud Run
allowed_origins = [
    "https://mtzo777-hub.github.io",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever("faq.json")


class FAQRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.35


@app.get("/")
def root():
    return {"message": "Hotel Voice Agent Backend Running", "version": APP_VERSION}


@app.get("/health")
def health():
    return {
        "status": "ok" if retriever.is_ready else "degraded",
        "retriever": "ok" if retriever.is_ready else "unavailable",
        "faq_count": retriever.faq_count,
        "version": APP_VERSION
    }


@app.post("/faq/answer")
async def faq_answer(req: FAQRequest, request: Request):
    """
    Same functional behavior as before:
    returns retriever.answer(query, top_k, min_score)
    Only added: request_id + latency_ms fields and structured logs.
    """
    start = time.time()
    request_id = uuid.uuid4().hex

    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    # Log request (structured dict shows nicely in Logs Explorer)
    logger.info({
        "event": "faq_answer_request",
        "request_id": request_id,
        "query": req.query,
        "query_len": len(req.query or ""),
        "top_k": req.top_k,
        "min_score": req.min_score,
        "client_ip": client_ip,
        "user_agent": user_agent,
    })

    # ✅ keep same call as your current working version
    result = retriever.answer(
        req.query,
        top_k=req.top_k,
        min_score=req.min_score
    )

    latency_ms = int((time.time() - start) * 1000)

    # Log response metadata
    logger.info({
        "event": "faq_answer_response",
        "request_id": request_id,
        "matched": result.get("matched"),
        "best_id": result.get("best_id"),
        "best_score": result.get("best_score"),
        "route": result.get("route"),
        "latency_ms": latency_ms,
        "error": None,
    })

    # Add helpful tracing fields (does not break frontend; extra fields are safe)
    result["request_id"] = request_id
    result["latency_ms"] = latency_ms

    return result
