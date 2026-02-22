import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from faq_retriever import FAQRetriever

APP_NAME = "hotel_voice_agent"
APP_VERSION = os.getenv("APP_VERSION", "2026.02.22")
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.35"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

app = FastAPI(title="Hotel Voice Agent Backend", version=APP_VERSION)

# Allow frontend hosted on GitHub Pages, plus local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://mtzo777-hub.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever: Optional[FAQRetriever] = None
retriever_error: Optional[str] = None


def log_json(payload: Dict[str, Any]) -> None:
    # Cloud Run captures stdout; JSON logs are easy to filter in Cloud Logging.
    print(json.dumps(payload, ensure_ascii=False))


@app.on_event("startup")
def _startup() -> None:
    global retriever, retriever_error
    try:
        retriever = FAQRetriever()
        retriever_error = None
        log_json(
            {
                "app": APP_NAME,
                "event": "startup_ok",
                "version": APP_VERSION,
                "faq_path": getattr(retriever, "faq_path", None),
                "faq_count": len(getattr(retriever, "faq", []) or []),
            }
        )
    except Exception as e:
        retriever = None
        retriever_error = repr(e)
        log_json(
            {
                "app": APP_NAME,
                "event": "startup_degraded",
                "version": APP_VERSION,
                "error": retriever_error,
            }
        )


@app.get("/health")
def health() -> Dict[str, Any]:
    # Always return 200 so Cloud Run keeps serving even if retriever is degraded.
    if retriever is None:
        return {"status": "degraded", "retriever": "unavailable", "error": retriever_error, "version": APP_VERSION}
    return {
        "status": "ok",
        "retriever": "ok",
        "faq_count": len(getattr(retriever, "faq", []) or []),
        "version": APP_VERSION,
    }


@app.post("/faq/answer")
async def faq_answer(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """
    Request JSON:
      {
        "query": "When is check-in time?",
        "top_k": 5,
        "min_score": 0.35
      }

    Response JSON:
      {
        "request_id": "...",
        "matched": true/false,
        "answer": "...",
        "best_score": 0.0-1.0,
        "best_id": "...",
        "route": "...",
        "latency_ms": 12
      }
    """
    req_id = str(uuid.uuid4())
    t0 = time.time()

    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or DEFAULT_TOP_K)
    min_score = float(payload.get("min_score") or DEFAULT_MIN_SCORE)

    client_ip = request.headers.get(
        "x-forwarded-for") or (request.client.host if request.client else None)
    user_agent = request.headers.get("user-agent")

    log_json(
        {
            "app": APP_NAME,
            "event": "faq_answer_request",
            "request_id": req_id,
            "query": query,
            "query_len": len(query),
            "top_k": top_k,
            "min_score": min_score,
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
    )

    if retriever is None:
        resp = {
            "request_id": req_id,
            "matched": False,
            "answer": "Sorry, the FAQ knowledge base is currently unavailable.",
            "best_score": 0.0,
            "best_id": "",
            "route": "degraded_no_retriever",
            "latency_ms": int((time.time() - t0) * 1000),
            "error": retriever_error,
        }
        log_json({"app": APP_NAME, "event": "faq_answer_response",
                 "request_id": req_id, **resp})
        return resp

    try:
        result = retriever.answer(
            query=query, top_k=top_k, min_score=min_score)

        resp = {
            "request_id": req_id,
            "matched": bool(result.matched),
            "answer": result.answer,
            "best_score": float(result.best_score),
            "best_id": result.best_id,
            "route": result.route,
            "latency_ms": int((time.time() - t0) * 1000),
        }

        log_json(
            {
                "app": APP_NAME,
                "event": "faq_answer_response",
                "request_id": req_id,
                "matched": resp["matched"],
                "best_score": resp["best_score"],
                "best_id": resp["best_id"],
                "route": resp["route"],
                "latency_ms": resp["latency_ms"],
                "reason": getattr(result, "reason", None),
            }
        )
        return resp

    except Exception as e:
        resp = {
            "request_id": req_id,
            "matched": False,
            "answer": "Sorry, the service encountered an internal error.",
            "best_score": 0.0,
            "best_id": "",
            "route": "error",
            "latency_ms": int((time.time() - t0) * 1000),
            "error": repr(e),
        }
        log_json({"app": APP_NAME, "event": "faq_answer_error",
                 "request_id": req_id, "error": repr(e)})
        return resp
