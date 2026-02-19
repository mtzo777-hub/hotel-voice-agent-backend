"""
main.py
-------
FastAPI backend for Hotel Voice Agent.

Key Cloud Run fixes:
- Always bind to host 0.0.0.0 and port = $PORT (default 8080)
- Provide BOTH /health and /healthz
- Retriever does not require OpenAI; API key is only needed for optional features
- CORS includes your GitHub Pages frontend
"""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever


APP_TITLE = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

PORT = int(os.getenv("PORT", "8080"))
FAQ_JSON_PATH = os.getenv("FAQ_JSON_PATH", "/app/faq.json")

# Optional (only needed if you add TTS later)
# from Secret Manager -> Cloud Run env via --set-secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mtzo777-hub.github.io",
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever(faq_json_path=FAQ_JSON_PATH)


class FAQRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Top K results")
    min_score: float = Field(0.35, ge=0.0, le=100.0,
                             description="Min score to accept match")


@app.get("/")
def root():
    return {"message": "ok", "service": APP_TITLE, "version": APP_VERSION}


@app.get("/health")
def health():
    st = retriever.status()
    return {
        "ok": True,
        "port": PORT,
        "retriever": {
            "ready": st.ready,
            "mode": st.mode,
            "faq_json_path": st.faq_json_path,
            "faq_count": st.faq_count,
            "error": st.error,
        },
    }


@app.get("/healthz")
def healthz():
    return health()


@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    return retriever.search(req.query, top_k=req.top_k, min_score=req.min_score)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
