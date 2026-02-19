import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever


APP_TITLE = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

# Cloud Run provides PORT env var. Default to 8080 for local/dev.
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# CORS: allow your GitHub Pages frontend + local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mtzo777-hub.github.io",
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever(base_dir="/app", faq_json_filename="faq.json")


class FAQRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Top K results")
    min_score: float = Field(0.35, ge=0.0, le=1.0,
                             description="Minimum score to accept match")


@app.get("/")
def root():
    return {"service": "hotel-voice-agent-backend", "ok": True, "version": APP_VERSION}


@app.get("/health")
def health():
    return {
        "ok": True,
        "port": PORT,
        "retriever": retriever.status(),
    }


@app.get("/healthz")
def healthz():
    # Simple health probe endpoint
    return {"ok": True}


@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    return retriever.answer(query=req.query, top_k=req.top_k, min_score=req.min_score)
