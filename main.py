import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever


app = FastAPI(title="Hotel Voice Agent Backend", version="1.0.0")

# Allow your GitHub Pages frontend + local dev.
# If you want to be extra strict, keep only these origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mtzo777-hub.github.io",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = FAQRetriever(faq_json_path=os.getenv(
    "FAQ_JSON_PATH", "/app/faq.json"))


class FAQRequest(BaseModel):
    query: str = Field(..., title="Query", description="User question")
    top_k: int = Field(5, ge=1, le=20, title="Top K",
                       description="Top K results")
    min_score: float = Field(
        0.35, ge=0.0, le=1.0, title="Min Score", description="Minimum match score")


@app.get("/")
def root():
    return {"ok": True, "service": "hotel-voice-agent-backend"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "port": int(os.getenv("PORT", "8080")),
        "retriever": retriever.status(),
    }


@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    """
    Main retrieval endpoint used by Swagger + your GitHub Pages frontend.
    """
    return retriever.answer(query=req.query, top_k=req.top_k, min_score=req.min_score)
