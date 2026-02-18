import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from faq_retriever import FAQRetriever

APP_NAME = "Hotel Voice Agent Backend"
APP_VERSION = "1.0.0"

FAQ_JSON_PATH = os.getenv("FAQ_JSON_PATH", "/app/faq.json")

# IMPORTANT:
# Cloud Run sets PORT; default to 8080 for local/docker.
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# CORS: allow your GitHub Pages frontend + local dev
origins = [
    "https://mtzo777-hub.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single retriever instance
retriever = FAQRetriever(faq_json_path=FAQ_JSON_PATH)


class FAQRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.35, ge=0.0, le=1.0)


@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "healthz": "/healthz",
    }


# Keep BOTH paths so you can use either one
@app.get("/health")
def health():
    return {"ok": True, "port": PORT, "retriever": retriever.status()}


@app.get("/healthz")
def healthz():
    # Some tools expect /healthz specifically
    return {"ok": True}


@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    return retriever.answer(query=req.query, top_k=req.top_k, min_score=req.min_score)
