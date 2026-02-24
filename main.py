from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faq_retriever import FAQRetriever

APP_VERSION = "2026.02.25"

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version=APP_VERSION
)

# ✅ CORS: allow your GitHub Pages frontend to call Cloud Run
# Add localhost origins if you test locally too.
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
def faq_answer(req: FAQRequest):
    return retriever.answer(
        req.query,
        top_k=req.top_k,
        min_score=req.min_score
    )
