from fastapi import FastAPI
from pydantic import BaseModel
from faq_retriever import FAQRetriever

app = FastAPI(
    title="Hotel Voice Agent Backend",
    version="2026.02.24"
)

retriever = FAQRetriever("faq.json")


class FAQRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.35


@app.get("/")
def root():
    return {"message": "Hotel Voice Agent Backend Running"}


@app.get("/health")
def health():
    return {
        "status": "ok" if retriever.is_ready else "degraded",
        "retriever": "ok" if retriever.is_ready else "unavailable",
        "faq_count": retriever.faq_count,
        "version": "2026.02.24"
    }


@app.post("/faq/answer")
def faq_answer(req: FAQRequest):
    return retriever.answer(
        req.query,
        top_k=req.top_k,
        min_score=req.min_score
    )
