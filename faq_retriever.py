import json
import os
import time
import uuid
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class FAQRetriever:
    def __init__(self, faq_path: str = "faq.json"):
        self.faq_path = faq_path
        self.is_ready = False
        self.faq_count = 0
        self.bm25 = None
        self.entries = []

        self._load_faq()

    # -----------------------------------------------------
    # Load FAQ safely (Cloud Run must NOT crash)
    # -----------------------------------------------------
    def _load_faq(self):
        if not os.path.exists(self.faq_path):
            print("faq.json not found")
            return

        try:
            with open(self.faq_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print("faq.json format invalid")
                return

            self.entries = data
            self.faq_count = len(data)

            if BM25Okapi is None:
                print("rank_bm25 not installed")
                return

            corpus = [self._tokenize(item["id"]) for item in data]
            self.bm25 = BM25Okapi(corpus)

            self.is_ready = True
            print(f"FAQ loaded: {self.faq_count} entries")

        except Exception as e:
            print("Failed loading FAQ:", e)

    # -----------------------------------------------------
    def _tokenize(self, text: str):
        text = text.replace("_", " ")
        text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text.lower())
        return text.split()

    # -----------------------------------------------------
    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35):

        request_id = uuid.uuid4().hex
        start = time.time()

        if not self.is_ready or not self.bm25:
            return {
                "request_id": request_id,
                "matched": False,
                "answer": "Sorry, the FAQ knowledge base is currently unavailable.",
                "best_score": 0,
                "best_id": None,
                "route": "degraded_no_index",
                "latency_ms": 0,
                "top": []
            }

        tokens = self._tokenize(query)
        raw_scores = self.bm25.get_scores(tokens)

        # CRITICAL FIX FOR CLOUD
        scores = raw_scores.tolist()

        scored = list(zip(self.entries, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_results = scored[:top_k]

        best_entry, best_score = top_results[0]

        matched = best_score >= min_score

        latency = int((time.time() - start) * 1000)

        if matched:
            answer_text = best_entry["text"]
            route = "bm25"
        else:
            answer_text = "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance."
            route = "not_found"

        return {
            "request_id": request_id,
            "matched": matched,
            "answer": answer_text,
            "best_score": round(float(best_score), 4),
            "best_id": best_entry["id"],
            "route": route,
            "latency_ms": latency,
            "top": [
                {
                    "id": e["id"],
                    "question": e["id"],
                    "score": round(float(s), 4)
                }
                for e, s in top_results
            ]
        }
