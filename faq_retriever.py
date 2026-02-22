"""
faq_retriever.py
Lexical FAQ retriever with safer "not-found" behavior.

Key changes vs earlier version:
- Removed "normalize by max(score)" which made best_score ~ 1.0 for almost any query.
- Introduced confidence score based on:
  (a) BM25 raw score magnitude,
  (b) token overlap ratio (coverage),
  (c) distinctiveness vs the 2nd-best candidate.
- Returns matched=False when confidence < min_score (safe fallback contract).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi


# --- Tokenization / normalization ------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# A small, pragmatic stopword list (kept local to avoid heavy deps).
# Add conversational fillers to reduce false positives for "ok", "hmm", etc.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "its", "may",
    "me", "my", "of", "on", "or", "our", "please", "should", "so", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",

    # fillers / short acknowledgements (often cause accidental matches)
    "ok", "okay", "hmm", "um", "uh", "yeah", "yep", "nope", "alright", "right",

    # courtesy / endings (frontend should handle, but keep backend safe too)
    "thanks", "thank", "bye", "goodbye",
}


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    toks = [t for t in toks if t not in _STOPWORDS and len(t) >= 2]
    return toks


# --- Data model -----------------------------------------------------------------

@dataclass(frozen=True)
class FaqEntry:
    id: str
    question: str
    answer: str
    aliases: Tuple[str, ...]


# --- Retriever ------------------------------------------------------------------

class FaqRetriever:
    """
    Loads faq.json and provides BM25 + overlap-based retrieval.
    """

    def __init__(self, faq_path: str):
        self.faq_path = faq_path
        self.entries: List[FaqEntry] = []
        self._docs_tokens: List[List[str]] = []
        self._docs_token_sets: List[set] = []
        self._bm25: Optional[BM25Okapi] = None

        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.faq_path):
            raise FileNotFoundError(f"faq.json not found at: {self.faq_path}")

        with open(self.faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # support either {"faqs":[...]} or direct list
        items = data.get("faqs", data)
        if not isinstance(items, list):
            raise ValueError(
                "faq.json must be a list or have key 'faqs' as a list")

        entries: List[FaqEntry] = []
        docs_tokens: List[List[str]] = []
        docs_sets: List[set] = []

        for obj in items:
            if not isinstance(obj, dict):
                continue

            faq_id = str(obj.get("id", "")).strip() or str(
                obj.get("key", "")).strip()
            question = str(obj.get("question", "")).strip()
            answer = str(obj.get("answer", "")).strip()

            # Some earlier datasets used "text" as the answer field
            if not answer and obj.get("text"):
                answer = str(obj.get("text", "")).strip()

            aliases_raw = obj.get("aliases") or obj.get("alias") or []
            if isinstance(aliases_raw, str):
                aliases_raw = [aliases_raw]
            aliases = tuple(str(a).strip()
                            for a in aliases_raw if str(a).strip())

            # Skip unusable rows
            if not faq_id or (not question and not answer):
                continue

            if not question:
                # If only answer exists, use id as question label so it is still retrievable
                question = faq_id.replace("_", " ").strip()

            entries.append(FaqEntry(id=faq_id, question=question,
                           answer=answer, aliases=aliases))

            # Retrieval text: question + id + aliases (NOT answer) to reduce false matches.
            retrieval_text = " ".join(
                [faq_id.replace("_", " "), question, *aliases])
            toks = tokenize(retrieval_text)
            docs_tokens.append(toks)
            docs_sets.append(set(toks))

        if not entries:
            raise ValueError("No valid FAQ entries found in faq.json")

        self.entries = entries
        self._docs_tokens = docs_tokens
        self._docs_token_sets = docs_sets
        self._bm25 = BM25Okapi(docs_tokens)

    # ---- scoring ---------------------------------------------------------------

    @staticmethod
    def _bounded(raw: float, k: float = 3.0) -> float:
        """Map a positive raw score to (0,1) with diminishing returns."""
        if raw <= 0:
            return 0.0
        return raw / (raw + k)

    @staticmethod
    def _distinctiveness(best_raw: float, second_raw: float) -> float:
        """How clearly best beats second best (0..1)."""
        if best_raw <= 0:
            return 0.0
        d = (best_raw - second_raw) / (best_raw + 1e-9)
        if d < 0:
            d = 0.0
        if d > 1:
            d = 1.0
        return d

    def _confidence(
        self,
        query_tokens: List[str],
        best_idx: int,
        best_raw: float,
        second_raw: float,
    ) -> Tuple[float, float]:
        """
        Return (confidence_0to1, overlap_ratio).
        """
        if not query_tokens:
            return 0.0, 0.0

        qset = set(query_tokens)
        dset = self._docs_token_sets[best_idx]

        # coverage of query tokens
        overlap = len(qset & dset) / max(len(qset), 1)
        base = self._bounded(best_raw, k=3.0)
        distinct = self._distinctiveness(best_raw, second_raw)

        # Combine (empirically stable):
        # - overlap gates confidence strongly
        # - base captures BM25 strength
        # - distinct penalizes ambiguous matches
        conf = overlap * base * (0.6 + 0.4 * distinct)

        # Extra penalty: very low overlap should not pass even with moderate BM25.
        if overlap < 0.34:
            conf *= 0.4

        return conf, overlap

    # ---- public API ------------------------------------------------------------

    def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Returns a response dict:
        {
          matched: bool,
          answer: str,
          best_id: str,
          best_score: float,   # 0..1 confidence
          best_raw: float,     # BM25 raw score
          overlap: float,      # 0..1
          top: [ {id, score, raw, overlap, question}, ... ]   # limited debug info
        }
        """
        if self._bm25 is None:
            raise RuntimeError("Retriever not initialized")

        q = (query or "").strip()
        q_tokens = tokenize(q)

        if not q_tokens:
            return {
                "matched": False,
                "answer": "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                "best_id": None,
                "best_score": 0.0,
                "best_raw": 0.0,
                "overlap": 0.0,
                "top": [],
                "error": "empty_or_stopword_query",
            }

        raw_scores = self._bm25.get_scores(
            q_tokens)  # list[float], length = docs
        if not raw_scores:
            return {
                "matched": False,
                "answer": "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                "best_id": None,
                "best_score": 0.0,
                "best_raw": 0.0,
                "overlap": 0.0,
                "top": [],
                "error": "no_scores",
            }

        # top-k indices by raw score
        k = max(1, min(int(top_k), len(raw_scores)))
        top_idx = sorted(range(len(raw_scores)),
                         key=lambda i: raw_scores[i], reverse=True)[:k]

        best_idx = top_idx[0]
        best_raw = float(raw_scores[best_idx])
        second_raw = float(raw_scores[top_idx[1]]) if len(top_idx) > 1 else 0.0

        best_score, best_overlap = self._confidence(
            q_tokens, best_idx, best_raw, second_raw)

        # Build debug top list with per-candidate overlap/confidence
        top_list: List[Dict[str, Any]] = []
        for i in top_idx:
            raw_i = float(raw_scores[i])
            score_i, ov_i = self._confidence(
                q_tokens, i, raw_i, best_raw if i != best_idx else second_raw)
            ent = self.entries[i]
            top_list.append({
                "id": ent.id,
                "question": ent.question,
                "raw": round(raw_i, 6),
                "overlap": round(ov_i, 4),
                "score": round(score_i, 6),
            })

        ent_best = self.entries[best_idx]

        if best_score < float(min_score):
            return {
                "matched": False,
                "answer": "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
                "best_id": ent_best.id,
                "best_score": round(best_score, 6),
                "best_raw": round(best_raw, 6),
                "overlap": round(best_overlap, 4),
                "top": top_list,
                "error": "below_min_score",
            }

        # Matched
        return {
            "matched": True,
            "answer": ent_best.answer or "Sorry, I don’t have that information yet. Please contact the hotel reception for assistance.",
            "best_id": ent_best.id,
            "best_score": round(best_score, 6),
            "best_raw": round(best_raw, 6),
            "overlap": round(best_overlap, 4),
            "top": top_list,
            "error": None,
        }
