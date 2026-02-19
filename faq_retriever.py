"""
faq_retriever.py

Goal:
- Fast, reliable FAQ retrieval from faq.json (packaged in the Docker image).
- Works well even when user query wording doesn't exactly match the FAQ "id".
- Avoids heavy dependencies (no FAISS / embeddings required).

Approach:
1) Intent routing (rules) for the most common mismatch queries (fees vs policies, address, etc.)
2) Hybrid lexical ranking:
   - Word-level TF-IDF (unigrams + bigrams)
   - Character n-gram TF-IDF (handles typos like "message" vs "massage")
   - Small boost using fuzzy match against FAQ ids
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Helpers
# ----------------------------

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^a-z0-9\s]+")

_COMMON_NORMALIZATIONS = [
    (re.compile(r"\bcheck\s*[-]?\s*in\b", re.I), "checkin"),
    (re.compile(r"\bcheck\s*[-]?\s*out\b", re.I), "checkout"),
    (re.compile(r"\bwheel\s*chair\b", re.I), "wheelchair"),
    (re.compile(r"\bwi\s*fi\b", re.I), "wifi"),
]

_SYNONYMS = {
    "where": ["location", "address"],
    "located": ["location", "address"],
    "location": ["address"],
    "address": ["location"],
    "how much": ["price", "fee", "cost", "charge"],
    "cost": ["price", "fee", "charge"],
    "price": ["cost", "fee", "charge"],
    "fee": ["price", "cost", "charge"],
    "late checkout": ["checkout late", "late check out"],
    "early checkin": ["checkin early", "early check in"],
    "massage": ["spa", "treatment"],
    "smoking": ["smoke", "cigarette", "vape"],
    "soundproof": ["quiet", "noise", "noiseproof", "sound proof"],
    "capacity": ["how many", "max people", "max persons", "occupancy"],
    "visitor": ["guest", "visitors", "friends", "family"],
    "accessible": ["wheelchair", "disabled", "accessibility"],
}

# quick typo fixes (only whole word replacement)
_TYPO_FIXES = {
    # typo seen in your logs
    "message": "massage",
    "adress": "address",
    "chec in": "checkin",
    "chec out": "checkout",
}


def _norm(text: str) -> str:
    s = (text or "").strip().lower()

    for bad, good in _TYPO_FIXES.items():
        s = re.sub(rf"\b{re.escape(bad)}\b", good, s)

    for pat, repl in _COMMON_NORMALIZATIONS:
        s = pat.sub(repl, s)

    s = _NONWORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _id_tokens(faq_id: str) -> str:
    # "late_checkout_fee" -> "late checkout fee"
    return faq_id.replace("_", " ").strip().lower()


@dataclass
class FAQDoc:
    id: str
    text: str
    doc: str  # searchable document (id tokens + answer text)


class FAQRetriever:
    def __init__(self, faq_json_path: Optional[str] = None):
        self.faq_json_path = faq_json_path or os.getenv(
            "FAQ_JSON_PATH", "/app/faq.json")

        self._docs: List[FAQDoc] = []
        self._word_vec: Optional[TfidfVectorizer] = None
        self._char_vec: Optional[TfidfVectorizer] = None
        self._word_X = None
        self._char_X = None

        self._ready: bool = False
        self._error: Optional[str] = None

        self._load()

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "mode": "hybrid_lexical",
            "faq_json_path": self.faq_json_path,
            "faq_count": len(self._docs),
            "error": self._error,
        }

    def _load(self) -> None:
        try:
            if not os.path.exists(self.faq_json_path):
                self._ready = False
                self._error = f"faq.json not found at {self.faq_json_path}"
                return

            raw = json.loads(
                open(self.faq_json_path, "r", encoding="utf-8").read())
            docs: List[FAQDoc] = []
            for item in raw:
                faq_id = str(item.get("id", "")).strip()
                text = str(item.get("text", "")).strip()
                if not faq_id or not text:
                    continue
                doc = f"{_id_tokens(faq_id)}. {text}"
                docs.append(FAQDoc(id=faq_id, text=text, doc=doc))

            if not docs:
                self._ready = False
                self._error = "faq.json loaded but contained 0 valid Q/A pairs"
                return

            self._docs = docs

            self._word_vec = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                stop_words="english",
            )
            self._char_vec = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
            )

            corpus = [d.doc for d in self._docs]
            self._word_X = self._word_vec.fit_transform(corpus)
            self._char_X = self._char_vec.fit_transform(corpus)

            self._ready = True
            self._error = None

        except Exception as e:
            self._ready = False
            self._error = f"Failed to load faq.json: {type(e).__name__}: {e}"

    def _has_id(self, faq_id: str) -> bool:
        return any(d.id == faq_id for d in self._docs)

    def _route(self, q_norm: str) -> Optional[str]:
        """
        Return FAQ id if a rule confidently matches; otherwise None.
        These rules address the exact mismatch cases from your attached logs.
        """

        def has(*words: str) -> bool:
            return all(w in q_norm for w in words)

        # Address / location
        if ("address" in q_norm) or has("where", "hotel") or ("located" in q_norm and "hotel" in q_norm):
            return "address" if self._has_id("address") else None

        # Hotel identity / name
        if has("hotel", "name") or ("what is" in q_norm and "hotel" in q_norm and "called" in q_norm):
            return "hotel_identity" if self._has_id("hotel_identity") else None

        # Late checkout (fee vs policy)
        if "late" in q_norm and "checkout" in q_norm:
            if any(x in q_norm for x in ["how much", "price", "cost", "fee", "charge"]):
                if self._has_id("late_checkout_fee"):
                    return "late_checkout_fee"
            if self._has_id("late_checkout_policy"):
                return "late_checkout_policy"

        # Early check-in (fee vs policy)
        if "early" in q_norm and "checkin" in q_norm:
            if any(x in q_norm for x in ["how much", "price", "cost", "fee", "charge"]):
                if self._has_id("early_checkin_fee"):
                    return "early_checkin_fee"
            if self._has_id("early_checkin"):
                return "early_checkin"

        # Wheelchair / accessibility
        if "wheelchair" in q_norm or "accessible" in q_norm or "accessibility" in q_norm or "disabled" in q_norm:
            if self._has_id("wheelchair_access"):
                return "wheelchair_access"

        # Soundproof rooms
        if "soundproof" in q_norm or "noiseproof" in q_norm or ("quiet" in q_norm and "room" in q_norm):
            if self._has_id("room_feature_soundproofing"):
                return "room_feature_soundproofing"

        # Smoking policy
        if "smoking" in q_norm or "smoke" in q_norm or "vape" in q_norm or "cigarette" in q_norm:
            if self._has_id("smoking_policy"):
                return "smoking_policy"

        # Room capacity / max people
        if any(x in q_norm for x in ["how many", "max", "maximum", "capacity", "occupancy", "persons", "people"]) and "room" in q_norm:
            if self._has_id("room_capacity"):
                return "room_capacity"

        # Visitor policy
        if any(x in q_norm for x in ["visitor", "visitors", "guest", "guests", "friends", "family"]) and any(
            x in q_norm for x in ["allowed", "permit", "policy", "can i", "can we", "bring"]
        ):
            if self._has_id("visitor_policy"):
                return "visitor_policy"

        # No-show / cancellation
        if "no" in q_norm and "show" in q_norm:
            if self._has_id("no_show_policy"):
                return "no_show_policy"

        # Massage services (typo: "message")
        if "massage" in q_norm or ("spa" in q_norm and "service" in q_norm):
            if self._has_id("service_massage"):
                return "service_massage"

        return None

    def _expand_query(self, q_norm: str) -> str:
        expanded = [q_norm]
        for k, syns in _SYNONYMS.items():
            if k in q_norm:
                expanded.extend(syns)
        # dedupe but keep order
        return " ".join(dict.fromkeys(expanded))

    def _hybrid_rank(self, q_norm: str, top_k: int) -> List[Tuple[int, float]]:
        if not self._ready or not self._docs:
            return []

        q_exp = self._expand_query(q_norm)

        w = cosine_similarity(self._word_vec.transform(
            [q_exp]), self._word_X).ravel()
        c = cosine_similarity(self._char_vec.transform(
            [q_exp]), self._char_X).ravel()

        scores = 0.65 * w + 0.30 * c

        for i, d in enumerate(self._docs):
            idtok = _id_tokens(d.id)
            f = fuzz.token_set_ratio(q_norm, idtok) / 100.0  # 0..1
            scores[i] += 0.05 * f

            # Bias fees when query asks "how much/price/cost"
            if any(x in q_norm for x in ["how much", "price", "cost", "fee", "charge"]):
                if d.id.endswith("_fee"):
                    scores[i] += 0.03

            # Bias policies if explicitly asks policy/rules
            if "policy" in q_norm or "rule" in q_norm:
                if "policy" in d.id:
                    scores[i] += 0.03

        idxs = scores.argsort()[::-1][: max(1, top_k)]
        return [(int(i), float(scores[i])) for i in idxs]

    def search(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
        q_norm = _norm(query)

        if not self._ready:
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": self._error or "Unknown error",
            }

        # 1) Rules first (fixes your fee/policy/address/massage mismatches)
        routed_id = self._route(q_norm)
        if routed_id:
            d = next((x for x in self._docs if x.id == routed_id), None)
            if d:
                return {
                    "answer": d.text,
                    "matched": True,
                    "best_score": 1.0,
                    "top": [{"id": d.id, "score": 1.0}],
                    "error": None,
                }

        # 2) Hybrid lexical ranking
        ranked = self._hybrid_rank(q_norm, top_k=top_k)
        top: List[Dict[str, Any]] = []

        best_doc = None
        best_score = 0.0

        for idx, score in ranked:
            d = self._docs[idx]
            top.append({"id": d.id, "score": round(score, 4)})
            if best_doc is None:
                best_doc = d
                best_score = score

        matched = bool(best_doc) and (best_score >= float(min_score))

        return {
            "answer": best_doc.text if (best_doc and matched) else "No confident match found.",
            "matched": matched,
            "best_score": round(float(best_score), 4),
            "top": top,
            "error": None if matched else f"Below min_score={min_score}. Try lowering min_score or rephrasing.",
        }
