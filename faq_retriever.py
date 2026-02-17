from __future__ import annotations

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("hotel-voice-agent")

# Load backend/.env early (important when module is imported by uvicorn)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL",
                        "text-embedding-3-small").strip()

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "rag_store"
INDEX_PATH = STORE_DIR / "faq.index"
TEXTS_PATH = STORE_DIR / "faq_texts.json"
META_PATH = STORE_DIR / "faq_meta.json"

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.35"))

# --- MINIMAL PATCH 1: do NOT crash the whole server at import time ---
if not OPENAI_API_KEY:
    logger.warning(
        "OPENAI_API_KEY missing. Service will run, but embedding search will be disabled "
        "(only deterministic routes / exact id matches will work)."
    )
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _norm_text(s: str) -> str:
    """
    Normalize user query to reduce sensitivity:
    - lowercase
    - normalize "check in"/"check-out" variants -> "checkin"/"checkout"
    - normalize hyphens
    - remove punctuation (keep underscore)
    - collapse whitespace
    """
    s = (s or "").strip().lower()

    # Normalize hyphens early so "check-out" becomes "check out"
    s = s.replace("-", " ")

    # Normalize forms like "check-in_time" -> "checkin_time"
    s = re.sub(r"\bcheck\s+in_", "checkin_", s)
    s = re.sub(r"\bcheck\s+out_", "checkout_", s)

    # Normalize common variants so "check in"/"check-in" behave like "checkin"
    s = re.sub(r"\bcheck\s+in\b", "checkin", s)
    s = re.sub(r"\bcheck\s+out\b", "checkout", s)

    # Remove punctuation, keep underscore
    s = re.sub(r"[^\w\s_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_id_like(s: str) -> str:
    """
    Convert a query into an id-like token:
    - "early check in policy" -> "early_checkin_policy"
    - "room types" -> "room_types"
    """
    s = _norm_text(s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _token_set(s_toggle: str) -> set[str]:
    s_toggle = _norm_text(s_toggle)
    return set([t for t in s_toggle.split(" ") if t])


class FAQRetriever:
    """
    Retrieval logic:
    1) Load FAISS index + texts + meta (must exist).
    2) Build id->text map for exact match / id-like match.
    3) search(query, top_k, min_score) returns a dict ALWAYS (no tuples).
    """

    def __init__(self) -> None:
        if not INDEX_PATH.exists() or not TEXTS_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(
                "RAG store missing. Run: python build_faiss_index.py\n"
                f"Expected:\n- {INDEX_PATH}\n- {TEXTS_PATH}\n- {META_PATH}"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            self.texts: List[str] = json.load(f)

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta: List[Dict[str, Any]] = json.load(f)

        # id -> (text, row_index)
        self.id_map: Dict[str, Tuple[str, int]] = {}
        for i, m in enumerate(self.meta):
            faq_id = str(m.get("id", "")).strip()
            if faq_id:
                self.id_map[faq_id.lower()] = (self.texts[i], i)

        # Precompute token sets for lightweight rerank
        self._text_tokens: List[set[str]] = [_token_set(t) for t in self.texts]

    def _embed_query(self, q: str) -> np.ndarray:
        # --- MINIMAL PATCH 2: no client => do NOT crash server; fail per-request gracefully ---
        if client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured for embedding search.")
        emb = client.embeddings.create(model=EMBED_MODEL, input=q)
        vec = np.array(emb.data[0].embedding, dtype=np.float32)
        vec = _l2_normalize(vec)
        return vec

    def _faiss_search(self, qvec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, idxs = self.index.search(qvec.astype(np.float32), top_k)
        return scores[0], idxs[0]

    def _keyword_route(self, qn: str) -> str | None:
        """
        Deterministic routing for common intents.
        IMPORTANT: keep this conservative; avoid overriding more specific intents.
        """

        # ----------------------------------------------------------
        # currency exchange should route to currency_exchange
        # Avoid confusing with "currency" billing in SGD.
        # ----------------------------------------------------------
        if (
            ("exchange" in qn and "currency" in qn)
            or ("money changer" in qn)
            or ("moneychanger" in qn)
            or ("forex" in qn)
        ):
            if "currency_exchange" in self.id_map:
                return "currency_exchange"

        # ---- MINIMAL STABILITY ROUTES ----
        if ("contact_number" in qn) or re.search(r"\bcontact\s*_?\s*number\b", qn):
            if "contact_phone" in self.id_map:
                return "contact_phone"

        if ("cashless" in qn) or ("contactless" in qn):
            if "cashless_payments" in self.id_map:
                return "cashless_payments"

        if "deposit" in qn:
            if "deposit_policy" in self.id_map:
                return "deposit_policy"

        if re.search(r"\b(room\s+clean(ing)?|housekeeping)\b", qn):
            if "room_cleaning" in self.id_map:
                return "room_cleaning"

        if ("room_rates" in qn) or re.search(r"\broom\s+rates?\b", qn):
            if "taxes_fees" in self.id_map:
                return "taxes_fees"

        if re.search(
            r"\b(name of the hotel|hotel name|which hotel|what hotel|hotel that i am talking to|hotel am i talking to)\b",
            qn,
        ):
            if "hotel_identity" in self.id_map:
                return "hotel_identity"

        if ("hotel" in qn and "identity" in qn):
            if not re.search(r"\b(privacy|security|id requirement|id verification|keycard|data)\b", qn):
                if "hotel_identity" in self.id_map:
                    return "hotel_identity"

        if ("hotel" in qn and "name" in qn):
            if not re.search(r"\b(address|location|phone|telephone|email|contact)\b", qn):
                if "hotel_identity" in self.id_map:
                    return "hotel_identity"

        if ("front_desk_hours" in qn) or re.search(r"\bfront\s+desk_hours\b", qn) or ("desk_hours" in qn and "front" in qn):
            if "front_desk_hours" in self.id_map:
                return "front_desk_hours"

        if ("halal_options" in qn) or ("halal_" in qn) or re.search(r"\bhalal\b", qn):
            if "halal_options" in self.id_map:
                return "halal_options"

        if re.search(r"\bhotel\s+identity\b", qn) or re.search(r"\bhotel\W*s\W*identity\b", qn):
            if "hotel_identity" in self.id_map:
                return "hotel_identity"

        if re.search(r"\brefund\b", qn):
            if "refund_policy" in self.id_map:
                return "refund_policy"

        if re.search(r"\bcancellation\s+policy\b", qn) or re.search(r"\bcancel(lation)?\b", qn):
            if "cancellation_policy" in self.id_map:
                return "cancellation_policy"

        if re.search(r"\bfront\s+desk\b", qn) and re.search(r"\b(hour|hours|open|opening)\b", qn):
            if "front_desk_hours" in self.id_map:
                return "front_desk_hours"

        if re.search(r"\bwake(\s*up)?\b", qn):
            if "wake_up_call" in self.id_map:
                return "wake_up_call"

        if re.search(r"\b(document|documents|id requirement|photo id|passport)\b", qn):
            if "checkin_documents" in self.id_map:
                return "checkin_documents"

        # Fees (early/late) should not route to checkin_time/checkout_time
        if re.search(r"\b(fee|fees|charge|charges|cost|price)\b", qn):
            return None

        if re.search(r"\b(tv|television|channels)\b", qn) and not re.search(r"\b(stream|streaming|netflix|youtube)\b", qn):
            if "room_feature_tv" in self.id_map:
                return "room_feature_tv"

        if re.search(r"\b(restaurant|cuisine)\b", qn):
            if "restaurant_cuisine" in self.id_map:
                return "restaurant_cuisine"

        if re.search(r"\b(email|e mail|mail)\b", qn):
            if "contact_email" in self.id_map:
                return "contact_email"

        if re.search(r"\b(phone|telephone|contact\s*_?\s*number|hotline)\b", qn):
            if "contact_phone" in self.id_map:
                return "contact_phone"

        if re.search(r"\b(address|location|where are you|where is the hotel)\b", qn):
            if "address" in self.id_map:
                return "address"

        if re.search(r"\b(checkin)\b", qn) and re.search(r"\b(time|when|start|starts|from)\b", qn):
            if "checkin_time" in self.id_map:
                return "checkin_time"

        if re.search(r"\b(checkout)\b", qn) and re.search(r"\b(time|when|by)\b", qn):
            if "checkout_time" in self.id_map:
                return "checkout_time"

        if re.search(r"\b(payment|pay|credit card|debit|cash|visa|mastercard|amex)\b", qn):
            for candidate in ["payment_methods", "payment_method", "accepted_payments"]:
                if candidate in self.id_map:
                    return candidate

        if re.search(r"\b(room types|room_type|types of rooms|room_types)\b", qn):
            if "room_types" in self.id_map:
                return "room_types"

        return None

    def _exact_id_match(self, q: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Try to match by:
        1) exact id token in query
        2) id-like conversion (underscored)
        3) SAFE prefix-id resolution ONLY for early_checkin / late_checkout
        """
        qn = _norm_text(q)
        id_like = _to_id_like(q)

        id_like_variants = [id_like]

        if id_like.endswith("_hour"):
            id_like_variants.append(id_like + "s")
            id_like_variants.append(id_like.replace("_hour", "_hours"))

        if id_like.endswith("_hours"):
            id_like_variants.append(id_like[:-1])

        if id_like == "front_desk_hour":
            id_like_variants.append("front_desk_hours")
        if id_like == "front_desk_hours":
            id_like_variants.append("front_desk_hour")

        tokens = set(re.findall(r"[a-z0-9_]+", qn))

        if ("front_desk" in tokens) and (("hours" in tokens) or ("hour" in tokens)):
            if "front_desk_hours" in self.id_map:
                text, _ = self.id_map["front_desk_hours"]
                return True, {
                    "answer": text,
                    "matched": True,
                    "best_score": 0.95,
                    "top": [{"score": 0.95, "id": "front_desk_hours", "text": text}],
                }

        for t in tokens:
            if t in self.id_map:
                text, _ = self.id_map[t]
                return True, {
                    "answer": text,
                    "matched": True,
                    "best_score": 1.0,
                    "top": [{"score": 1.0, "id": t, "text": text}],
                }

        for cand in id_like_variants:
            if cand in self.id_map:
                text, _ = self.id_map[cand]
                return True, {
                    "answer": text,
                    "matched": True,
                    "best_score": 1.0,
                    "top": [{"score": 1.0, "id": cand, "text": text}],
                }

        fee_words = {"fee", "fees", "charge", "charges", "cost", "price"}
        allowed_prefixes = {"early_checkin", "late_checkout"}

        for prefix in allowed_prefixes:
            if prefix in tokens or prefix == id_like or prefix.replace("_", "") in id_like.replace("_", ""):
                candidates = [k for k in self.id_map.keys()
                              if k.startswith(prefix + "_")]
                if not candidates:
                    continue

                want_fee = any(w in qn for w in fee_words)

                preferred = None
                if want_fee:
                    for k in candidates:
                        if k.endswith("_fee"):
                            preferred = k
                            break
                else:
                    for k in candidates:
                        if k.endswith("_policy"):
                            preferred = k
                            break

                chosen = preferred or candidates[0]
                text, _ = self.id_map[chosen]
                return True, {
                    "answer": text,
                    "matched": True,
                    "best_score": 0.95,
                    "top": [{"score": 0.95, "id": chosen, "text": text}],
                }

        return False, {}

    def _simple_overlap_score(self, q_tokens: set[str], doc_tokens: set[str]) -> float:
        if not q_tokens or not doc_tokens:
            return 0.0
        inter = len(q_tokens.intersection(doc_tokens))
        return inter / (len(q_tokens) + 1e-9)

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {
                "answer": "Please ask a question about the hotel.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
            }

        top_k = int(top_k if top_k is not None else DEFAULT_TOP_K)
        top_k = max(1, min(top_k, 10))
        min_score = float(
            min_score if min_score is not None else DEFAULT_MIN_SCORE)

        qn = _norm_text(q)

        # A) Conservative keyword routing
        routed_id = self._keyword_route(qn)
        if routed_id and routed_id in self.id_map:
            text, _ = self.id_map[routed_id]
            return {
                "answer": text,
                "matched": True,
                "best_score": 0.95,
                "top": [{"score": 0.95, "id": routed_id, "text": text}],
            }

        # B) Exact/prefix id matching
        matched, payload = self._exact_id_match(q)
        if matched:
            return payload

        # C) Embedding + FAISS retrieval (requires OpenAI key)
        if client is None:
            return {
                "answer": (
                    "Service is running, but embedding search is not configured "
                    "(OPENAI_API_KEY missing). Please contact the administrator."
                ),
                "matched": False,
                "best_score": 0.0,
                "top": [],
            }

        qvec = self._embed_query(q)
        scores, idxs = self._faiss_search(qvec, top_k=top_k)

        q_tokens = _token_set(q)
        candidates: List[Dict[str, Any]] = []

        for score, idx in zip(scores.tolist(), idxs.tolist()):
            if idx < 0 or idx >= len(self.texts):
                continue
            meta = self.meta[idx] if idx < len(self.meta) else {}
            faq_id = str(meta.get("id", "")).strip() or f"row_{idx}"
            text = self.texts[idx]

            overlap = self._simple_overlap_score(
                q_tokens, self._text_tokens[idx])
            blended = float(score) * 0.85 + float(overlap) * 0.15

            # Boost currency_exchange when user intent includes "exchange"
            if ("exchange" in qn or "money changer" in qn or "moneychanger" in qn or "forex" in qn):
                if faq_id.lower() == "currency_exchange":
                    blended += 0.25
                if faq_id.lower() == "currency":
                    blended -= 0.10

            candidates.append(
                {
                    "score": float(blended),
                    "raw_score": float(score),
                    "overlap": float(overlap),
                    "id": faq_id,
                    "text": text,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)

        best = candidates[0] if candidates else None
        if not best or best["score"] < min_score:
            return {
                "answer": "Sorry — I can’t find that information in the hotel FAQ.",
                "matched": False,
                "best_score": float(best["score"]) if best else 0.0,
                "top": [{"score": c["score"], "id": c["id"], "text": c["text"]} for c in candidates],
            }

        return {
            "answer": best["text"],
            "matched": True,
            "best_score": float(best["score"]),
            "top": [{"score": c["score"], "id": c["id"], "text": c["text"]} for c in candidates],
        }
