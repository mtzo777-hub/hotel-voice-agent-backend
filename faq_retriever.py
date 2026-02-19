import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional


# Keep stopwords conservative to avoid breaking matches.
STOPWORDS = set(
    """
a an the is are was were am be been being to of in on at for from by with as or and if then than
i you we they he she it my your our their his her its me us them
please tell give show explain
sunshine
""".split()
)


def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9\s_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_text(s).replace("-", " ").replace("_", " ")
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return toks


def _idify_query(q: str) -> str:
    """
    Convert a user query into an "id-like" string.
    Example: "When is check-in time?" -> "check_in_time" (then we normalize to checkin_time if needed)
    """
    q = _normalize_text(q)
    q = q.replace("-", " ")
    q = re.sub(r"\?", "", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Remove common leading question phrases (safe trimming)
    q = re.sub(
        r"^(what|when|where|how)\s+(is|are|do|does|can|could|should|would)\s+", "", q)
    q = re.sub(r"^(what|when|where|how)\s+", "", q)

    q = q.replace(" ", "_")
    return q


def _infer_id(query: str, idset: set) -> Optional[str]:
    """
    Very important: this is a "SAFE" mapping layer.
    It only forces a specific ID when we are confident.
    """
    qn = _normalize_text(query)
    qid = _idify_query(query)

    candidates = [
        qid,
        re.sub(r"^the_", "", qid),
        re.sub(r"^hotel_", "", qid),
        re.sub(r"^the_hotel_", "", qid),
    ]

    # Normalize common check-in/out variants
    candidates += [
        qid.replace("check_in", "checkin").replace("check_out", "checkout"),
        qid.replace("checkin", "check_in").replace("checkout", "check_out"),
    ]

    for c in candidates:
        if c in idset:
            return c

    toks = set(_tokenize(query))

    # Address / location intent → address
    if ("address" in toks or "location" in toks or "located" in toks or "directions" in toks) and "address" in idset:
        return "address"
    if ("where" in qn and "hotel" in qn) and "address" in idset:
        return "address"

    # Check-in/out time intent
    if ("check" in toks and "in" in toks and "time" in toks) and "checkin_time" in idset:
        return "checkin_time"
    if ("check" in toks and "out" in toks and "time" in toks) and "checkout_time" in idset:
        return "checkout_time"

    return None


class _BM25Index:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs_tokens = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = (sum(len(d)
                      for d in docs_tokens) / self.N) if self.N else 0.0

        df = Counter()
        for d in docs_tokens:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

        self.k1 = k1
        self.b = b

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not query_tokens or self.N == 0:
            return scores

        qtf = Counter(query_tokens)

        for i, doc in enumerate(self.docs_tokens):
            dl = len(doc)
            tf = Counter(doc)
            denom_const = self.k1 * \
                (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))

            s = 0.0
            for t, _qt in qtf.items():
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_const)

            scores[i] = s

        return scores


class FAQRetriever:
    """
    Stable retriever design (Option B best-practice for *your time constraint*):
    - No vector store required
    - Deterministic lexical retrieval (BM25)
    - Exact ID / ID-like matching overrides ranking (critical)
    - Minimal safe intent mapping for paraphrases
    """

    def __init__(self, faq_json_path: str = "/app/faq.json"):
        self.faq_json_path = faq_json_path
        self.mode = "bm25_lexical"

        self._faq: List[Dict[str, str]] = []
        self._idset: set = set()
        self._doc_tokens: List[List[str]] = []
        self._q_tokens: List[List[str]] = []
        self._index: Optional[_BM25Index] = None

        self.ready: bool = False
        self.error: Optional[str] = None

        self._load()

    def _load(self) -> None:
        try:
            with open(self.faq_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            faq: List[Dict[str, str]] = []
            for it in data:
                fid = str(it.get("id", "")).strip()
                if not fid:
                    continue
                faq.append(
                    {
                        "id": fid,
                        "question": str(it.get("question", "")).strip(),
                        "answer": str(it.get("answer", "")).strip(),
                    }
                )

            self._faq = faq
            self._idset = set(x["id"] for x in self._faq)

            # Index on (id words + question). Avoid indexing full answer (can cause false positives).
            self._doc_tokens = [
                _tokenize(x["id"].replace("_", " ") + " " + x.get("question", "")) for x in self._faq
            ]
            self._q_tokens = [_tokenize(x.get("question", ""))
                              for x in self._faq]

            self._index = _BM25Index(self._doc_tokens)
            self.ready = True
            self.error = None
        except Exception as e:
            self.ready = False
            self.error = f"{type(e).__name__}: {e}"

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": self.mode,
            "faq_json_path": self.faq_json_path,
            "faq_count": len(self._faq),
            "error": self.error,
        }

    def _overlap_boost(self, q_set: set, doc_i: int) -> float:
        if not q_set:
            return 0.0
        dt = set(self._q_tokens[doc_i]) if doc_i < len(
            self._q_tokens) else set()
        if not dt:
            return 0.0
        # fraction of query tokens appearing in the FAQ question
        return len(q_set & dt) / max(1, len(q_set))

    def answer(self, query: str, top_k: int = 5, min_score: float = 0.35) -> Dict[str, Any]:
        if not self.ready or not self._index:
            return {
                "answer": "Retriever is not ready.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": self.error or "not_loaded",
            }

        # 1) HARD OVERRIDE: exact ID / confident intent mapping
        forced_id = _infer_id(query, self._idset)
        if forced_id:
            it = next((x for x in self._faq if x["id"] == forced_id), None)
            if it:
                return {
                    "answer": it["answer"],
                    "matched": True,
                    "best_score": 1.0,
                    "top": [{"id": it["id"], "question": it["question"], "score": 1.0}],
                    "error": None,
                }

        # 2) BM25 scoring
        q_tokens = _tokenize(query)
        if not q_tokens:
            return {
                "answer": "Please ask a question.",
                "matched": False,
                "best_score": 0.0,
                "top": [],
                "error": "empty_query_after_normalization",
            }

        raw_scores = self._index.score(q_tokens)
        max_s = max(raw_scores) if raw_scores else 0.0
        norm_scores = [(s / max_s) if max_s > 0 else 0.0 for s in raw_scores]

        q_set = set(q_tokens)

        # 3) Re-rank with small overlap boost (keeps old working queries stable)
        ranked = sorted(
            range(len(norm_scores)),
            key=lambda i: (norm_scores[i] + 0.15 *
                           self._overlap_boost(q_set, i)),
            reverse=True,
        )

        ranked = ranked[: max(top_k, 10)]

        top = []
        for i in ranked[:top_k]:
            top.append(
                {
                    "id": self._faq[i]["id"],
                    "question": self._faq[i]["question"],
                    "score": round(norm_scores[i], 4),
                }
            )

        best_i = ranked[0]
        best_score = norm_scores[best_i]
        matched = best_score >= float(min_score)

        return {
            "answer": self._faq[best_i]["answer"] if matched else "No confident match found.",
            "matched": matched,
            "best_score": round(best_score, 4),
            "top": top,
            "error": None,
        }
