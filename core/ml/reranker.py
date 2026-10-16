"""Optional bounded Cross-Encoder reranker.

Requirements:
- Disabled by default (behind SERVY_RERANKER_ENABLED).
- Operates ONLY on already-authorized, retrieved candidate passages (top-K).
- NEVER queries or scans the full database.
- Safe fallback if model is unavailable or encounters errors.
"""
from __future__ import annotations

import logging
import threading
from typing import List

from django.conf import settings

logger = logging.getLogger("servy.rag.reranker")

RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
_cross_encoder = None
_lock = threading.Lock()


def is_reranker_enabled() -> bool:
    return getattr(settings, "SERVY_RERANKER_ENABLED", False) is True


def get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        with _lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder(
                        RERANKER_MODEL_ID,
                        device=getattr(settings, "SERVY_EMBEDDING_DEVICE", "cpu"),
                    )
                    logger.info("Loaded CrossEncoder(%s) successfully", RERANKER_MODEL_ID)
                except Exception as exc:
                    logger.warning("Could not load CrossEncoder(%s): %s", RERANKER_MODEL_ID, exc)
                    _cross_encoder = None
    return _cross_encoder


import math
import time


def calibrated_sigmoid_score(logit: float) -> float:
    """Calibrate raw CrossEncoder logits into a well-behaved probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-float(logit)))


def rerank_candidates(query: str, candidates: List, top_k: int = 8) -> List:
    """Rerank an already-authorized, bounded list of RetrievedChunk objects.

    Enforces:
    - Bounded candidate pool (top 8 max).
    - Source authority preservation: only reranks authoritative technical documents;
      historical service cases are never promoted above authoritative guidance.
    - Score preservation: saves calibrated retrieval score in orig.retrieval_score and
      calibrated probability in orig.rerank_score.
    - Safe exception fallback to original order.
    """
    if not is_reranker_enabled() or not candidates:
        return candidates[:top_k]

    model = get_reranker()
    if model is None:
        return candidates[:top_k]

    try:
        t0 = time.perf_counter()
        # Segregate by source authority: only rerank authoritative technical documents
        docs = [c for c in candidates if c.source_kind == "knowledge_document"]
        others = [c for c in candidates if c.source_kind != "knowledge_document"]

        # Pool bound: up to 8 documents maximum
        pool = docs[: min(len(docs), 8)]
        if pool:
            pairs = [[query, c.text] for c in pool]
            raw_scores = model.predict(pairs)
            for orig, raw_score in zip(pool, raw_scores):
                logit = float(raw_score)
                # Calibrated probability via sigmoid
                prob = calibrated_sigmoid_score(logit)
                orig.rerank_score = prob
                orig.rerank_logit = logit
                orig.retrieval_score = orig.score

            pool.sort(key=lambda x: getattr(x, "rerank_score", 0.0), reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("CrossEncoder reranked %d candidates in %.1f ms", len(pool), elapsed_ms)

        combined = pool + docs[8:] + others
        return combined[:top_k]

    except Exception as exc:
        logger.warning("CrossEncoder reranking failed, falling back to original order: %s", exc)
        return candidates[:top_k]
