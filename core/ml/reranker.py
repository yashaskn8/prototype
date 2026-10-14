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


def rerank_candidates(query: str, candidates: List, top_k: int = 10) -> List:
    """Rerank an already-authorized, bounded list of RetrievedChunk objects.

    If reranker is disabled or unavailable, returns the original list sorted by score.
    """
    if not is_reranker_enabled() or not candidates:
        return candidates[:top_k]

    model = get_reranker()
    if model is None:
        return candidates[:top_k]

    try:
        # Bounded subset: only rerank up to 20 candidates
        pool = candidates[: min(len(candidates), 20)]
        pairs = [[query, c.text] for c in pool]
        scores = model.predict(pairs)

        scored = list(zip(pool, scores))
        # Sort descending by cross-encoder score
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for orig, ce_score in scored[:top_k]:
            # Retain the chunk metadata, updating score
            orig.score = float(ce_score)
            reranked.append(orig)

        return reranked
    except Exception as exc:
        logger.warning("CrossEncoder reranking failed, falling back to original order: %s", exc)
        return candidates[:top_k]
