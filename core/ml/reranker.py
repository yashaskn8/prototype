"""Optional, authorization-bounded cross-encoder reranker.

The V5 BGE cross-encoder is an opt-in LOCAL model, not an embedding model.
Never load a remote model when a local checkpoint was requested. Loading or
scoring failures preserve the already-authorized candidate ordering.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from pathlib import Path
from typing import List

from django.conf import settings

logger = logging.getLogger("servy.rag.reranker")
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
_cross_encoder = None
_loaded_config = None
_lock = threading.Lock()


def is_reranker_enabled() -> bool:
    return getattr(settings, "SERVY_RERANKER_ENABLED", False) is True


def _local_checkpoint():
    configured = str(getattr(settings, "SERVY_RERANKER_MODEL_PATH", "") or "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(settings.BASE_DIR) / path
    return path.resolve()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_reranker():
    """Cache by configuration, verify optional weights digest, load offline."""
    global _cross_encoder, _loaded_config
    path = _local_checkpoint()
    device = str(getattr(settings, "SERVY_RERANKER_DEVICE", "cpu"))
    expected_sha = str(getattr(settings, "SERVY_RERANKER_EXPECTED_SHA256", "") or "").lower().strip()
    length = 256
    key = (str(path) if path else RERANKER_MODEL_ID, device, length, expected_sha)
    if _loaded_config == key and _cross_encoder is not None:
        return _cross_encoder
    with _lock:
        if _loaded_config == key and _cross_encoder is not None:
            return _cross_encoder
        # Do not reuse a previously loaded model when configuration changes.
        _cross_encoder = None
        _loaded_config = None
        try:
            from sentence_transformers import CrossEncoder
            if path is not None:
                weights = path / "model.safetensors"
                if not path.is_dir() or not weights.is_file() or not (path / "config.json").is_file():
                    raise FileNotFoundError(f"Missing V5 model files in {path}")
                if expected_sha:
                    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
                        raise ValueError("Invalid SERVY_RERANKER_EXPECTED_SHA256")
                    if _sha256_file(weights) != expected_sha:
                        raise ValueError("V5 weights SHA-256 mismatch")
                model = CrossEncoder(
                    str(path), device=device, max_length=length,
                    local_files_only=True,
                )
            else:
                # Legacy opt-in mode is unchanged when no V5 checkpoint is configured.
                model = CrossEncoder(RERANKER_MODEL_ID, device=device)
            _cross_encoder = model
            _loaded_config = key
            logger.info(
                "CrossEncoder loaded: %s (device=%s)",
                str(path) if path else RERANKER_MODEL_ID, device,
            )
            return model
        except Exception as exc:
            logger.warning(
                "Reranker unavailable; returning original authorized ranking: %s",
                exc,
            )
            return None


def calibrated_sigmoid_score(logit: float) -> float:
    """Stable monotone calibration for ranking metadata (not a probability guarantee)."""
    x = float(logit)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    y = math.exp(x)
    return y / (1.0 + y)


def rerank_candidates(query: str, candidates: List, top_k: int = 8) -> List:
    """Rerank only pre-authorized technical chunks in a bounded top-N pool.

    No source discovery, authorization bypass, or changes to historical cases.
    Fail closed to the existing order if scoring fails.
    """
    if not is_reranker_enabled() or not candidates:
        return candidates[:top_k]
    model = get_reranker()
    if model is None:
        return candidates[:top_k]
    try:
        t0 = time.perf_counter()
        docs = [c for c in candidates if c.source_kind == "knowledge_document"]
        others = [c for c in candidates if c.source_kind != "knowledge_document"]
        pool_size = int(getattr(settings, "SERVY_RERANKER_POOL_SIZE", 8))
        if not 1 <= pool_size <= 8:
            raise ValueError("SERVY_RERANKER_POOL_SIZE must be 1..8")
        pool = docs[:pool_size]
        # Validate the full prediction BEFORE mutating any result objects.
        if pool:
            pairs = [[query, c.text] for c in pool]
            raw = model.predict(pairs)
            scores = [float(x) for x in raw]
            if len(scores) != len(pool) or not all(math.isfinite(x) for x in scores):
                raise ValueError("Invalid or non-finite CrossEncoder scores")
            for candidate, logit in zip(pool, scores):
                candidate.rerank_score = calibrated_sigmoid_score(logit)
                candidate.rerank_logit = logit
                candidate.retrieval_score = candidate.score
            pool.sort(key=lambda c: c.rerank_logit, reverse=True)
        logger.debug(
            "CrossEncoder reranked %d authorized chunks in %.1f ms",
            len(pool), (time.perf_counter() - t0) * 1000,
        )
        return (pool + docs[pool_size:] + others)[:top_k]
    except Exception as exc:
        logger.warning("CrossEncoder scoring failed; preserving original ranking: %s", exc)
        return candidates[:top_k]
