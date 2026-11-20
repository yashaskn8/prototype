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


class OnnxV5:
    """The verified V5 checkpoint exported to ONNX fp32 (see export_v5_onnx).

    Same tokenizer, max_length and raw logits as the PyTorch CrossEncoder; the
    export was accepted only with identical ranking and |logit diff| <= 1e-3.
    """

    def __init__(self, directory: Path, threads: int):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(directory / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def predict(self, pairs, activation_fn=None, show_progress_bar=False, **_):
        import numpy as np
        encoded = self.tokenizer([list(p) for p in pairs], padding=True, truncation=True, max_length=256,
                                 return_tensors="np")
        logits = self.session.run(["logits"], {"input_ids": encoded["input_ids"].astype(np.int64),
                                               "attention_mask": encoded["attention_mask"].astype(np.int64)})[0]
        return logits[:, 0]


def _verified_onnx_dir(expected_sha: str):
    """The configured ONNX export directory, only if its manifest and every file hash verify."""
    import json
    configured = str(getattr(settings, "SERVY_RERANKER_ONNX_DIR", "") or "").strip()
    if not configured:
        raise FileNotFoundError("SERVY_RERANKER_ONNX_DIR is not configured")
    directory = Path(configured).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "servy-v5-onnx-v1" or manifest.get("precision") != "fp32"
            or manifest.get("source_weights_sha256") != expected_sha):
        raise ValueError("V5 ONNX manifest does not match the pinned V5 weights")
    for name, digest in (manifest.get("files") or {}).items():
        if _sha256_file(directory / name) != digest:
            raise ValueError(f"V5 ONNX file hash mismatch: {name}")
    if "model.onnx.data" not in (manifest.get("files") or {}):
        raise ValueError("V5 ONNX manifest is incomplete")
    return directory


def reranker_runtime() -> str:
    """The configured V5 runtime (what get_reranker will try first)."""
    backend = str(getattr(settings, "SERVY_RERANKER_BACKEND", "torch") or "torch").lower()
    return "onnx-fp32" if backend == "onnx" else "torch-fp32"


def active_runtime() -> str | None:
    """The V5 runtime actually loaded, or None before loading."""
    if _cross_encoder is None:
        return None
    return "onnx-fp32" if isinstance(_cross_encoder, OnnxV5) else "torch-fp32"


def get_reranker():
    """Cache by configuration, verify optional weights digest, load offline."""
    global _cross_encoder, _loaded_config
    path = _local_checkpoint()
    device = str(getattr(settings, "SERVY_RERANKER_DEVICE", "cpu"))
    expected_sha = str(getattr(settings, "SERVY_RERANKER_EXPECTED_SHA256", "") or "").lower().strip()
    length = 256
    backend = reranker_runtime()
    key = (str(path) if path else "", device, length, expected_sha, backend,
           str(getattr(settings, "SERVY_RERANKER_ONNX_DIR", "") or ""))
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
                from core.ml.model_lock import MODEL_LOAD_LOCK
                model = None
                if backend == "onnx-fp32":
                    try:
                        onnx_dir = _verified_onnx_dir(expected_sha or SERVY_V5_WEIGHTS_SHA256)
                        with MODEL_LOAD_LOCK:
                            model = OnnxV5(onnx_dir, int(getattr(settings, "SERVY_RERANKER_THREADS", 8)))
                    except Exception as exc:
                        # Correctness first: the verified PyTorch checkpoint is the rollback.
                        logger.warning("V5 ONNX runtime unavailable (%s); using the PyTorch checkpoint",
                                       type(exc).__name__)
                        model = None
                if model is None:
                    with MODEL_LOAD_LOCK:
                        model = CrossEncoder(
                            str(path), device=device, max_length=length,
                            local_files_only=True,
                        )
            else:
                # V5 is the only reranker. Without its local checkpoint path
                # there is no network or legacy-model fallback.
                raise ValueError("SERVY_RERANKER_MODEL_PATH_REQUIRED_FOR_PINNED_CHECKPOINT")
            _cross_encoder = model
            _loaded_config = key
            logger.info(
                "CrossEncoder loaded: %s (device=%s)",
                str(path), device,
            )
            return model
        except Exception as exc:
            logger.warning(
                "Reranker unavailable; returning original authorized ranking: %s",
                exc,
            )
            return None


def require_verified_v5_reranker() -> None:
    """Fail startup unless the configured, SHA-pinned V5 runs real scoring.

    Called only for explicit SERVY_RERANKER_REQUIRED deployments. Optional
    development deployments retain their existing observable fallback.
    """
    if not is_reranker_enabled():
        raise RuntimeError("SERVY_RERANKER_REQUIRED needs SERVY_RERANKER_ENABLED=1")
    if _local_checkpoint() is None:
        raise RuntimeError("SERVY_RERANKER_REQUIRED needs a local model path")
    expected = str(getattr(settings, "SERVY_RERANKER_EXPECTED_SHA256", "") or "").strip().lower()
    if expected != SERVY_V5_WEIGHTS_SHA256:
        raise RuntimeError("SERVY_RERANKER_REQUIRED needs the pinned V5 weights SHA-256")

    # Use the serving scorer, not a different default-activation prediction.
    # No database, tenant data, network API, or production corpus is needed.
    from types import SimpleNamespace

    probe = SimpleNamespace(
        source_kind="knowledge_document",
        text="Check the hydraulic pressure gauge before servicing the pump.",
        score=0.5,
    )
    ranked = rerank_candidates("How can I inspect pump pressure?", [probe], top_k=1)
    if (
        len(ranked) != 1
        or ranked[0] is not probe
        or getattr(probe, "reranker_model", None) != "servy_v5_bge_verified"
        or not math.isfinite(float(getattr(probe, "rerank_logit", float("nan"))))
        or not math.isfinite(float(getattr(probe, "reranker_score", float("nan"))))
    ):
        raise RuntimeError("Required V5 failed to load or score; refusing to start")


# Exact digest of the exploratory V5 BGE reranker, verified on 2026-10-19.
# This is public checkpoint identity metadata, not a secret or a promotion signal.
SERVY_V5_WEIGHTS_SHA256 = "b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda"


def _applied_model_label() -> str:
    """Report what actually scored candidates; do not confuse it with embeddings."""
    if _local_checkpoint() is None:
        return "legacy_ms_marco"
    expected = str(getattr(settings, "SERVY_RERANKER_EXPECTED_SHA256", "") or "").lower().strip()
    return "servy_v5_bge_verified" if expected == SERVY_V5_WEIGHTS_SHA256 else "local_cross_encoder"


def calibrated_sigmoid_score(logit: float) -> float:
    """Stable monotone calibration for ranking metadata (not a probability guarantee)."""
    x = float(logit)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    y = math.exp(x)
    return y / (1.0 + y)



def build_contextual_candidate_text(candidate, asset_context: str = "") -> str:
    """Add scoped asset/document/section metadata without altering source text."""
    metadata = []
    if asset_context:
        metadata.append(f"Asset: {asset_context}")
    title = str(getattr(candidate, "title", "") or "").strip()
    heading = str(getattr(candidate, "heading", "") or "").strip()
    doc_type = str(getattr(candidate, "doc_type", "") or "").strip()
    if title:
        metadata.append(f"Document: {title}")
    if heading:
        metadata.append(f"Section: {heading}")
    if doc_type:
        metadata.append(f"Document type: {doc_type}")
    passage = str(getattr(candidate, "text", "") or "").strip()
    prefix = "\n".join(metadata)
    budget = max(800, 3200 - len(prefix))
    passage = passage[:budget]
    return f"{prefix}\nPassage: {passage}" if prefix else passage


def _predict(model, pairs):
    """Score pairs with V5 (loading it first when not yet loaded). Pure computation."""
    import torch
    model = model or get_reranker()
    if model is None:
        raise RuntimeError("V5 reranker failed to load")
    with torch.inference_mode():
        return model.predict(pairs, activation_fn=torch.nn.Identity(), show_progress_bar=False)


def rerank_candidates(query: str, candidates: List, top_k: int = 8, *, context_aware: bool = False, asset_context: str = "") -> List:
    """Rerank only pre-authorized technical chunks in a bounded top-N pool.

    No source discovery, authorization bypass, or changes to historical cases.
    Fail closed to the existing order if scoring fails.
    """
    if not candidates:
        return candidates[:top_k]
    if not is_reranker_enabled():
        if getattr(settings, "SERVY_RERANKER_REQUIRED", False):
            raise RuntimeError("Required V5 reranker is disabled")
        return candidates[:top_k]
    from core.services.request_deadline import StageBusy, StageTimeout, run_bounded, stage_budget
    if stage_budget("v5_reranking") is None:
        model = get_reranker()
        if model is None:
            if getattr(settings, "SERVY_RERANKER_REQUIRED", False):
                raise RuntimeError("Required V5 reranker failed to load")
            return candidates[:top_k]
    else:
        # Under a request deadline, loading happens inside the bounded scorer so
        # a cold (unloaded) model cannot block the request for its full load time.
        model = None
    try:
        t0 = time.perf_counter()
        docs = [c for c in candidates if c.source_kind == "knowledge_document"]
        others = [c for c in candidates if c.source_kind != "knowledge_document"]
        pool_size = int(getattr(settings, "SERVY_RERANKER_POOL_SIZE", 5))
        # Pool 5 matches the measured BGE+V5 development configuration;
        # larger pools are research-only until they pass paired evaluations.
        if not 1 <= pool_size <= 10:
            raise ValueError("SERVY_RERANKER_POOL_SIZE must be 1..10")
        pool = docs[:pool_size]
        # Validate the full prediction BEFORE mutating any result objects.
        if pool:
            pairs = [
                [query, build_contextual_candidate_text(c, asset_context) if context_aware else c.text]
                for c in pool
            ]
            # Explicit identity activation matches the offline research scorer.
            # For one-label CrossEncoders, predict() commonly defaults to
            # sigmoid; applying calibrated_sigmoid_score again would distort
            # the diagnostic score scale even though rank order is monotone.
            try:
                raw = run_bounded("v5_reranking", _predict, model, pairs)
            except (StageTimeout, StageBusy) as exc:
                # Keep the authorized pre-rerank order; report V5 as not applied.
                from core.services.rag_execution_log import emit_execution_event
                emit_execution_event("v5_reranking_outcome", "Passage reranker deadline", "skipped",
                                     error_type=type(exc).__name__)
                logger.info("V5 reranking not applied (%s); keeping retrieval order", type(exc).__name__)
                return candidates[:top_k]
            scores = [float(x) for x in raw]
            if len(scores) != len(pool) or not all(math.isfinite(x) for x in scores):
                raise ValueError("Invalid or non-finite CrossEncoder scores")
            for candidate, logit in zip(pool, scores):
                candidate.reranker_score = calibrated_sigmoid_score(logit)
                candidate.reranker_model = _applied_model_label()
                candidate.rerank_logit = logit
                candidate.retrieval_score = candidate.score
            pool.sort(key=lambda c: c.rerank_logit, reverse=True)
        logger.info(
            "CrossEncoder reranked %d authorized chunks in %.1f ms",
            len(pool), (time.perf_counter() - t0) * 1000,
        )
        return (pool + docs[pool_size:] + others)[:top_k]
    except Exception as exc:
        if getattr(settings, "SERVY_RERANKER_REQUIRED", False):
            raise RuntimeError("Required V5 reranker inference failed") from exc
        logger.warning("CrossEncoder scoring failed; preserving original ranking: %s", exc)
        return candidates[:top_k]
