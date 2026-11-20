"""Bounded, privacy-safe per-request RAG metrics (one JSON line per finished request).

Enabled only when SERVY_RAG_METRICS_LOG names a file (the launcher points it at
the private demo folder). Rotation keeps at most 3 x 2 MB. Each line holds
identifiers and counts only: request id, asset id, outcome and model statuses,
stage durations, citation count and process memory. Never question text,
answers, manual passages, prompts, access codes or customer details.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from django.conf import settings

_STAGES = ("dense_retrieval", "sparse_retrieval", "rank_fusion", "v5_reranking", "retrieval_pipeline",
           "customer_qwen_generation", "customer_evidence_validation", "customer_rag_execution")
_logger = logging.getLogger("servy.rag.metrics")
_logger.propagate = False
_lock = threading.Lock()
_configured_path = None


def _handler_for(path: str):
    global _configured_path
    with _lock:
        if _configured_path == path:
            return True
        for handler in list(_logger.handlers):
            _logger.removeHandler(handler)
            handler.close()
        handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _configured_path = path
        return True


def record(contract: dict, events: list, asset_id=None) -> None:
    path = str(getattr(settings, "SERVY_RAG_METRICS_LOG", "") or "").strip()
    if not path:
        return
    try:
        import psutil
        outcome = contract.get("outcome") or {}
        diagnostics = contract.get("diagnostics") or {}
        stages = {e["stage"]: e.get("duration_ms") for e in events
                  if e.get("stage") in _STAGES and e.get("status") == "completed" and e.get("duration_ms") is not None}
        line = {
            "request_id": uuid.uuid4().hex,
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "asset_id": asset_id,
            "status": outcome.get("status"),
            "response_mode": outcome.get("response_mode"),
            "response_type": contract.get("response_type"),
            "total_ms": outcome.get("total_duration_ms"),
            "deadline_exceeded": outcome.get("deadline_exceeded"),
            "stage_timeouts": outcome.get("stage_timeouts"),
            "retrieval_status": outcome.get("retrieval_status"),
            "retrieved": (diagnostics.get("retrieval") or {}).get("retrieved"),
            "reranker_status": outcome.get("reranker_status"),
            "generation_status": outcome.get("generation_status"),
            "generation_applied": outcome.get("generation_applied"),
            "qwen_accepted": (diagnostics.get("qwen") or {}).get("accepted_claims"),
            "qwen_eligible": (diagnostics.get("qwen") or {}).get("eligible_claims"),
            "citations": len((contract.get("presentation") or {}).get("citations") or []),
            "fallback_reason": outcome.get("fallback_reason"),
            "stage_ms": stages,
            "rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        }
        if _handler_for(path):
            _logger.info(json.dumps(line, separators=(",", ":")))
    except Exception:
        # Metrics must never affect a customer answer.
        return
