"""Request-scoped, privacy-safe execution events for Servy RAG.

The event sink is optional. Normal JSON/API calls and offline evaluations keep
their existing behavior; streaming callers bind a sink for the lifetime of one
request. Events contain stage labels, timestamps, durations and allow-listed
counts only—never questions, prompts, source text, credentials or record IDs.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import time
import uuid
from typing import Callable, Iterator


_EventSink = Callable[[dict], object]
_CURRENT_SINK: ContextVar[_EventSink | None] = ContextVar(
    "servy_rag_execution_event_sink", default=None
)

_ALLOWED_METRICS = frozenset({
    "candidate_count",
    "retrieved_count",
    "supported_count",
    "past_case_count",
    "citation_count",
    "attempt_count",
    "confidential_scope",
    "retrieval_backend",
    "reranker_applied",
    "reranker_model",
    "answer_engine",
    "model_provider",
})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_metrics(metrics: dict | None) -> dict:
    if not isinstance(metrics, dict):
        return {}
    safe = {}
    for key, value in metrics.items():
        if key not in _ALLOWED_METRICS:
            continue
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            safe[key] = max(0, min(value, 1_000_000))
        elif isinstance(value, str) and len(value) <= 80:
            safe[key] = value
    return safe


@contextmanager
def bind_execution_log(sink: _EventSink) -> Iterator[None]:
    """Bind an event callback to the current synchronous execution context."""
    if not callable(sink):
        raise TypeError("Execution log sink must be callable.")
    token = _CURRENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_SINK.reset(token)


def emit_execution_event(
    stage: str,
    label: str,
    status: str = "info",
    *,
    event_id: str | None = None,
    duration_ms: float | None = None,
    metrics: dict | None = None,
    error_type: str | None = None,
) -> None:
    """Publish one safe event. Sink failures never interrupt RAG execution."""
    sink = _CURRENT_SINK.get()
    if sink is None:
        return
    if status not in {"started", "completed", "failed", "skipped", "info"}:
        status = "info"
    event = {
        "type": "stage",
        "event_id": event_id or uuid.uuid4().hex[:16],
        "stage": str(stage)[:80],
        "label": str(label)[:140],
        "status": status,
        "timestamp": _timestamp(),
        "metrics": _safe_metrics(metrics),
    }
    if duration_ms is not None:
        event["duration_ms"] = max(0, min(round(float(duration_ms), 1), 3_600_000))
    if error_type:
        event["error_type"] = str(error_type)[:80]
    try:
        sink(event)
    except Exception:
        # Observability must never change retrieval, authorization or answers.
        return


@contextmanager
def execution_stage(stage: str, label: str) -> Iterator[None]:
    """Emit paired started/completed events with measured elapsed time."""
    sink = _CURRENT_SINK.get()
    if sink is None:
        yield
        return
    event_id = uuid.uuid4().hex[:16]
    started = time.perf_counter()
    emit_execution_event(stage, label, "started", event_id=event_id)
    try:
        yield
    except Exception as exc:
        emit_execution_event(
            stage, label, "failed", event_id=event_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=type(exc).__name__,
        )
        raise
    else:
        emit_execution_event(
            stage, label, "completed", event_id=event_id,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
