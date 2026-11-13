"""SSE bridge for request-scoped RAG execution logs.

The synchronous RAG pipeline runs in a bounded worker thread. Authenticated
views perform role, tenant, throttle and resource checks before calling this
bridge. The stream carries safe stage telemetry and one final response only.
"""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Event, Thread
import json
import logging

from django.db import close_old_connections
from django.http import JsonResponse, StreamingHttpResponse

from .rag_execution_log import bind_execution_log

logger = logging.getLogger(__name__)
_QUEUE_SIZE = 128
_HEARTBEAT_SECONDS = 12
_MAX_ACTIVE_STREAMS = 4
_STREAM_SLOTS = BoundedSemaphore(_MAX_ACTIVE_STREAMS)


def _sse_frame(event_name: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"


def rag_execution_stream(work):
    """Create a private event stream for one already-authorized RAG request."""
    if not callable(work):
        raise TypeError("RAG stream work must be callable.")

    if not _STREAM_SLOTS.acquire(blocking=False):
        return JsonResponse(
            {"detail": "RAG streaming is at capacity. Please retry shortly."},
            status=503,
        )

    events: Queue = Queue(maxsize=_QUEUE_SIZE)
    closed = Event()

    def publish(event: dict, *, terminal: bool = False) -> None:
        if closed.is_set() or not isinstance(event, dict):
            return
        if terminal:
            # A final result/error must never be evicted by the queue sentinel
            # or by later progress events. Drop the oldest display-only event
            # when a slow consumer has filled the bounded queue.
            while not closed.is_set():
                try:
                    events.put_nowait(event)
                    return
                except Full:
                    try:
                        events.get_nowait()
                    except Empty:
                        continue
            return

        try:
            events.put_nowait(event)
        except Full:
            try:
                events.get_nowait()
            except Empty:
                pass
            try:
                events.put_nowait(event)
            except Full:
                return

    def run_work() -> None:
        try:
            close_old_connections()
            with bind_execution_log(publish):
                result = work()
            publish({"type": "result", "data": result}, terminal=True)
        except Exception as exc:
            logger.exception("Streaming RAG request failed (%s)", type(exc).__name__)
            publish({
                "type": "error",
                "data": {"detail": "RAG request failed. Please retry."},
            }, terminal=True)
        finally:
            try:
                close_old_connections()
            finally:
                _STREAM_SLOTS.release()

    worker = Thread(target=run_work, name="servy-rag-stream", daemon=True)
    try:
        worker.start()
    except Exception:
        _STREAM_SLOTS.release()
        logger.exception("Could not start the RAG stream worker")
        return JsonResponse(
            {"detail": "RAG streaming could not be started. Please retry."},
            status=503,
        )

    def generate():
        yield ": servy-rag-stream\n\n"
        try:
            while True:
                try:
                    item = events.get(timeout=_HEARTBEAT_SECONDS)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue
                if not isinstance(item, dict):
                    continue
                event_type = item.get("type")
                if event_type == "stage":
                    yield _sse_frame("log", item)
                elif event_type in {"result", "error"}:
                    yield _sse_frame(event_type, item.get("data") or {})
                    break
        finally:
            # This stops further log queuing. A synchronous RAG operation
            # already in progress is allowed to finish safely in its worker.
            closed.set()

    response = StreamingHttpResponse(
        generate(), content_type="text/event-stream; charset=utf-8"
    )
    response["Cache-Control"] = "no-cache, no-transform, private"
    response["X-Accel-Buffering"] = "no"
    response["X-Content-Type-Options"] = "nosniff"
    response["Referrer-Policy"] = "no-referrer"
    return response
