"""SSE bridge for request-scoped RAG execution logs.

The RAG pipeline remains synchronous. This bridge runs that request in a
bounded worker thread and forwards real execution events to an authenticated
Django StreamingHttpResponse without persisting customer or technician logs.
"""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread
import json
import logging

from django.db import close_old_connections
from django.http import StreamingHttpResponse

from .rag_execution_log import bind_execution_log

logger = logging.getLogger(__name__)
_QUEUE_SIZE = 128
_HEARTBEAT_SECONDS = 12


def _sse_frame(event_name: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"


def rag_execution_stream(work):
    """Return an SSE response for a callable that returns the final API payload.

    Authentication, role checks, throttling and resource scoping must happen
    before this function is called. The callable runs in a fresh DB connection
    and inherits only the validated objects/identifiers selected by the view.
    """
    if not callable(work):
        raise TypeError("RAG stream work must be callable.")

    events: Queue = Queue(maxsize=_QUEUE_SIZE)
    closed = Event()
    finished = object()

    def enqueue(event: dict) -> None:
        if closed.is_set() or not isinstance(event, dict):
            return
        try:
            events.put_nowait(event)
        except Full:
            # Preserve forward progress if a client is slow. Discard only the
            # oldest display event; the underlying RAG request is unaffected.
            try:
                events.get_nowait()
            except Empty:
                pass
            try:
                events.put_nowait(event)
            except Full:
                return

    def run_work() -> None:
        close_old_connections()
        try:
            with bind_execution_log(enqueue):
                result = work()
            enqueue({"type": "result", "data": result})
        except Exception as exc:
            logger.exception("Streaming RAG request failed (%s)", type(exc).__name__)
            enqueue({
                "type": "error",
                "data": {"detail": "RAG request failed. Please retry."},
            })
        finally:
            close_old_connections()
            if not closed.is_set():
                try:
                    events.put_nowait(finished)
                except Full:
                    try:
                        events.get_nowait()
                    except Empty:
                        pass
                    try:
                        events.put_nowait(finished)
                    except Full:
                        pass

    Thread(target=run_work, name="servy-rag-stream", daemon=True).start()

    def generate():
        yield ": servy-rag-stream\n\n"
        try:
            while True:
                try:
                    item = events.get(timeout=_HEARTBEAT_SECONDS)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is finished:
                    break
                if not isinstance(item, dict):
                    continue
                event_type = item.get("type")
                if event_type == "stage":
                    yield _sse_frame("log", item)
                elif event_type in {"result", "error"}:
                    yield _sse_frame(event_type, item.get("data") or {})
                    if event_type in {"result", "error"}:
                        break
        finally:
            closed.set()

    response = StreamingHttpResponse(generate(), content_type="text/event-stream; charset=utf-8")
    response["Cache-Control"] = "no-cache, no-transform, private"
    response["X-Accel-Buffering"] = "no"
    response["X-Content-Type-Options"] = "nosniff"
    response["Referrer-Policy"] = "no-referrer"
    return response
