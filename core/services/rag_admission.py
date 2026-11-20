"""Single active RAG request per server process (single-user prototype).

The models share one CPU (BGE, V5) and one 4 GB GPU (Qwen); a second
simultaneous request only slows both and pushes answers into fallbacks. A
second request is therefore refused immediately with a clear busy status,
never queued. The slot is held for the whole pipeline, including the SSE
worker, and released in ``finally`` so failures and cancellations cannot leak it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

from django.conf import settings
from rest_framework import status
from rest_framework.exceptions import APIException

_SLOT = Lock()
_ADMITTED: ContextVar[bool] = ContextVar("servy_rag_admitted", default=False)
BUSY_MESSAGE = "Servy is still answering your previous question. Please wait for it to finish, then try again."


class RagBusy(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = BUSY_MESSAGE
    default_code = "rag_busy"


def _enabled() -> bool:
    return bool(getattr(settings, "SERVY_RAG_SINGLE_REQUEST", True))


def try_acquire() -> bool:
    """Take the slot without waiting. Always True when single-request mode is off."""
    return _SLOT.acquire(blocking=False) if _enabled() else True


def release() -> None:
    if _enabled() and _SLOT.locked():
        _SLOT.release()


def busy() -> bool:
    return _enabled() and _SLOT.locked()


@contextmanager
def admitted_work():
    """Mark this execution context as already holding the slot (SSE worker)."""
    token = _ADMITTED.set(True)
    try:
        yield
    finally:
        _ADMITTED.reset(token)


@contextmanager
def rag_request():
    """Hold the slot for one pipeline run; reuse it if the caller already holds it."""
    if _ADMITTED.get():
        yield
        return
    if not try_acquire():
        raise RagBusy()
    try:
        with admitted_work():
            yield
    finally:
        release()
