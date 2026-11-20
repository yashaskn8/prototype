"""One monotonic end-to-end deadline per customer RAG request.

The clock starts when the backend accepts the turn. Every bounded stage asks for
``stage_budget(name)``: the smaller of its configured cap and the time left
after a reserve for the mandatory work that always runs (authorization,
evidence validation, persistence, serialization). Stages never restart the
clock and never retry beyond what remains.

``run_bounded`` executes CPU-bound model work (vector search, V5 scoring) in a
small dedicated pool and stops waiting when the budget is spent. Python cannot
interrupt a running PyTorch call, so the abandoned call finishes in the
background; its result is discarded and never applied to the response, and
the pool size bounds how much such work can exist at once.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import BoundedSemaphore
from typing import Callable, Dict, Iterator, Optional

from django.conf import settings

_CURRENT: ContextVar[Optional["RequestDeadline"]] = ContextVar("servy_request_deadline", default=None)

# Measured warm durations on the development laptop (i7-10870H CPU for BGE/V5,
# RTX 3050 Ti for Qwen): vector search ~0.05 s, V5 (5 pairs) 1.35-1.6 s,
# Qwen (6 claims) ~2.7-3.3 s. Long passages push V5 to ~2.05 s, hence a 2.6 s cap.
# Caps leave headroom without letting one stage take all.
DEFAULT_STAGE_CAPS = {
    "dense_retrieval": 1.0,
    "v5_reranking": 2.6,
    "customer_qwen_generation": 3.5,
}


class StageTimeout(Exception):
    """A bounded stage did not finish within its budget."""


class StageBusy(Exception):
    """No capacity to start bounded work (earlier abandoned work still running)."""


class RequestDeadline:
    def __init__(self, total_seconds: float, reserve_seconds: float, caps: Dict[str, float]):
        self.started = time.monotonic()
        self.total = float(total_seconds)
        self.reserve = float(reserve_seconds)
        self.caps = dict(caps)
        self.timeouts: Dict[str, float] = {}

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.total - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def stage_budget(self, stage: str) -> float:
        """Seconds this stage may use now (0 when it must not start)."""
        available = self.remaining() - self.reserve
        cap = self.caps.get(stage)
        return max(0.0, min(available, cap) if cap is not None else available)

    def record_timeout(self, stage: str, budget: float) -> None:
        self.timeouts[stage] = round(budget, 3)


def current_deadline() -> Optional[RequestDeadline]:
    return _CURRENT.get()


def stage_budget(stage: str) -> Optional[float]:
    """Budget for a stage, or None when no request deadline is active (offline tools, tests)."""
    deadline = _CURRENT.get()
    return None if deadline is None else deadline.stage_budget(stage)


@contextmanager
def request_deadline(total_seconds: Optional[float] = None) -> Iterator[RequestDeadline]:
    total = float(total_seconds if total_seconds is not None
                  else getattr(settings, "SERVY_RAG_DEADLINE_SECONDS", 5.0))
    caps = dict(DEFAULT_STAGE_CAPS)
    caps.update(getattr(settings, "SERVY_RAG_STAGE_CAPS", None) or {})
    deadline = RequestDeadline(total, float(getattr(settings, "SERVY_RAG_DEADLINE_RESERVE_SECONDS", 0.4)), caps)
    token = _CURRENT.set(deadline)
    try:
        yield deadline
    finally:
        _CURRENT.reset(token)


_POOL_SIZE = 2
_POOL = ThreadPoolExecutor(max_workers=_POOL_SIZE, thread_name_prefix="servy-bounded")
_SLOTS = BoundedSemaphore(_POOL_SIZE)


def run_bounded(stage: str, fn: Callable, *args, **kwargs):
    """Run ``fn`` within the stage budget; raise StageTimeout / StageBusy instead of waiting longer.

    Without an active deadline the call runs inline, unchanged.
    """
    budget = stage_budget(stage)
    if budget is None:
        return fn(*args, **kwargs)
    deadline = current_deadline()
    if budget <= 0:
        deadline.record_timeout(stage, 0.0)
        raise StageTimeout(f"{stage}: no budget left")
    if not _SLOTS.acquire(blocking=False):
        raise StageBusy(f"{stage}: earlier bounded work is still running")
    context = copy_context()

    def work():
        try:
            return context.run(fn, *args, **kwargs)
        finally:
            _SLOTS.release()

    try:
        future = _POOL.submit(work)
    except Exception:
        _SLOTS.release()
        raise
    try:
        return future.result(timeout=budget)
    except FutureTimeout:
        deadline.record_timeout(stage, budget)
        raise StageTimeout(f"{stage}: exceeded {budget:.2f}s") from None


def wait_until_idle(timeout: float) -> bool:
    """Wait for abandoned bounded work to finish (tests and orderly shutdown)."""
    end = time.monotonic() + timeout
    acquired = 0
    try:
        while acquired < _POOL_SIZE:
            if not _SLOTS.acquire(timeout=max(0.0, end - time.monotonic())):
                return False
            acquired += 1
        return True
    finally:
        for _ in range(acquired):
            _SLOTS.release()
