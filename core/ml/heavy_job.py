"""At most one memory-heavy model job per machine (laptop with ~4 GB to spare).

Loading the serving models takes ~2 GB of RAM per process (V5 alone ~1.5 GB).
The serving gateway, the V5 ONNX export and the strict serving check each load
them, so running two at once can push Windows into heavy paging. A lock file
records the owning PID; a lock left by a process that no longer exists is
removed automatically. A second job is refused immediately (never waits).
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(os.getenv("SERVY_HEAVY_JOB_LOCK", str(Path(tempfile.gettempdir()) / "servy_heavy_model_job.lock")))


class HeavyJobBusy(RuntimeError):
    pass


def _owner():
    try:
        pid_text, _, name = LOCK.read_text(encoding="utf-8").partition(" ")
        return int(pid_text), name.strip()
    except (OSError, ValueError):
        return None, ""


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True


def acquire(name: str) -> None:
    """Take the machine-wide heavy-job lock or raise HeavyJobBusy."""
    if os.getenv("SERVY_HEAVY_JOB_LOCK_DISABLED") == "1":
        return
    for _ in range(2):
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid, owner = _owner()
            if pid == os.getpid():
                return
            if pid is not None and _alive(pid):
                raise HeavyJobBusy(f"Another model job is running ({owner}, PID {pid}); wait for it or stop it first.")
            try:
                LOCK.unlink()  # stale lock from a process that no longer exists
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {name}")
        return
    raise HeavyJobBusy("Could not take the model job lock.")


def release() -> None:
    pid, _ = _owner()
    if pid == os.getpid():
        try:
            LOCK.unlink()
        except OSError:
            pass


@contextmanager
def heavy_job(name: str):
    acquire(name)
    try:
        yield
    finally:
        release()
