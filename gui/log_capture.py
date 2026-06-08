"""
Prompt (2026-06): GUI control terminal — capture engine stdout/stderr into a thread-safe queue
for live terminal display without modifying core pipeline logging.
"""

from __future__ import annotations

import sys
import threading
from queue import Queue
from typing import Optional, TextIO


class _TeeStream:
    """Duplicate writes to the original stream and a log queue."""

    def __init__(self, original: TextIO, log_queue: Queue[str]) -> None:
        self._original = original
        self._log_queue = log_queue
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self._lock:
            try:
                self._original.write(data)
            except Exception:
                pass
            self._log_queue.put(data)
        return len(data)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False


class LogCapture:
    """Context manager: tee stdout/stderr into `log_queue` for the engine run duration."""

    def __init__(self, log_queue: Queue[str]) -> None:
        self._log_queue = log_queue
        self._stdout_prev: Optional[TextIO] = None
        self._stderr_prev: Optional[TextIO] = None

    def __enter__(self) -> "LogCapture":
        self._stdout_prev = sys.stdout
        self._stderr_prev = sys.stderr
        sys.stdout = _TeeStream(self._stdout_prev, self._log_queue)  # type: ignore[assignment]
        sys.stderr = _TeeStream(self._stderr_prev, self._log_queue)  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stdout_prev is not None:
            sys.stdout = self._stdout_prev
        if self._stderr_prev is not None:
            sys.stderr = self._stderr_prev
