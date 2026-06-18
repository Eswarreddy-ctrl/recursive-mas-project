"""
metrics/timer.py — Wall-clock timer for measuring workflow end-to-end latency.

Usage:
    t = Timer()
    t.start()
    elapsed = t.stop()          # float seconds

    with Timer() as t:          # context manager
        ...
    elapsed = t.elapsed()

    t = Timer(mock_elapsed=2.5) # deterministic for tests
"""

from __future__ import annotations
import time
from typing import Optional


class Timer:
    def __init__(self, mock_elapsed: Optional[float] = None):
        self._mock_elapsed = mock_elapsed
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> "Timer":
        if self._start_time is not None and self._end_time is None:
            raise RuntimeError("Timer is already running. Call reset() before starting again.")
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer has not started. Call start() first.")
        if self._mock_elapsed is not None:
            self._end_time = self._start_time + self._mock_elapsed
        else:
            self._end_time = time.perf_counter()
        return self._end_time - self._start_time

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        if self._end_time is not None:
            return self._end_time - self._start_time
        if self._mock_elapsed is not None:
            return self._mock_elapsed
        return time.perf_counter() - self._start_time

    def reset(self) -> "Timer":
        self._start_time = None
        self._end_time = None
        return self

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()