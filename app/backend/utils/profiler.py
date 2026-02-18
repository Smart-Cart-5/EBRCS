"""Lightweight rolling profiler helpers with optional no-op mode."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import math
import time
from typing import Iterator


class RollingStats:
    def __init__(self, maxlen: int = 200):
        self.values: deque[float] = deque(maxlen=maxlen)

    def add(self, value_ms: float) -> None:
        self.values.append(float(value_ms))

    @property
    def avg_ms(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def p95_ms(self) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
        return float(ordered[idx])


class _NoopMeasure:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FrameProfiler:
    def __init__(self, collector: "ProfileCollector"):
        self.collector = collector
        self.breakdown_ms: dict[str, float] = {}
        self._noop = _NoopMeasure()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.collector.enable:
            with self._noop:
                yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.breakdown_ms[name] = self.breakdown_ms.get(name, 0.0) + elapsed_ms

    def add_ms(self, name: str, value_ms: float) -> None:
        if not self.collector.enable:
            return
        self.breakdown_ms[name] = self.breakdown_ms.get(name, 0.0) + float(value_ms)

    def finish(self) -> None:
        self.collector.record(self.breakdown_ms)


class ProfileCollector:
    def __init__(self, *, kind: str, enable: bool, every_n_frames: int, logger, maxlen: int = 200):
        self.kind = kind
        self.enable = bool(enable)
        self.every_n_frames = max(1, int(every_n_frames))
        self.logger = logger
        self.frame_count = 0
        self.total_stats = RollingStats(maxlen=maxlen)
        self.breakdown_stats: dict[str, RollingStats] = {}

    def start_frame(self) -> FrameProfiler:
        return FrameProfiler(self)

    def record(self, breakdown_ms: dict[str, float]) -> None:
        if not self.enable:
            return
        self.frame_count += 1
        total_ms = sum(breakdown_ms.values())
        self.total_stats.add(total_ms)

        for name, ms in breakdown_ms.items():
            if name not in self.breakdown_stats:
                self.breakdown_stats[name] = RollingStats(maxlen=self.total_stats.values.maxlen or 200)
            self.breakdown_stats[name].add(ms)

        if self.frame_count % self.every_n_frames == 0:
            breakdown_parts = []
            for name in sorted(self.breakdown_stats.keys()):
                breakdown_parts.append(f"{name}:{self.breakdown_stats[name].avg_ms:.2f}")
            self.logger.info(
                "[PROF] kind=%s frames=%d avg_ms=%.2f p95_ms=%.2f breakdown=%s",
                self.kind,
                self.frame_count,
                self.total_stats.avg_ms,
                self.total_stats.p95_ms,
                ",".join(breakdown_parts),
            )
