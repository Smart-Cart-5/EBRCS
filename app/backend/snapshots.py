"""Snapshot buffer for event-time product classification."""

from __future__ import annotations

from typing import Any

import numpy as np


class SnapshotBuffer:
    def __init__(self, max_frames: int = 8):
        self.max_frames = max(1, int(max_frames))
        self._entries: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._entries = []

    def add(self, frame: np.ndarray, box: list[float] | None) -> None:
        if box is None:
            return
        h, w = frame.shape[:2]
        x1 = int(max(0.0, min(1.0, box[0])) * w)
        y1 = int(max(0.0, min(1.0, box[1])) * h)
        x2 = int(max(0.0, min(1.0, box[2])) * w)
        y2 = int(max(0.0, min(1.0, box[3])) * h)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return
        area = (x2 - x1) * (y2 - y1)
        self._entries.append({"area": float(area), "crop": crop.copy()})

        if len(self._entries) > self.max_frames:
            self._entries = sorted(self._entries, key=lambda x: x["area"], reverse=True)[: self.max_frames]

    def best_crops(self, limit: int | None = None) -> list[np.ndarray]:
        if not self._entries:
            return []
        n = self.max_frames if limit is None else max(1, int(limit))
        entries = sorted(self._entries, key=lambda x: x["area"], reverse=True)[:n]
        return [entry["crop"] for entry in entries]
