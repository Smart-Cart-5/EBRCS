from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def clamp_virtual_scale(scale: float) -> float:
    return float(max(0.30, min(1.0, float(scale))))


def apply_virtual_distance(frame: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = frame.shape[:2]
    s = clamp_virtual_scale(scale)
    if s >= 0.9999:
        return frame, {
            "scale": float(s),
            "applied": False,
            "offset_x": 0,
            "offset_y": 0,
            "inner_w": int(w),
            "inner_h": int(h),
            "original_w": int(w),
            "original_h": int(h),
        }

    inner_w = max(1, int(round(w * s)))
    inner_h = max(1, int(round(h * s)))
    resized = cv2.resize(frame, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(frame)
    off_x = max(0, (w - inner_w) // 2)
    off_y = max(0, (h - inner_h) // 2)
    canvas[off_y : off_y + inner_h, off_x : off_x + inner_w] = resized
    return canvas, {
        "scale": float(s),
        "applied": True,
        "offset_x": int(off_x),
        "offset_y": int(off_y),
        "inner_w": int(inner_w),
        "inner_h": int(inner_h),
        "original_w": int(w),
        "original_h": int(h),
    }

