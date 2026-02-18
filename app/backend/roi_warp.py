"""4-point ROI warp helpers."""

from __future__ import annotations

import cv2
import numpy as np


def order_points_tl_tr_br_bl(points: list[list[float]]) -> list[list[float]]:
    if len(points) != 4:
        raise ValueError("warp points must contain exactly 4 points")
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [[float(tl[0]), float(tl[1])], [float(tr[0]), float(tr[1])], [float(br[0]), float(br[1])], [float(bl[0]), float(bl[1])]]


def warp_frame(
    frame: np.ndarray,
    points_norm: list[list[float]],
    warp_size: tuple[int, int],
) -> np.ndarray:
    h, w = frame.shape[:2]
    src = np.array(
        [[p[0] * w, p[1] * h] for p in order_points_tl_tr_br_bl(points_norm)],
        dtype=np.float32,
    )
    out_w, out_h = int(warp_size[0]), int(warp_size[1])
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, m, (out_w, out_h))
