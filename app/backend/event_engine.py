"""Hand-driven ADD event engine (MVP, single candidate)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("backend.event_engine")


@dataclass
class EventUpdate:
    state: str
    status: str
    add_confirmed: bool = False
    remove_confirmed: bool = False
    track_box: list[float] | None = None
    track_id: str | int | None = None
    prev_inside: bool | None = None
    curr_inside: bool | None = None
    stable_frames: int = 0
    event_payload: dict[str, Any] | None = None


@dataclass
class TrackState:
    confirmed_inside: bool | None = None
    candidate_inside: bool | None = None
    candidate_frames: int = 0
    stable_frames: int = 0
    last_transition_ts: float = 0.0
    last_event_type: str | None = None
    last_box: list[float] | None = None
    last_confidence: float = 0.0
    last_seen_frame: int = -1


class AddEventEngine:
    IDLE = "IDLE"
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"

    def __init__(
        self,
        transition_confirm_frames: int = 3,
        event_cooldown_sec: float = 0.75,
        inside_mode: str = "center",
        inside_iou_threshold: float = 0.15,
        inside_overlap_threshold: float = 0.2,
        track_ttl_frames: int = 45,
        allow_untracked_index_fallback: bool = False,
        roi_hysteresis_inset_ratio: float = 0.05,
        roi_hysteresis_outset_ratio: float = 0.05,
    ):
        self.transition_confirm_frames = max(1, int(transition_confirm_frames))
        self.event_cooldown_sec = max(0.0, float(event_cooldown_sec))
        self.inside_mode = str(inside_mode or "center").strip().lower()
        self.inside_iou_threshold = max(0.0, float(inside_iou_threshold))
        self.inside_overlap_threshold = max(0.0, float(inside_overlap_threshold))
        self.track_ttl_frames = max(1, int(track_ttl_frames))
        self.allow_untracked_index_fallback = bool(allow_untracked_index_fallback)
        self.roi_hysteresis_inset_ratio = max(0.0, float(roi_hysteresis_inset_ratio))
        self.roi_hysteresis_outset_ratio = max(0.0, float(roi_hysteresis_outset_ratio))
        self.track_states: dict[str | int, TrackState] = {}

    @staticmethod
    def _center(box: list[float]) -> tuple[float, float]:
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

    @staticmethod
    def _box_to_pixel(box: list[float], frame_shape: tuple[int, ...]) -> list[float]:
        h, w = frame_shape[:2]
        return [
            float(box[0]) * float(w),
            float(box[1]) * float(h),
            float(box[2]) * float(w),
            float(box[3]) * float(h),
        ]

    @staticmethod
    def _safe_int(v: Any) -> int | None:
        try:
            return int(v)
        except Exception:
            return None

    def _extract_track_id(self, product: dict[str, Any], idx: int) -> str | int | None:
        for key in ("track_id", "tracker_id", "id"):
            if key in product:
                value = product.get(key)
                as_int = self._safe_int(value)
                if as_int is not None:
                    return as_int
                if value is not None:
                    s = str(value).strip()
                    if s:
                        return s
        if self.allow_untracked_index_fallback:
            return f"det_{idx}"
        return None

    @staticmethod
    def _roi_overlap_ratio(box: list[float], roi_poly: np.ndarray | None) -> float:
        if roi_poly is None or len(roi_poly) < 3:
            return 1.0
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        box_area = bw * bh
        if box_area <= 1e-9:
            return 0.0
        rect = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        try:
            inter_area, _ = cv2.intersectConvexConvex(roi_poly.astype(np.float32), rect)
        except Exception:
            return 0.0
        inter = max(0.0, float(inter_area))
        return inter / max(1e-9, box_area)

    @staticmethod
    def _roi_iou(box: list[float], roi_poly: np.ndarray | None) -> float:
        if roi_poly is None or len(roi_poly) < 3:
            return 1.0
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        box_area = bw * bh
        roi_area = abs(float(cv2.contourArea(roi_poly.astype(np.float32))))
        if box_area <= 1e-9 or roi_area <= 1e-9:
            return 0.0
        rect = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        try:
            inter_area, _ = cv2.intersectConvexConvex(roi_poly.astype(np.float32), rect)
        except Exception:
            return 0.0
        inter = max(0.0, float(inter_area))
        union = max(1e-9, box_area + roi_area - inter)
        return inter / union

    def _inside_roi(
        self,
        box: list[float],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
        *,
        mode: str = "normal",
    ) -> bool:
        if roi_poly is None:
            return True
        box_px = self._box_to_pixel(box, frame_shape)
        poly = roi_poly.astype(np.float32)
        if self.inside_mode == "center":
            if mode in {"in", "out"} and len(poly) >= 3:
                center = np.mean(poly, axis=0)
                if mode == "in":
                    ratio = max(0.0, 1.0 - self.roi_hysteresis_inset_ratio)
                else:
                    ratio = 1.0 + self.roi_hysteresis_outset_ratio
                poly = (poly - center) * ratio + center
            cx = (box_px[0] + box_px[2]) * 0.5
            cy = (box_px[1] + box_px[3]) * 0.5
            return bool(cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0)

        if self.inside_mode == "iou":
            return bool(self._roi_iou(box_px, poly) >= self.inside_iou_threshold)

        if self.inside_mode == "overlap":
            return bool(self._roi_overlap_ratio(box_px, poly) >= self.inside_overlap_threshold)

        cx = (box_px[0] + box_px[2]) * 0.5
        cy = (box_px[1] + box_px[3]) * 0.5
        return bool(cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0)

    @staticmethod
    def _roi_bounds(roi_poly: np.ndarray | None) -> dict[str, float] | None:
        if roi_poly is None or len(roi_poly) < 3:
            return None
        xs = roi_poly[:, 0]
        ys = roi_poly[:, 1]
        return {
            "x_min": float(np.min(xs)),
            "y_min": float(np.min(ys)),
            "x_max": float(np.max(xs)),
            "y_max": float(np.max(ys)),
        }

    def _cleanup_tracks(self, frame_id: int) -> None:
        stale_ids: list[str | int] = []
        for track_id, trk in self.track_states.items():
            if frame_id - int(trk.last_seen_frame) > self.track_ttl_frames:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            self.track_states.pop(track_id, None)

    def update(
        self,
        *,
        best_pair: dict[str, Any] | None,
        products: list[dict[str, Any]],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
        frame_id: int | None = None,
        session_id: str | None = None,
    ) -> EventUpdate:
        del best_pair  # Crossing logic is track transition based.
        frame_id_i = int(frame_id if frame_id is not None else 0)
        self._cleanup_tracks(frame_id_i)

        focus_box: list[float] | None = None
        focus_track_id: str | int | None = None
        focus_curr_inside: bool | None = None
        focus_stable_frames = 0
        focus_conf = -1.0
        pending_event: EventUpdate | None = None
        skipped_untracked = 0

        for idx, product in enumerate(products):
            box = product.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            track_id = self._extract_track_id(product, idx)
            if track_id is None:
                skipped_untracked += 1
                continue
            confidence = float(product.get("confidence", 0.0))
            trk = self.track_states.setdefault(track_id, TrackState())
            trk.last_seen_frame = frame_id_i
            trk.last_box = list(box)
            trk.last_confidence = confidence

            mode = "normal"
            if trk.confirmed_inside is True:
                mode = "out"
            elif trk.confirmed_inside is False:
                mode = "in"
            observed_inside = bool(self._inside_roi(box, roi_poly, frame_shape, mode=mode))

            if trk.confirmed_inside is None:
                trk.confirmed_inside = observed_inside
                trk.stable_frames = 1
                trk.candidate_inside = None
                trk.candidate_frames = 0
            elif observed_inside == trk.confirmed_inside:
                trk.stable_frames += 1
                trk.candidate_inside = None
                trk.candidate_frames = 0
            else:
                if trk.candidate_inside != observed_inside:
                    trk.candidate_inside = observed_inside
                    trk.candidate_frames = 1
                else:
                    trk.candidate_frames += 1

                if trk.candidate_frames >= self.transition_confirm_frames:
                    prev_inside = bool(trk.confirmed_inside)
                    curr_inside = bool(observed_inside)
                    trk.confirmed_inside = curr_inside
                    trk.stable_frames = 1
                    trk.candidate_inside = None
                    trk.candidate_frames = 0

                    event_type = "ADD" if (not prev_inside and curr_inside) else "REMOVE"
                    now_ts = time.time()
                    cooldown_ok = (now_ts - float(trk.last_transition_ts)) >= self.event_cooldown_sec
                    trk.last_transition_ts = now_ts

                    if cooldown_ok:
                        trk.last_event_type = event_type
                        payload = {
                            "session_id": session_id,
                            "frame_id": frame_id_i,
                            "track_id": track_id,
                            "event_type": event_type,
                            "prev_inside": prev_inside,
                            "curr_inside": curr_inside,
                            "roi_id": "main",
                            "roi_bounds": self._roi_bounds(roi_poly),
                            "confidence": confidence,
                            "inside_mode": self.inside_mode,
                            "stable_frames": self.transition_confirm_frames,
                        }
                        logger.info(
                            "EVENT_CROSS: session=%s frame_id=%s track=%s prev=%s curr=%s stable_frames=%d -> %s",
                            session_id,
                            frame_id_i,
                            track_id,
                            "in" if prev_inside else "out",
                            "in" if curr_inside else "out",
                            self.transition_confirm_frames,
                            event_type,
                        )
                        pending_event = EventUpdate(
                            state=self.INSIDE if curr_inside else self.OUTSIDE,
                            status=f"{event_type} 확정",
                            add_confirmed=(event_type == "ADD"),
                            remove_confirmed=(event_type == "REMOVE"),
                            track_box=list(box),
                            track_id=track_id,
                            prev_inside=prev_inside,
                            curr_inside=curr_inside,
                            stable_frames=self.transition_confirm_frames,
                            event_payload=payload,
                        )

            if confidence > focus_conf:
                focus_conf = confidence
                focus_box = list(box)
                focus_track_id = track_id
                focus_curr_inside = bool(trk.confirmed_inside) if trk.confirmed_inside is not None else None
                focus_stable_frames = int(trk.stable_frames)

        if pending_event is not None:
            return pending_event

        if skipped_untracked > 0 and products:
            logger.debug(
                "EVENT_CROSS skip: session=%s frame_id=%s reason=untracked_products skipped=%d fallback=%s",
                session_id,
                frame_id_i,
                skipped_untracked,
                self.allow_untracked_index_fallback,
            )

        if focus_box is None:
            return EventUpdate(state=self.IDLE, status="IDLE")

        status = "ROI 내부" if focus_curr_inside else "ROI 외부"
        return EventUpdate(
            state=self.INSIDE if focus_curr_inside else self.OUTSIDE,
            status=status,
            track_box=focus_box,
            track_id=focus_track_id,
            prev_inside=focus_curr_inside,
            curr_inside=focus_curr_inside,
            stable_frames=focus_stable_frames,
            event_payload=None,
        )
