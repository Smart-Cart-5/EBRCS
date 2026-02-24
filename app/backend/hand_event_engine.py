from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _center_distance(a: list[float], b: list[float]) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    dx = acx - bcx
    dy = acy - bcy
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class _Track:
    track_id: int
    box: list[float]
    last_frame: int
    missed: int = 0
    inside_roi: bool = False


@dataclass
class _AssocInfo:
    streak: int = 0
    last_frame: int = -1
    last_score: float = 0.0


@dataclass
class _HandState:
    prev_inside: bool = False
    phase: str = "IDLE"
    created_at_ms: int = 0
    cooldown_until_ms: int = 0
    candidate_object_track_id: int | None = None
    candidate_source: str = "none"
    candidate_score: float = 0.0
    candidate_streak: int = 0
    candidate_box: list[float] | None = None
    candidate_missing_frames: int = 0
    evidence_frames: int = 0
    last_stable_object_track_id: int | None = None
    assoc_streaks: dict[int, _AssocInfo] = field(default_factory=dict)
    last_fail_reason: str = ""


class HandEventEngine:
    def __init__(
        self,
        *,
        association_iou_weight: float = 0.5,
        association_dist_weight: float = 0.5,
        association_max_center_dist: float = 0.40,
        association_min_score: float = 0.03,
        association_min_iou: float = 0.0,
        allow_det_fallback: bool = True,
        det_fallback_min_score: float = 0.01,
        association_top_k: int = 3,
        association_stable_frames: int = 1,
        add_evidence_frames: int = 2,
        remove_evidence_frames: int = 2,
        remove_missing_grace_frames: int = 3,
        candidate_timeout_ms: int = 1200,
        cooldown_ms: int = 700,
        candidate_switch_min_delta: float = 0.08,
        track_iou_match_threshold: float = 0.05,
        track_max_missed_frames: int = 8,
        hand_representative_point: str = "center",
        object_representative_point: str = "bottom_center",
        roi_margin_px: float = 0.0,
        roi_margin_ratio: float = 0.0,
    ):
        self.association_iou_weight = float(association_iou_weight)
        self.association_dist_weight = float(association_dist_weight)
        self.association_max_center_dist = float(max(1e-6, association_max_center_dist))
        self.association_min_score = float(association_min_score)
        self.association_min_iou = float(max(0.0, association_min_iou))
        self.allow_det_fallback = bool(allow_det_fallback)
        self.det_fallback_min_score = float(max(0.0, det_fallback_min_score))
        self.association_top_k = max(1, int(association_top_k))
        self.association_stable_frames = max(1, int(association_stable_frames))
        self.add_evidence_frames = max(1, int(add_evidence_frames))
        self.remove_evidence_frames = max(1, int(remove_evidence_frames))
        self.remove_missing_grace_frames = max(1, int(remove_missing_grace_frames))
        self.candidate_timeout_ms = max(50, int(candidate_timeout_ms))
        self.cooldown_ms = max(0, int(cooldown_ms))
        self.candidate_switch_min_delta = float(max(0.0, candidate_switch_min_delta))
        self.track_iou_match_threshold = float(max(0.0, track_iou_match_threshold))
        self.track_max_missed_frames = max(1, int(track_max_missed_frames))
        self.hand_representative_point = str(hand_representative_point)
        self.object_representative_point = str(object_representative_point)
        self.roi_margin_px = float(max(0.0, roi_margin_px))
        self.roi_margin_ratio = float(max(0.0, roi_margin_ratio))

        self._next_track_id = 1
        self._next_det_tmp_id = -1
        self._hand_tracks: dict[int, _Track] = {}
        self._object_tracks: dict[int, _Track] = {}
        self._det_fallback_tracks: dict[int, _Track] = {}
        self._hand_states: dict[int, _HandState] = {}

    @staticmethod
    def _timestamp_ms(frame_result: dict[str, Any]) -> int:
        ts = frame_result.get("timestamp_ms")
        return int(ts) if ts is not None else int(time.time() * 1000)

    def _point_in_roi(
        self,
        *,
        box: list[float],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
        point_kind: str,
    ) -> bool:
        if roi_poly is None or len(roi_poly) < 3:
            return True
        h, w = frame_shape[:2]
        px, py = self._representative_point_xy(box=box, frame_shape=frame_shape, point_kind=point_kind)
        dist = float(cv2.pointPolygonTest(roi_poly.astype(np.float32), (float(px), float(py)), True))
        margin = self.roi_margin_px + self.roi_margin_ratio * float((w * w + h * h) ** 0.5)
        return bool(dist >= -margin)

    @staticmethod
    def _representative_point_xy(
        *,
        box: list[float],
        frame_shape: tuple[int, ...],
        point_kind: str,
    ) -> tuple[float, float]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        if point_kind == "center":
            return (x1 + x2) * 0.5 * w, (y1 + y2) * 0.5 * h
        if point_kind == "top_center":
            return (x1 + x2) * 0.5 * w, y1 * h
        if point_kind == "bottom_center":
            return (x1 + x2) * 0.5 * w, y2 * h
        if point_kind == "wrist":
            # Approximate wrist at lower-center region of hand box.
            return (x1 + x2) * 0.5 * w, (y1 + 0.9 * (y2 - y1)) * h
        return (x1 + x2) * 0.5 * w, y2 * h

    def _track_score(self, det_box: list[float], track_box: list[float]) -> float:
        iou = _box_iou(det_box, track_box)
        dist = _center_distance(det_box, track_box)
        dist_score = max(0.0, 1.0 - (dist / self.association_max_center_dist))
        return 0.7 * iou + 0.3 * dist_score

    def _inside_for_event(
        self,
        *,
        box: list[float],
        roi_poly_runtime: np.ndarray | None,
        roi_poly_original: np.ndarray | None,
        frame_shape: tuple[int, ...],
        point_kind: str,
        warp_on: bool,
        warp_matrix_inv: np.ndarray | None,
    ) -> bool:
        # Event inside/outside should be evaluated in original-space ROI when available.
        if isinstance(roi_poly_original, np.ndarray) and len(roi_poly_original) >= 3:
            px, py = self._representative_point_xy(box=box, frame_shape=frame_shape, point_kind=point_kind)
            if warp_on and isinstance(warp_matrix_inv, np.ndarray):
                try:
                    pvec = np.array([[[float(px), float(py)]]], dtype=np.float32)
                    porig = cv2.perspectiveTransform(pvec, warp_matrix_inv).reshape(-1, 2)[0]
                    px, py = float(porig[0]), float(porig[1])
                except Exception:
                    pass
            h, w = frame_shape[:2]
            margin = self.roi_margin_px + self.roi_margin_ratio * float((w * w + h * h) ** 0.5)
            dist = float(cv2.pointPolygonTest(roi_poly_original.astype(np.float32), (float(px), float(py)), True))
            return bool(dist >= -margin)
        return self._point_in_roi(
            box=box,
            roi_poly=roi_poly_runtime,
            frame_shape=frame_shape,
            point_kind=point_kind,
        )

    def _update_tracks(
        self,
        *,
        detections: list[dict[str, Any]],
        tracks: dict[int, _Track],
        frame_id: int,
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
        point_kind: str,
    ) -> list[_Track]:
        det_boxes: list[list[float]] = []
        for det in detections:
            box = det.get("box")
            if isinstance(box, list) and len(box) == 4:
                det_boxes.append([float(v) for v in box])

        candidates: list[tuple[float, int, int]] = []
        for det_idx, box in enumerate(det_boxes):
            for tid, tr in tracks.items():
                score = self._track_score(box, tr.box)
                if score > 0.0:
                    candidates.append((score, det_idx, tid))
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_det: set[int] = set()
        matched_track: set[int] = set()
        for score, det_idx, tid in candidates:
            if det_idx in matched_det or tid in matched_track:
                continue
            box = det_boxes[det_idx]
            iou = _box_iou(box, tracks[tid].box)
            if iou < self.track_iou_match_threshold and score < 0.2:
                continue
            tr = tracks[tid]
            tr.box = box
            tr.last_frame = frame_id
            tr.missed = 0
            tr.inside_roi = self._point_in_roi(
                box=box,
                roi_poly=roi_poly,
                frame_shape=frame_shape,
                point_kind=point_kind,
            )
            matched_det.add(det_idx)
            matched_track.add(tid)

        for tid in list(tracks.keys()):
            if tid in matched_track:
                continue
            tr = tracks[tid]
            tr.missed += 1
            if tr.missed > self.track_max_missed_frames:
                tracks.pop(tid, None)

        for det_idx, box in enumerate(det_boxes):
            if det_idx in matched_det:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            tracks[tid] = _Track(
                track_id=tid,
                box=box,
                last_frame=frame_id,
                missed=0,
                inside_roi=self._point_in_roi(
                    box=box,
                    roi_poly=roi_poly,
                    frame_shape=frame_shape,
                    point_kind=point_kind,
                ),
            )

        return [tr for tr in tracks.values() if tr.missed == 0]

    def _build_association_topk(
        self,
        *,
        hand_track: _Track,
        object_tracks: list[_Track],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        scored: list[dict[str, Any]] = []
        debug_counts = {
            "objects_total": int(len(object_tracks)),
            "below_min_iou": 0,
            "below_min_score": 0,
            "dist_out_of_range": 0,
        }
        for obj in object_tracks:
            iou = _box_iou(hand_track.box, obj.box)
            center_dist = _center_distance(hand_track.box, obj.box)
            dist_score = max(0.0, 1.0 - (center_dist / self.association_max_center_dist))
            score = self.association_iou_weight * iou + self.association_dist_weight * dist_score
            min_score_th = self.det_fallback_min_score if int(obj.track_id) < 0 else self.association_min_score
            if iou < self.association_min_iou:
                debug_counts["below_min_iou"] += 1
                continue
            if dist_score <= 0.0:
                debug_counts["dist_out_of_range"] += 1
            if score < min_score_th:
                debug_counts["below_min_score"] += 1
                continue
            scored.append(
                {
                    "object_track_id": int(obj.track_id),
                    "score": float(score),
                    "iou": float(iou),
                    "center_dist": float(center_dist),
                    "inside_roi": bool(obj.inside_roi),
                }
            )
        scored.sort(key=lambda x: float(x["score"]), reverse=True)
        return scored[: self.association_top_k], debug_counts

    def _build_det_fallback_tracks(
        self,
        *,
        detections: list[dict[str, Any]],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
        frame_id: int,
    ) -> list[_Track]:
        det_boxes: list[list[float]] = []
        for det in detections:
            box = det.get("box")
            if not (isinstance(box, list) and len(box) == 4):
                continue
            det_boxes.append([float(v) for v in box])

        candidates: list[tuple[float, int, int]] = []
        for det_idx, box in enumerate(det_boxes):
            for tid, tr in self._det_fallback_tracks.items():
                score = self._track_score(box, tr.box)
                if score > 0.0:
                    candidates.append((score, det_idx, tid))
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_det: set[int] = set()
        matched_track: set[int] = set()
        for score, det_idx, tid in candidates:
            if det_idx in matched_det or tid in matched_track:
                continue
            box = det_boxes[det_idx]
            iou = _box_iou(box, self._det_fallback_tracks[tid].box)
            if iou < self.track_iou_match_threshold and score < 0.2:
                continue
            tr = self._det_fallback_tracks[tid]
            tr.box = box
            tr.last_frame = frame_id
            tr.missed = 0
            tr.inside_roi = self._point_in_roi(
                box=box,
                roi_poly=roi_poly,
                frame_shape=frame_shape,
                point_kind=self.object_representative_point,
            )
            matched_det.add(det_idx)
            matched_track.add(tid)

        for tid in list(self._det_fallback_tracks.keys()):
            if tid in matched_track:
                continue
            tr = self._det_fallback_tracks[tid]
            tr.missed += 1
            if tr.missed > 2:
                self._det_fallback_tracks.pop(tid, None)

        for det_idx, box in enumerate(det_boxes):
            if det_idx in matched_det:
                continue
            tid = self._next_det_tmp_id
            self._next_det_tmp_id -= 1
            self._det_fallback_tracks[tid] = _Track(
                track_id=tid,
                box=box,
                last_frame=frame_id,
                missed=0,
                inside_roi=self._point_in_roi(
                    box=box,
                    roi_poly=roi_poly,
                    frame_shape=frame_shape,
                    point_kind=self.object_representative_point,
                ),
            )
        return [tr for tr in self._det_fallback_tracks.values() if tr.missed == 0]

    @staticmethod
    def _emit_event(
        *,
        event_type: str,
        hand_track_id: int,
        object_track_id: int | None,
        confidence: float,
        frame_id: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        return {
            "event_type": str(event_type),
            "hand_track_id": int(hand_track_id),
            "object_track_id": int(object_track_id) if object_track_id is not None else None,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "frame_id": int(frame_id),
            "timestamp_ms": int(timestamp_ms),
        }

    @staticmethod
    def _timeout_reason(*, phase: str, has_candidate: bool, evidence_frames: int, fail_reason: str) -> str:
        if not has_candidate:
            return "timeout_no_candidate"
        if phase == "CANDIDATE_ADD":
            if evidence_frames <= 0:
                return fail_reason or "timeout_not_stable"
            return "timeout_not_enough_add_evidence"
        if phase == "CANDIDATE_REMOVE":
            if evidence_frames <= 0:
                return fail_reason or "timeout_object_not_lost"
            return "timeout_not_enough_remove_evidence"
        return "timeout"

    def update(self, frame_result: dict[str, Any]) -> dict[str, Any]:
        frame_id = int(frame_result.get("frame_id", 0))
        timestamp_ms = self._timestamp_ms(frame_result)
        hands = list(frame_result.get("hands", []))
        objects = list(frame_result.get("objects", []))
        objects_fallback = list(frame_result.get("objects_fallback", []))
        roi_poly = frame_result.get("roi_poly")
        roi_poly_original = frame_result.get("roi_poly_original")
        frame_shape = frame_result.get("frame_shape") or (720, 1280, 3)
        warp_on = bool(frame_result.get("warp_on", False))
        warp_matrix = frame_result.get("warp_matrix")
        warp_matrix_inv = frame_result.get("warp_matrix_inv")

        active_hands = self._update_tracks(
            detections=hands,
            tracks=self._hand_tracks,
            frame_id=frame_id,
            roi_poly=roi_poly,
            frame_shape=frame_shape,
            point_kind=self.hand_representative_point,
        )
        active_objects = self._update_tracks(
            detections=objects,
            tracks=self._object_tracks,
            frame_id=frame_id,
            roi_poly=roi_poly,
            frame_shape=frame_shape,
            point_kind=self.object_representative_point,
        )
        fallback_tracks = self._build_det_fallback_tracks(
            detections=objects_fallback,
            roi_poly=roi_poly,
            frame_shape=frame_shape,
            frame_id=frame_id,
        ) if self.allow_det_fallback else []

        # Normalize inside/outside for event logic in ORIGINAL ROI space.
        for tr in active_hands:
            tr.inside_roi = self._inside_for_event(
                box=tr.box,
                roi_poly_runtime=roi_poly,
                roi_poly_original=roi_poly_original,
                frame_shape=frame_shape,
                point_kind=self.hand_representative_point,
                warp_on=warp_on,
                warp_matrix_inv=warp_matrix_inv if isinstance(warp_matrix_inv, np.ndarray) else None,
            )
        for tr in active_objects:
            tr.inside_roi = self._inside_for_event(
                box=tr.box,
                roi_poly_runtime=roi_poly,
                roi_poly_original=roi_poly_original,
                frame_shape=frame_shape,
                point_kind=self.object_representative_point,
                warp_on=warp_on,
                warp_matrix_inv=warp_matrix_inv if isinstance(warp_matrix_inv, np.ndarray) else None,
            )
        for tr in fallback_tracks:
            tr.inside_roi = self._inside_for_event(
                box=tr.box,
                roi_poly_runtime=roi_poly,
                roi_poly_original=roi_poly_original,
                frame_shape=frame_shape,
                point_kind=self.object_representative_point,
                warp_on=warp_on,
                warp_matrix_inv=warp_matrix_inv if isinstance(warp_matrix_inv, np.ndarray) else None,
            )

        events: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        association_debug: list[dict[str, Any]] = []
        hand_debug: list[dict[str, Any]] = []
        object_debug = [
            {
                "track_id": int(obj.track_id),
                "box": obj.box,
                "inside_roi": bool(obj.inside_roi),
                "source": "track",
            }
            for obj in active_objects
        ]
        object_debug.extend(
            [
                {
                    "track_id": int(obj.track_id),
                    "box": obj.box,
                    "inside_roi": bool(obj.inside_roi),
                    "source": "det_fallback",
                }
                for obj in fallback_tracks
            ]
        )

        for hid in list(self._hand_states.keys()):
            if hid not in self._hand_tracks:
                self._hand_states.pop(hid, None)

        object_by_id = {obj.track_id: obj for obj in active_objects}
        fallback_by_id = {obj.track_id: obj for obj in fallback_tracks}

        for hand in active_hands:
            hstate = self._hand_states.setdefault(hand.track_id, _HandState(prev_inside=bool(hand.inside_roi)))
            current_inside = bool(hand.inside_roi)
            prev_inside = bool(hstate.prev_inside)
            crossing = "NONE"
            if (not prev_inside) and current_inside:
                crossing = "OUT_TO_IN"
            elif prev_inside and (not current_inside):
                crossing = "IN_TO_OUT"

            is_candidate_phase = hstate.phase in {"CANDIDATE_ADD", "CANDIDATE_REMOVE"}
            assoc_pool = list(active_objects)
            det_fallback_used = False
            if is_candidate_phase and (not assoc_pool) and fallback_tracks:
                assoc_pool = list(fallback_tracks)
                det_fallback_used = True
            topk, assoc_counts = self._build_association_topk(hand_track=hand, object_tracks=assoc_pool)
            if det_fallback_used:
                assoc_counts["det_fallback_used"] = 1
            else:
                assoc_counts["det_fallback_used"] = 0
            topk_by_id = {int(c["object_track_id"]): c for c in topk}

            next_assoc: dict[int, _AssocInfo] = {}
            for cand in topk:
                oid = int(cand["object_track_id"])
                prev = hstate.assoc_streaks.get(oid, _AssocInfo())
                streak = (prev.streak + 1) if prev.last_frame == (frame_id - 1) else 1
                cand["streak"] = int(streak)
                next_assoc[oid] = _AssocInfo(streak=streak, last_frame=frame_id, last_score=float(cand["score"]))
            hstate.assoc_streaks = next_assoc

            stable_candidates = [c for c in topk if int(c.get("streak", 0)) >= self.association_stable_frames]
            best_stable = stable_candidates[0] if stable_candidates else None
            best_any = topk[0] if topk else None
            if best_stable is not None:
                hstate.last_stable_object_track_id = int(best_stable["object_track_id"])

            if timestamp_ms < hstate.cooldown_until_ms:
                hstate.phase = "COOLDOWN"
            elif hstate.phase == "COOLDOWN":
                hstate.phase = "IDLE"

            if hstate.phase != "COOLDOWN":
                if crossing == "OUT_TO_IN":
                    hstate.phase = "CANDIDATE_ADD"
                    hstate.created_at_ms = timestamp_ms
                    hstate.evidence_frames = 0
                    hstate.candidate_missing_frames = 0
                    hstate.last_fail_reason = ""
                    hstate.candidate_object_track_id = None
                    hstate.candidate_source = "none"
                    hstate.candidate_score = 0.0
                    hstate.candidate_streak = 0
                    hstate.candidate_box = None
                    if best_stable is not None:
                        hstate.candidate_object_track_id = int(best_stable["object_track_id"])
                        hstate.candidate_source = "crossing_stable"
                        hstate.candidate_score = float(best_stable["score"])
                        hstate.candidate_streak = int(best_stable.get("streak", 0))
                        obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                            hstate.candidate_object_track_id
                        )
                        hstate.candidate_box = list(obj.box) if obj is not None else None
                    elif best_any is not None:
                        hstate.candidate_object_track_id = int(best_any["object_track_id"])
                        hstate.candidate_source = "crossing_top1"
                        hstate.candidate_score = float(best_any["score"])
                        hstate.candidate_streak = int(best_any.get("streak", 0))
                        obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                            hstate.candidate_object_track_id
                        )
                        hstate.candidate_box = list(obj.box) if obj is not None else None
                    transitions.append(
                        {
                            "hand_track_id": int(hand.track_id),
                            "from": "IDLE",
                            "to": "CANDIDATE_ADD",
                            "reason": "crossing_out_to_in",
                            "candidate_object_track_id": hstate.candidate_object_track_id,
                            "candidate_source": hstate.candidate_source,
                        }
                    )
                elif crossing == "IN_TO_OUT":
                    hstate.phase = "CANDIDATE_REMOVE"
                    hstate.created_at_ms = timestamp_ms
                    hstate.evidence_frames = 0
                    hstate.candidate_missing_frames = 0
                    hstate.last_fail_reason = ""
                    hstate.candidate_object_track_id = None
                    hstate.candidate_source = "none"
                    hstate.candidate_score = 0.0
                    hstate.candidate_streak = 0
                    hstate.candidate_box = None
                    if best_stable is not None:
                        hstate.candidate_object_track_id = int(best_stable["object_track_id"])
                        hstate.candidate_source = "crossing_stable"
                        hstate.candidate_score = float(best_stable["score"])
                        hstate.candidate_streak = int(best_stable.get("streak", 0))
                        obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                            hstate.candidate_object_track_id
                        )
                        hstate.candidate_box = list(obj.box) if obj is not None else None
                    elif hstate.last_stable_object_track_id is not None:
                        hstate.candidate_object_track_id = int(hstate.last_stable_object_track_id)
                        hstate.candidate_source = "crossing_last_stable"
                    elif best_any is not None:
                        hstate.candidate_object_track_id = int(best_any["object_track_id"])
                        hstate.candidate_source = "crossing_top1"
                        hstate.candidate_score = float(best_any["score"])
                        hstate.candidate_streak = int(best_any.get("streak", 0))
                        obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                            hstate.candidate_object_track_id
                        )
                        hstate.candidate_box = list(obj.box) if obj is not None else None
                    transitions.append(
                        {
                            "hand_track_id": int(hand.track_id),
                            "from": "IDLE",
                            "to": "CANDIDATE_REMOVE",
                            "reason": "crossing_in_to_out",
                            "candidate_object_track_id": hstate.candidate_object_track_id,
                            "candidate_source": hstate.candidate_source,
                        }
                    )

            # Late binding while candidate state is active.
            if hstate.phase in {"CANDIDATE_ADD", "CANDIDATE_REMOVE"}:
                if hstate.candidate_object_track_id is None:
                    bind = best_stable or best_any
                    if bind is not None:
                        hstate.candidate_object_track_id = int(bind["object_track_id"])
                        hstate.candidate_source = "late_bind_stable" if bind is best_stable else "late_bind_top1"
                        hstate.candidate_score = float(bind["score"])
                        hstate.candidate_streak = int(bind.get("streak", 0))
                        obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                            hstate.candidate_object_track_id
                        )
                        hstate.candidate_box = list(obj.box) if obj is not None else None
                        hstate.evidence_frames = 0
                        hstate.candidate_missing_frames = 0
                        transitions.append(
                            {
                                "hand_track_id": int(hand.track_id),
                                "from": hstate.phase,
                                "to": hstate.phase,
                                "reason": "late_bind_candidate",
                                "candidate_object_track_id": hstate.candidate_object_track_id,
                                "candidate_source": hstate.candidate_source,
                            }
                        )
                else:
                    current = topk_by_id.get(int(hstate.candidate_object_track_id))
                    best = best_stable or best_any
                    if current is not None:
                        hstate.candidate_score = float(current.get("score", hstate.candidate_score))
                        hstate.candidate_streak = int(current.get("streak", hstate.candidate_streak))
                        obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                            hstate.candidate_object_track_id
                        )
                        hstate.candidate_box = list(obj.box) if obj is not None else hstate.candidate_box
                    elif best is not None:
                        best_score = float(best.get("score", 0.0))
                        if best_score >= (float(hstate.candidate_score) + self.candidate_switch_min_delta):
                            old_cid = hstate.candidate_object_track_id
                            hstate.candidate_object_track_id = int(best["object_track_id"])
                            hstate.candidate_source = "late_switch_det" if int(hstate.candidate_object_track_id) < 0 else "late_switch"
                            hstate.candidate_score = best_score
                            hstate.candidate_streak = int(best.get("streak", 0))
                            obj = object_by_id.get(hstate.candidate_object_track_id) or fallback_by_id.get(
                                hstate.candidate_object_track_id
                            )
                            hstate.candidate_box = list(obj.box) if obj is not None else None
                            hstate.evidence_frames = 0
                            hstate.candidate_missing_frames = 0
                            transitions.append(
                                {
                                    "hand_track_id": int(hand.track_id),
                                    "from": hstate.phase,
                                    "to": hstate.phase,
                                    "reason": "late_switch_candidate",
                                    "candidate_object_track_id": hstate.candidate_object_track_id,
                                    "prev_candidate_object_track_id": old_cid,
                                }
                            )

            # Candidate evidence/confirm logic.
            if (
                hstate.phase in {"CANDIDATE_ADD", "CANDIDATE_REMOVE"}
                and hstate.candidate_object_track_id is not None
                and int(hstate.candidate_object_track_id) < 0
                and hstate.candidate_box is not None
                and active_objects
            ):
                best_match = None
                best_iou = 0.0
                for obj in active_objects:
                    iou = _box_iou(hstate.candidate_box, obj.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj
                if best_match is not None and best_iou >= 0.25:
                    prev_tmp_id = hstate.candidate_object_track_id
                    hstate.candidate_object_track_id = int(best_match.track_id)
                    hstate.candidate_source = "promoted_from_det"
                    hstate.candidate_box = list(best_match.box)
                    transitions.append(
                        {
                            "hand_track_id": int(hand.track_id),
                            "from": hstate.phase,
                            "to": hstate.phase,
                            "reason": "candidate_promoted_to_track",
                            "candidate_object_track_id": hstate.candidate_object_track_id,
                            "prev_candidate_object_track_id": prev_tmp_id,
                        }
                    )

            if hstate.phase == "CANDIDATE_ADD":
                oid = hstate.candidate_object_track_id
                obj = (object_by_id.get(oid) or fallback_by_id.get(oid)) if oid is not None else None
                if oid is None:
                    hstate.last_fail_reason = "no_candidate_yet"
                elif obj is None:
                    hstate.candidate_missing_frames += 1
                    hstate.evidence_frames = 0
                    hstate.last_fail_reason = "object_lost"
                elif not bool(obj.inside_roi):
                    hstate.evidence_frames = 0
                    hstate.last_fail_reason = "object_not_inside"
                else:
                    hstate.evidence_frames += 1
                    hstate.candidate_missing_frames = 0
                    hstate.last_fail_reason = ""

                if hstate.evidence_frames >= self.add_evidence_frames and hstate.candidate_object_track_id is not None:
                    emit_oid = (
                        None
                        if int(hstate.candidate_object_track_id) < 0
                        else int(hstate.candidate_object_track_id)
                    )
                    conf = 0.6 + 0.4 * min(1.0, hstate.evidence_frames / float(self.add_evidence_frames))
                    events.append(
                        self._emit_event(
                            event_type="ADD",
                            hand_track_id=hand.track_id,
                            object_track_id=emit_oid,
                            confidence=conf,
                            frame_id=frame_id,
                            timestamp_ms=timestamp_ms,
                        )
                    )
                    transitions.append(
                        {
                            "hand_track_id": int(hand.track_id),
                            "from": "CANDIDATE_ADD",
                            "to": "CONFIRMED_ADD",
                            "reason": "enough_inside_evidence",
                            "candidate_object_track_id": hstate.candidate_object_track_id,
                            "evidence_frames": int(hstate.evidence_frames),
                        }
                    )
                    hstate.phase = "IDLE"
                    hstate.cooldown_until_ms = timestamp_ms + self.cooldown_ms
                    hstate.evidence_frames = 0

            elif hstate.phase == "CANDIDATE_REMOVE":
                oid = hstate.candidate_object_track_id
                obj = (object_by_id.get(oid) or fallback_by_id.get(oid)) if oid is not None else None
                if oid is None:
                    hstate.last_fail_reason = "no_candidate_yet"
                elif (obj is None) or (not bool(obj.inside_roi)):
                    hstate.candidate_missing_frames += 1
                    if hstate.candidate_missing_frames < self.remove_missing_grace_frames:
                        hstate.last_fail_reason = "remove_missing_grace"
                    else:
                        hstate.evidence_frames += 1
                        hstate.last_fail_reason = ""
                else:
                    hstate.evidence_frames = 0
                    hstate.candidate_missing_frames = 0
                    hstate.last_fail_reason = "object_not_lost"

                if (
                    hstate.evidence_frames >= self.remove_evidence_frames
                    and hstate.candidate_missing_frames >= self.remove_missing_grace_frames
                    and hstate.candidate_object_track_id is not None
                ):
                    emit_oid = (
                        None
                        if int(hstate.candidate_object_track_id) < 0
                        else int(hstate.candidate_object_track_id)
                    )
                    conf = 0.6 + 0.4 * min(1.0, hstate.evidence_frames / float(self.remove_evidence_frames))
                    events.append(
                        self._emit_event(
                            event_type="REMOVE",
                            hand_track_id=hand.track_id,
                            object_track_id=emit_oid,
                            confidence=conf,
                            frame_id=frame_id,
                            timestamp_ms=timestamp_ms,
                        )
                    )
                    transitions.append(
                        {
                            "hand_track_id": int(hand.track_id),
                            "from": "CANDIDATE_REMOVE",
                            "to": "CONFIRMED_REMOVE",
                            "reason": "object_missing_inside_roi",
                            "candidate_object_track_id": hstate.candidate_object_track_id,
                            "evidence_frames": int(hstate.evidence_frames),
                        }
                    )
                    hstate.phase = "IDLE"
                    hstate.cooldown_until_ms = timestamp_ms + self.cooldown_ms
                    hstate.evidence_frames = 0

            # timeout
            if hstate.phase in {"CANDIDATE_ADD", "CANDIDATE_REMOVE"}:
                age_ms = max(0, int(timestamp_ms - hstate.created_at_ms))
                if age_ms > self.candidate_timeout_ms:
                    reason = self._timeout_reason(
                        phase=hstate.phase,
                        has_candidate=(hstate.candidate_object_track_id is not None),
                        evidence_frames=int(hstate.evidence_frames),
                        fail_reason=str(hstate.last_fail_reason),
                    )
                    transitions.append(
                        {
                            "hand_track_id": int(hand.track_id),
                            "from": hstate.phase,
                            "to": "IDLE",
                            "reason": reason,
                            "candidate_object_track_id": hstate.candidate_object_track_id,
                            "evidence_frames": int(hstate.evidence_frames),
                        }
                    )
                    hstate.phase = "IDLE"
                    hstate.evidence_frames = 0
                    hstate.candidate_object_track_id = None

            timeout_remaining_ms = 0
            if hstate.phase in {"CANDIDATE_ADD", "CANDIDATE_REMOVE"}:
                timeout_remaining_ms = max(0, self.candidate_timeout_ms - int(timestamp_ms - hstate.created_at_ms))

            association_debug.append(
                {
                    "hand_track_id": int(hand.track_id),
                    "stable_object_track_id": (int(best_stable["object_track_id"]) if best_stable is not None else None),
                    "topk": topk,
                    "counts": assoc_counts,
                    "min_score": self.association_min_score,
                    "min_iou": self.association_min_iou,
                }
            )
            hand_debug.append(
                {
                    "hand_track_id": int(hand.track_id),
                    "box": hand.box,
                    "rep_point": self._representative_point_xy(
                        box=hand.box,
                        frame_shape=frame_shape,
                        point_kind=self.hand_representative_point,
                    ),
                    "rep_point_warp": None,
                    "rep_point_original": None,
                    "inside_warp": None,
                    "inside_original": None,
                    "inside_roi": current_inside,
                    "crossing": crossing,
                    "rep_point_kind": self.hand_representative_point,
                    "roi_margin_px": round(float(self.roi_margin_px), 3),
                    "roi_margin_ratio": round(float(self.roi_margin_ratio), 4),
                    "phase": hstate.phase,
                    "candidate_object_track_id": hstate.candidate_object_track_id,
                    "candidate_source": hstate.candidate_source,
                    "candidate_score": round(float(hstate.candidate_score), 4),
                    "candidate_streak": int(hstate.candidate_streak),
                    "stable_object_track_id": (int(best_stable["object_track_id"]) if best_stable is not None else None),
                    "top1_object_track_id": (int(best_any["object_track_id"]) if best_any is not None else None),
                    "evidence_frames": int(hstate.evidence_frames),
                    "timeout_remaining_ms": int(timeout_remaining_ms),
                    "last_fail_reason": hstate.last_fail_reason,
                }
            )
            # Coordinate-space diagnostic: compare original vs warp-space inside tests.
            if hand_debug:
                hdbg = hand_debug[-1]
                rp_cur = hdbg.get("rep_point")
                if isinstance(rp_cur, tuple) and len(rp_cur) == 2:
                    px, py = float(rp_cur[0]), float(rp_cur[1])
                    rp_vec = np.array([[[px, py]]], dtype=np.float32)
                    if warp_on:
                        hdbg["rep_point_warp"] = (round(px, 1), round(py, 1))
                        if isinstance(warp_matrix_inv, np.ndarray):
                            try:
                                rp_orig = cv2.perspectiveTransform(rp_vec, warp_matrix_inv).reshape(-1, 2)[0]
                                hdbg["rep_point_original"] = (round(float(rp_orig[0]), 1), round(float(rp_orig[1]), 1))
                            except Exception:
                                hdbg["rep_point_original"] = None
                    else:
                        hdbg["rep_point_original"] = (round(px, 1), round(py, 1))
                        if isinstance(warp_matrix, np.ndarray):
                            try:
                                rp_warp = cv2.perspectiveTransform(rp_vec, warp_matrix).reshape(-1, 2)[0]
                                hdbg["rep_point_warp"] = (round(float(rp_warp[0]), 1), round(float(rp_warp[1]), 1))
                            except Exception:
                                hdbg["rep_point_warp"] = None

                    if isinstance(roi_poly, np.ndarray) and len(roi_poly) >= 3 and hdbg.get("rep_point_warp") is not None:
                        p = hdbg.get("rep_point_warp")
                        hdbg["inside_warp"] = bool(
                            cv2.pointPolygonTest(
                                roi_poly.astype(np.float32),
                                (float(p[0]), float(p[1])),
                                False,
                            ) >= 0
                        )
                    if (
                        isinstance(roi_poly_original, np.ndarray)
                        and len(roi_poly_original) >= 3
                        and hdbg.get("rep_point_original") is not None
                    ):
                        p = hdbg.get("rep_point_original")
                        hdbg["inside_original"] = bool(
                            cv2.pointPolygonTest(
                                roi_poly_original.astype(np.float32),
                                (float(p[0]), float(p[1])),
                                False,
                            ) >= 0
                        )
            hstate.prev_inside = current_inside

        ws_debug = {
            "hands": [
                {
                    "hand_track_id": h.get("hand_track_id"),
                    "phase": h.get("phase"),
                    "crossing": h.get("crossing"),
                    "rep_point_kind": h.get("rep_point_kind"),
                    "candidate_object_track_id": h.get("candidate_object_track_id"),
                    "candidate_score": h.get("candidate_score"),
                    "candidate_streak": h.get("candidate_streak"),
                    "stable_object_track_id": h.get("stable_object_track_id"),
                    "top1_object_track_id": h.get("top1_object_track_id"),
                    "evidence_frames": h.get("evidence_frames"),
                    "timeout_remaining_ms": h.get("timeout_remaining_ms"),
                    "last_fail_reason": h.get("last_fail_reason"),
                }
                for h in hand_debug
            ],
            "events": events,
            "transitions": transitions[-5:],
            "counts": {
                "hands_tracked": len(active_hands),
                "objects_tracked": len(active_objects),
                "objects_fallback": len(fallback_tracks),
            },
        }

        return {
            "events": events,
            "hands": hand_debug,
            "associations": association_debug,
            "object_tracks": object_debug,
            "transitions": transitions,
            "ws_debug": ws_debug,
            "counts": {
                "hands_tracked": len(active_hands),
                "objects_tracked": len(active_objects),
                "objects_fallback": len(fallback_tracks),
            },
        }
