from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping
from typing import Any

import cv2
import numpy as np

from checkout_core.inference import build_query_embedding

logger = logging.getLogger("checkout.frame_processor")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # pragma: no cover - optional dependency
    DeepSort = None  # type: ignore[assignment]


class _SimpleTracker:
    """Minimal IOU tracker used when deep_sort_realtime is unavailable."""

    def __init__(self, max_age: int = 12, n_init: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max(1, int(max_age))
        self.n_init = max(1, int(n_init))
        self.iou_threshold = float(iou_threshold)
        self._next_id = 1
        self._tracks: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union = max(1e-6, area_a + area_b - inter)
        return inter / union

    def update_tracks(self, detections, frame=None):  # noqa: ANN001, ANN201, ARG002
        del frame
        det_boxes: list[tuple[int, int, int, int]] = []
        for det in detections:
            ltrwh = det[0]
            x, y, w, h = ltrwh
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
            det_boxes.append((x1, y1, x2, y2))

        for meta in self._tracks.values():
            meta["lost"] = int(meta.get("lost", 0)) + 1

        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(det_boxes)))
        matches: list[tuple[str, int]] = []

        if self._tracks and det_boxes:
            pairs: list[tuple[float, str, int]] = []
            for tid, meta in self._tracks.items():
                tbox = meta["bbox"]
                for didx, dbox in enumerate(det_boxes):
                    iou = self._iou(tbox, dbox)
                    if iou >= self.iou_threshold:
                        pairs.append((iou, tid, didx))
            pairs.sort(reverse=True, key=lambda item: item[0])

            used_t: set[str] = set()
            used_d: set[int] = set()
            for _, tid, didx in pairs:
                if tid in used_t or didx in used_d:
                    continue
                used_t.add(tid)
                used_d.add(didx)
                matches.append((tid, didx))

        for tid, didx in matches:
            meta = self._tracks[tid]
            meta["bbox"] = det_boxes[didx]
            meta["hits"] = int(meta.get("hits", 0)) + 1
            meta["lost"] = 0
            meta["confirmed"] = bool(meta["hits"] >= self.n_init)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(didx)

        for didx in sorted(unmatched_dets):
            tid = str(self._next_id)
            self._next_id += 1
            self._tracks[tid] = {
                "bbox": det_boxes[didx],
                "hits": 1,
                "lost": 0,
                "confirmed": self.n_init <= 1,
            }

        for tid in list(unmatched_tracks):
            meta = self._tracks.get(tid)
            if meta is None:
                continue
            if int(meta.get("lost", 0)) > self.max_age:
                self._tracks.pop(tid, None)

        out = []
        for tid, meta in self._tracks.items():
            if int(meta.get("lost", 0)) > 0:
                continue
            out.append({
                "track_id": tid,
                "bbox": meta["bbox"],
                "hits": int(meta.get("hits", 1)),
                "confirmed": bool(meta.get("confirmed", False)),
                "time_since_update": int(meta.get("lost", 0)),
            })
        return out


def _build_motion_mask(frame: np.ndarray, bg_subtractor, roi_poly: np.ndarray | None) -> np.ndarray:
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=4)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    if roi_poly is not None:
        roi_mask = np.zeros_like(fg_mask)
        cv2.fillPoly(roi_mask, [roi_poly], 255)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    return fg_mask


def _expand_tlwh_with_padding(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    frame_w: int,
    frame_h: int,
    pad_ratio: float,
) -> list[float]:
    pad_w = int(round(max(1.0, w * max(0.0, float(pad_ratio)))))
    pad_h = int(round(max(1.0, h * max(0.0, float(pad_ratio)))))

    left = max(0, x - pad_w)
    top = max(0, y - pad_h)
    right = min(frame_w, x + w + pad_w)
    bottom = min(frame_h, y + h + pad_h)

    new_w = max(1, right - left)
    new_h = max(1, bottom - top)
    return [float(left), float(top), float(new_w), float(new_h)]


def _motion_detections(
    fg_mask: np.ndarray,
    min_area: int,
    *,
    frame_shape: tuple[int, ...],
    bbox_pad_ratio: float,
) -> tuple[list[tuple[list[float], float, str]], int]:
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = frame_shape[:2]

    detections: list[tuple[list[float], float, str]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= float(min_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        tlwh = _expand_tlwh_with_padding(
            x,
            y,
            w,
            h,
            frame_w=frame_w,
            frame_h=frame_h,
            pad_ratio=bbox_pad_ratio,
        )
        confidence = float(min(0.99, 0.35 + (area / max(float(min_area) * 25.0, 1.0))))
        detections.append((tlwh, confidence, "motion"))

    return detections, len(contours)


def _normalize_embedder_mode(mode: str) -> str:
    normalized = (mode or "simple").strip().lower()
    return normalized if normalized else "simple"


def _ensure_tracker(
    state: MutableMapping[str, Any],
    *,
    use_deepsort: bool,
    track_max_age: int,
    track_n_init: int,
    track_max_iou_distance: float,
    deepsort_embedder_mode: str,
    deepsort_max_cosine_distance: float,
    deepsort_gating_only_position: bool,
) -> tuple[Any, str, str]:
    tracker = state.get("_tracker")
    tracker_mode = str(state.get("_tracker_mode", ""))
    embedder_mode = _normalize_embedder_mode(deepsort_embedder_mode)
    target_mode = "deepsort" if use_deepsort and DeepSort is not None else "simple"

    tracker_key = (
        target_mode,
        track_max_age,
        track_n_init,
        float(track_max_iou_distance),
        embedder_mode,
        float(deepsort_max_cosine_distance),
        bool(deepsort_gating_only_position),
    )
    if tracker is not None and tracker_mode == target_mode and state.get("_tracker_key") == tracker_key:
        return tracker, tracker_mode, embedder_mode

    if target_mode == "deepsort":
        deepsort_embedder = None if embedder_mode in {"simple", "none"} else embedder_mode
        kwargs_base = {
            "max_age": track_max_age,
            "n_init": track_n_init,
            "max_iou_distance": track_max_iou_distance,
        }
        constructor_kwargs = [
            {
                **kwargs_base,
                "max_cosine_distance": deepsort_max_cosine_distance,
                "gating_only_position": deepsort_gating_only_position,
                "embedder": deepsort_embedder,
                "bgr": True,
            },
            {
                **kwargs_base,
                "max_cosine_distance": deepsort_max_cosine_distance,
                "embedder": deepsort_embedder,
                "bgr": True,
            },
            {
                **kwargs_base,
                "max_cosine_distance": deepsort_max_cosine_distance,
                "embedder": deepsort_embedder,
            },
            {
                **kwargs_base,
                "embedder": deepsort_embedder,
            },
            kwargs_base,
        ]

        tracker = None
        for kwargs in constructor_kwargs:
            try:
                tracker = DeepSort(**kwargs)  # type: ignore[misc]
                break
            except TypeError:
                continue
        if tracker is None:
            raise RuntimeError("Failed to initialize DeepSort with current runtime arguments.")
    else:
        tracker = _SimpleTracker(
            max_age=track_max_age,
            n_init=track_n_init,
            iou_threshold=track_max_iou_distance,
        )
        if use_deepsort and DeepSort is None:
            state["last_status"] = "tracking(simple_fallback)"

    state["_tracker"] = tracker
    state["_tracker_mode"] = target_mode
    state["_tracker_key"] = tracker_key
    return tracker, target_mode, embedder_mode


def _simple_hs_embedding(frame: np.ndarray, tlwh: list[float], hs_bins: int) -> np.ndarray:
    bins = max(4, int(hs_bins))
    dim = bins * bins
    fallback = np.full((dim,), 1.0 / np.sqrt(float(dim)), dtype=np.float32)

    x, y, w, h = tlwh
    x1 = int(max(0, np.floor(x)))
    y1 = int(max(0, np.floor(y)))
    x2 = int(min(frame.shape[1], np.ceil(x + w)))
    y2 = int(min(frame.shape[0], np.ceil(y + h)))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return fallback.copy()

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return fallback.copy()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256]).astype(np.float32)
    hist = hist.flatten()
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-8:
        return fallback.copy()
    hist /= norm
    return hist


def _build_simple_embeds(
    frame: np.ndarray,
    detections: list[tuple[list[float], float, str]],
    hs_bins: int,
) -> list[np.ndarray]:
    embeds: list[np.ndarray] = []
    for tlwh, _, _ in detections:
        embeds.append(_simple_hs_embedding(frame, tlwh, hs_bins))
    return embeds


def _update_tracks(
    tracker: Any,
    detections: list[tuple[list[float], float, str]],
    frame: np.ndarray,
    *,
    embedder_mode: str,
    simple_hs_bins: int,
    tsu_tolerance: int,
) -> tuple[list[dict[str, Any]], dict[int, int], int]:
    if isinstance(tracker, _SimpleTracker):
        simple_tracks = tracker.update_tracks(detections, frame=frame)
        tsu_hist: dict[int, int] = {}
        for track in simple_tracks:
            tsu = int(track.get("time_since_update", 0))
            tsu_hist[tsu] = int(tsu_hist.get(tsu, 0)) + 1
        return simple_tracks, tsu_hist, len(simple_tracks)

    try:
        if embedder_mode in {"simple", "none"}:
            embeds = _build_simple_embeds(frame, detections, simple_hs_bins)
            raw_tracks = tracker.update_tracks(detections, embeds=embeds, frame=frame)
        else:
            raw_tracks = tracker.update_tracks(detections, frame=frame)
    except Exception:
        logger.exception(
            "DeepSORT update failed: mode=%s det_count=%d",
            embedder_mode,
            len(detections),
        )
        return [], {}, 0

    tracks: list[dict[str, Any]] = []
    tsu_hist: dict[int, int] = {}
    tolerance = max(0, int(tsu_tolerance))

    for track in raw_tracks:
        tsu = int(getattr(track, "time_since_update", 0))
        tsu_hist[tsu] = int(tsu_hist.get(tsu, 0)) + 1
        if tsu > tolerance:
            continue

        bbox = None
        try:
            bbox = track.to_ltrb(orig=True)
        except TypeError:
            bbox = track.to_ltrb()
        except Exception:
            bbox = None
        if bbox is None:
            try:
                bbox = track.to_ltrb()
            except Exception:
                continue
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue

        try:
            confirmed = bool(track.is_confirmed())
        except Exception:
            confirmed = bool(getattr(track, "confirmed", False))

        tracks.append({
            "track_id": str(track.track_id),
            "bbox": (x1, y1, x2, y2),
            "hits": int(getattr(track, "hits", 1)),
            "confirmed": confirmed,
            "time_since_update": tsu,
        })
    return tracks, tsu_hist, len(raw_tracks)


def _classify_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    model_bundle,
    faiss_index,
    labels,
    match_threshold: float,
) -> tuple[str | None, float]:
    if faiss_index is None or faiss_index.ntotal <= 0 or labels is None or len(labels) == 0:
        return None, 0.0

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame.shape[1] - 1, x1))
    x2 = max(0, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    y2 = max(0, min(frame.shape[0], y2))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None, 0.0

    crop = frame[y1:y2, x1:x2]
    emb = build_query_embedding(crop, model_bundle)
    query = np.expand_dims(emb, axis=0)
    distances, indices = faiss_index.search(query, 1)
    best_score = float(distances[0][0])
    best_idx = int(indices[0][0])
    if best_score >= match_threshold and 0 <= best_idx < len(labels):
        return str(labels[best_idx]), best_score
    return None, best_score


def _side_of_gate(center_y: float, gate_y: int) -> str:
    return "above" if center_y < gate_y else "below"


def _should_reclassify(
    meta: MutableMapping[str, Any],
    *,
    infer_count: int,
    area: float,
    near_gate: bool,
    reclassify_every_n_frames: int,
    reclassify_area_gain: float,
) -> bool:
    if meta.get("label") is None:
        return True

    last_infer = int(meta.get("last_classified_infer", -10_000))
    if infer_count - last_infer < max(1, reclassify_every_n_frames):
        return False

    best_area = float(meta.get("best_area", 0.0))
    better_view = area >= max(1.0, best_area * max(1.0, reclassify_area_gain))
    weak_score = float(meta.get("score", 0.0)) < 0.75
    return better_view or (near_gate and weak_score)


def _resolve_label_vote(meta: MutableMapping[str, Any], label: str, score: float) -> None:
    votes = meta.setdefault("votes", {})
    if not isinstance(votes, dict):
        votes = {}
        meta["votes"] = votes

    # Prevent early label lock-in by decaying old votes.
    for key in list(votes.keys()):
        decayed = float(votes.get(key, 0.0)) * 0.85
        if decayed < 0.01:
            votes.pop(key, None)
        else:
            votes[key] = decayed

    votes[label] = float(votes.get(label, 0.0)) + 1.0

    # Cap total vote mass to keep adaptation responsive.
    total_vote = float(sum(float(v) for v in votes.values()))
    if total_vote > 24.0:
        scale = 24.0 / total_vote
        for key in list(votes.keys()):
            votes[key] = float(votes[key]) * scale

    current_label = meta.get("label")
    if current_label is None:
        meta["label"] = label
        meta["score"] = float(score)
        return

    if current_label == label:
        meta["score"] = max(float(meta.get("score", 0.0)), float(score))
        return

    current_votes = float(votes.get(current_label, 0.0))
    candidate_votes = float(votes.get(label, 0.0))
    current_score = float(meta.get("score", 0.0))
    if candidate_votes > current_votes or (candidate_votes == current_votes and score > current_score):
        meta["label"] = label
        meta["score"] = float(score)


def _compute_direction(
    prev_side: str | None,
    side: str,
    delta_y: float,
    min_delta: float,
) -> tuple[str | None, str]:
    if prev_side is None or prev_side == side:
        return None, "side_no_change"
    if abs(delta_y) < max(0.0, float(min_delta)):
        return None, "dy_low"
    if prev_side == "above" and side == "below" and delta_y > 0:
        return "IN", "ok"
    if prev_side == "below" and side == "above" and delta_y < 0:
        return "OUT", "ok"
    return None, "dy_low"


def _format_tsu_hist(tsu_hist: dict[int, int]) -> str:
    if not tsu_hist:
        return "-"
    parts = [f"{k}:{tsu_hist[k]}" for k in sorted(tsu_hist.keys())]
    return ",".join(parts)


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    union = max(1e-6, area_a + area_b - inter)
    return inter / union


def create_bg_subtractor():
    return cv2.createBackgroundSubtractorKNN(
        history=300,
        dist2Threshold=500,
        detectShadows=False,
    )


def process_checkout_frame(
    *,
    frame: np.ndarray,
    frame_count: int,
    bg_subtractor,
    model_bundle,
    faiss_index,
    labels,
    state: MutableMapping[str, Any],
    min_area: int,
    detect_every_n_frames: int,
    match_threshold: float,
    cooldown_seconds: float,
    roi_poly: np.ndarray | None = None,
    roi_clear_frames: int = 8,
    roi_entry_mode: bool = False,
    use_deepsort: bool = True,
    track_n_init: int = 3,
    track_max_age: int = 12,
    track_max_iou_distance: float = 0.9,
    track_stale_frames: int = 45,
    direction_gate_y_norm: float = 0.55,
    direction_min_delta_px: float = 18.0,
    direction_event_cooldown_sec: float | None = None,
    reclassify_every_n_frames: int = 6,
    reclassify_area_gain: float = 1.15,
    direction_event_band_px: int = 36,
    deepsort_embedder_mode: str = "simple",
    deepsort_max_cosine_distance: float = 0.45,
    deepsort_gating_only_position: bool = True,
    deepsort_bbox_pad_ratio: float = 0.08,
    deepsort_tsu_tolerance: int = 1,
    deepsort_simple_hs_bins: int = 16,
    pipeline_debug: bool = False,
    pipeline_log_every_n_infer: int = 30,
) -> np.ndarray:
    """Motion detection + DeepSORT tracking + direction-based IN/OUT updates."""
    del detect_every_n_frames
    del roi_clear_frames
    del roi_entry_mode

    display_frame = frame.copy()
    state.setdefault("billing_items", {})
    state.setdefault("item_scores", {})
    state.setdefault("last_label", "-")
    state.setdefault("last_score", 0.0)
    state.setdefault("last_status", "idle")
    state.setdefault("last_direction", "-")

    infer_count = int(state.get("_infer_count", 0)) + 1
    state["_infer_count"] = infer_count

    fg_mask = _build_motion_mask(frame, bg_subtractor, roi_poly)
    detections, contour_count = _motion_detections(
        fg_mask,
        min_area,
        frame_shape=frame.shape,
        bbox_pad_ratio=deepsort_bbox_pad_ratio,
    )

    tracker, tracker_mode, embedder_mode = _ensure_tracker(
        state,
        use_deepsort=use_deepsort,
        track_max_age=track_max_age,
        track_n_init=track_n_init,
        track_max_iou_distance=track_max_iou_distance,
        deepsort_embedder_mode=deepsort_embedder_mode,
        deepsort_max_cosine_distance=deepsort_max_cosine_distance,
        deepsort_gating_only_position=deepsort_gating_only_position,
    )
    tracks, tsu_hist, raw_track_total = _update_tracks(
        tracker,
        detections,
        frame,
        embedder_mode=embedder_mode,
        simple_hs_bins=deepsort_simple_hs_bins,
        tsu_tolerance=deepsort_tsu_tolerance,
    )
    track_meta = state.setdefault("_track_meta", {})

    debug_counts = {
        "confirmed": 0,
        "not_confirmed": 0,
        "hits_low": 0,
        "side_no_change": 0,
        "dy_low": 0,
        "label_missing": 0,
        "id_switch_reset": 0,
        "event_guard_track": 0,
        "event_guard_warmup": 0,
        "event_guard_stale_label": 0,
        "events_emitted": 0,
    }

    gate_y = int(max(0, min(frame.shape[0] - 1, frame.shape[0] * float(direction_gate_y_norm))))
    cv2.line(display_frame, (0, gate_y), (frame.shape[1] - 1, gate_y), (255, 200, 0), 2)
    cv2.putText(
        display_frame,
        "OUT (up) | IN (down)",
        (8, max(24, gate_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2,
    )

    active_ids: set[str] = set()
    event_cooldown = (
        float(cooldown_seconds)
        if direction_event_cooldown_sec is None
        else float(direction_event_cooldown_sec)
    )

    if tracks:
        state["last_status"] = f"tracking({tracker_mode})"
    else:
        state["last_label"] = "-"
        state["last_score"] = 0.0
        state["last_direction"] = "-"
        state["last_status"] = "no_track"

    for track in tracks:
        tid = str(track["track_id"])
        active_ids.add(tid)

        hits = int(track.get("hits", 1))
        confirmed = bool(track.get("confirmed", False))
        tsu = int(track.get("time_since_update", 0))
        if confirmed:
            debug_counts["confirmed"] += 1
        else:
            debug_counts["not_confirmed"] += 1
        if hits < max(1, track_n_init):
            debug_counts["hits_low"] += 1

        x1, y1, x2, y2 = track["bbox"]
        x1 = max(0, min(frame.shape[1] - 1, x1))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(frame.shape[1], x2))
        y2 = max(y1 + 1, min(frame.shape[0], y2))

        width = x2 - x1
        height = y2 - y1
        area = float(width * height)
        center_y = y1 + (height / 2.0)
        side = _side_of_gate(center_y, gate_y)
        near_gate = abs(center_y - gate_y) <= max(4, int(direction_event_band_px))

        meta = track_meta.setdefault(tid, {
            "label": None,
            "score": 0.0,
            "votes": {},
            "last_classified_frame": -10_000,
            "last_classified_infer": -10_000,
            "best_area": 0.0,
            "last_side": None,
            "last_cy": None,
            "last_bbox": None,
            "warmup_until_frame": -1,
            "last_event_ts": -1e9,
            "last_seen_frame": frame_count,
        })
        meta["last_seen_frame"] = frame_count

        prev_bbox_raw = meta.get("last_bbox")
        if tsu == 0 and isinstance(prev_bbox_raw, tuple) and len(prev_bbox_raw) == 4:
            prev_bbox = (
                int(prev_bbox_raw[0]),
                int(prev_bbox_raw[1]),
                int(prev_bbox_raw[2]),
                int(prev_bbox_raw[3]),
            )
            curr_bbox = (x1, y1, x2, y2)
            prev_cx = (prev_bbox[0] + prev_bbox[2]) * 0.5
            prev_cy = (prev_bbox[1] + prev_bbox[3]) * 0.5
            curr_cx = (curr_bbox[0] + curr_bbox[2]) * 0.5
            curr_cy = (curr_bbox[1] + curr_bbox[3]) * 0.5
            center_jump = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))

            prev_area = float(max(1, prev_bbox[2] - prev_bbox[0]) * max(1, prev_bbox[3] - prev_bbox[1]))
            area_ratio = max(area / prev_area, prev_area / max(1.0, area))
            bbox_iou = _bbox_iou(prev_bbox, curr_bbox)

            iou_bad = bbox_iou < 0.05
            area_bad = area_ratio > 2.5
            jump_bad = center_jump > (frame.shape[1] * 0.25)
            id_switch_suspected = jump_bad or (iou_bad and area_bad)
            if id_switch_suspected:
                meta["label"] = None
                meta["score"] = 0.0
                meta["votes"] = {}
                meta["last_classified_frame"] = -10_000
                meta["last_classified_infer"] = -10_000
                meta["best_area"] = 0.0
                meta["last_side"] = None
                meta["last_cy"] = None
                meta["warmup_until_frame"] = frame_count + max(1, int(track_n_init))
                meta["last_event_ts"] = time.time()
                debug_counts["id_switch_reset"] += 1

        should_classify = confirmed and hits >= max(1, track_n_init) and _should_reclassify(
            meta,
            infer_count=infer_count,
            area=area,
            near_gate=near_gate,
            reclassify_every_n_frames=reclassify_every_n_frames,
            reclassify_area_gain=reclassify_area_gain,
        )

        if should_classify:
            label, score = _classify_crop(
                frame,
                (x1, y1, x2, y2),
                model_bundle=model_bundle,
                faiss_index=faiss_index,
                labels=labels,
                match_threshold=match_threshold,
            )
            meta["last_classified_frame"] = frame_count
            meta["last_classified_infer"] = infer_count
            meta["best_area"] = max(float(meta.get("best_area", 0.0)), area)

            if label is not None:
                _resolve_label_vote(meta, label, score)
                final_label = str(meta.get("label"))
                final_score = float(meta.get("score", score))
                state["last_label"] = final_label
                state["last_score"] = final_score
                state["item_scores"][final_label] = max(
                    float(state["item_scores"].get(final_label, 0.0)),
                    final_score,
                )
                state["last_status"] = f"classified({tracker_mode})"
            else:
                state["last_status"] = f"classify_failed({tracker_mode})"

        last_center_y = meta.get("last_cy")
        delta_y = float(center_y - float(last_center_y)) if last_center_y is not None else 0.0
        direction, direction_reason = _compute_direction(
            prev_side=meta.get("last_side"),
            side=side,
            delta_y=delta_y,
            min_delta=direction_min_delta_px,
        )
        if direction is None:
            if direction_reason == "side_no_change":
                debug_counts["side_no_change"] += 1
            elif direction_reason == "dy_low":
                debug_counts["dy_low"] += 1

        if direction is not None:
            warmup_until_frame = int(meta.get("warmup_until_frame", -1))
            stable_track_for_event = (
                confirmed
                and hits >= max(1, track_n_init)
                and tsu == 0
            )
            if frame_count <= warmup_until_frame:
                debug_counts["event_guard_warmup"] += 1
                direction = None
            elif not stable_track_for_event:
                debug_counts["event_guard_track"] += 1
                direction = None

        if direction is not None:
            now = time.time()
            if now - float(meta.get("last_event_ts", -1e9)) >= max(0.0, event_cooldown):
                label = meta.get("label")
                score = float(meta.get("score", 0.0))

                if label is None:
                    label_now, score_now = _classify_crop(
                        frame,
                        (x1, y1, x2, y2),
                        model_bundle=model_bundle,
                        faiss_index=faiss_index,
                        labels=labels,
                        match_threshold=match_threshold,
                    )
                    meta["last_classified_frame"] = frame_count
                    meta["last_classified_infer"] = infer_count
                    meta["best_area"] = max(float(meta.get("best_area", 0.0)), area)
                    if label_now is not None:
                        _resolve_label_vote(meta, label_now, score_now)
                        label = meta.get("label")
                        score = float(meta.get("score", score_now))
                    else:
                        debug_counts["label_missing"] += 1
                        state["last_status"] = "event_ignored_unclassified"

                if label is not None:
                    last_cls_frame = int(meta.get("last_classified_infer", -10_000))
                    freshness_window = max(2, int(reclassify_every_n_frames) * 2)
                    if infer_count - last_cls_frame > freshness_window:
                        debug_counts["event_guard_stale_label"] += 1
                        direction = None

                if direction is not None and label is not None:
                    billing_items = state["billing_items"]
                    if direction == "IN":
                        billing_items[str(label)] = int(billing_items.get(str(label), 0)) + 1
                        state["last_status"] = f"IN {label}"
                    else:
                        current_qty = int(billing_items.get(str(label), 0))
                        if current_qty > 0:
                            next_qty = current_qty - 1
                            if next_qty > 0:
                                billing_items[str(label)] = next_qty
                            else:
                                billing_items.pop(str(label), None)
                            state["last_status"] = f"OUT {label}"
                        else:
                            state["last_status"] = f"OUT_ignored {label}"

                    state["last_label"] = str(label)
                    state["last_score"] = float(score)
                    state["last_direction"] = direction
                    meta["last_event_ts"] = now
                    debug_counts["events_emitted"] += 1
            else:
                direction = None

        if tsu == 0:
            meta["last_side"] = side
            meta["last_cy"] = center_y
            meta["last_bbox"] = (x1, y1, x2, y2)

        draw_color = (0, 220, 0) if meta.get("label") else (0, 180, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), draw_color, 2)
        text = f"#{tid} {meta.get('label') or 'unknown'}"
        if direction is not None:
            text += f" {direction}"
        cv2.putText(
            display_frame,
            text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    stale_after = max(1, int(track_stale_frames))
    for tid in list(track_meta.keys()):
        if tid in active_ids:
            continue
        if frame_count - int(track_meta[tid].get("last_seen_frame", frame_count)) > stale_after:
            track_meta.pop(tid, None)

    if roi_poly is not None:
        cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

    if pipeline_debug and infer_count % max(1, int(pipeline_log_every_n_infer)) == 0:
        logger.info(
            "pipeline infer=%d contours=%d dets=%d tracks_raw=%d tracks_kept=%d "
            "confirmed=%d tsu=%s fail(not_confirmed=%d hits_low=%d side_no_change=%d dy_low=%d "
            "label_missing=%d id_switch_reset=%d event_guard_track=%d event_guard_warmup=%d event_guard_stale_label=%d) "
            "events=%d tracker=%s embedder=%s status=%s",
            infer_count,
            contour_count,
            len(detections),
            raw_track_total,
            len(tracks),
            debug_counts["confirmed"],
            _format_tsu_hist(tsu_hist),
            debug_counts["not_confirmed"],
            debug_counts["hits_low"],
            debug_counts["side_no_change"],
            debug_counts["dy_low"],
            debug_counts["label_missing"],
            debug_counts["id_switch_reset"],
            debug_counts["event_guard_track"],
            debug_counts["event_guard_warmup"],
            debug_counts["event_guard_stale_label"],
            debug_counts["events_emitted"],
            tracker_mode,
            embedder_mode,
            state.get("last_status", ""),
        )

    return display_frame
