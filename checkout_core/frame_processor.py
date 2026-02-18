from __future__ import annotations

from collections.abc import MutableMapping
import logging
import time
from typing import Any

import cv2
import numpy as np

from checkout_core.counting import should_count_product
from checkout_core.inference import build_query_embedding

try:
    from backend import config as backend_config
    from backend.utils.profiler import ProfileCollector
except Exception:  # pragma: no cover - optional in non-backend contexts
    backend_config = None
    ProfileCollector = None


logger = logging.getLogger("checkout_core.frame_processor")
_FRAME_PROFILER = None


def _get_frame_profiler():
    global _FRAME_PROFILER
    if backend_config is None or ProfileCollector is None:
        return None
    if not getattr(backend_config, "ENABLE_PROFILING", False):
        return None
    if _FRAME_PROFILER is None:
        _FRAME_PROFILER = ProfileCollector(
            kind="frame",
            enable=backend_config.ENABLE_PROFILING,
            every_n_frames=getattr(backend_config, "PROFILE_EVERY_N_FRAMES", 30),
            logger=logger,
        )
    return _FRAME_PROFILER.start_frame()


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
    yolo_detector=None,
) -> np.ndarray:
    """Process a single frame and update checkout state in-place.

    Args:
        yolo_detector: Optional YOLODetector instance. If provided, uses YOLO for detection
                      instead of background subtraction.
    """
    frame_profiler = _get_frame_profiler()
    total_start = time.perf_counter() if frame_profiler is not None else 0.0
    display_frame = frame.copy()

    # Initialize detection_boxes in state if not present
    if "detection_boxes" not in state:
        state["detection_boxes"] = []

    # Choose detection method: YOLO or background subtraction
    if yolo_detector is not None:
        # YOLO-based detection
        if frame_profiler is not None:
            with frame_profiler.measure("detect"):
                detections = yolo_detector.detect(frame)
        else:
            detections = yolo_detector.detect(frame)
        state["detection_boxes"] = detections  # Store all detections (product + hand)

        # Filter only product detections for embedding
        product_detections = [d for d in detections if d["class"] == "product"]

        # Process each detected product
        for detection in product_detections:
            box = detection["box"]
            crop = yolo_detector.extract_crop(frame, box)

            if crop is None:
                continue

            # Check if inside ROI (if ROI mode is enabled)
            if roi_poly is not None:
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]
                cx = (x1 + x2) / 2 * w
                cy = (y1 + y2) / 2 * h
                inside_roi = cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0

                if inside_roi:
                    state["roi_empty_frames"] = 0
                    entry_event = not bool(state.get("roi_occupied", False))
                    state["roi_occupied"] = True
                else:
                    state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1
                    continue  # Skip products outside ROI

                # Check ROI clear
                if int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                    state["roi_occupied"] = False

                # Check if we should run inference (entry event or periodic)
                if roi_entry_mode:
                    periodic_slot = frame_count % max(1, detect_every_n_frames) == 0
                    allow_inference = inside_roi and (entry_event or periodic_slot)
                    if inside_roi:
                        state["last_status"] = "ROI 진입" if entry_event else "ROI 내부"
                    else:
                        state["last_status"] = "ROI 외부"
                        continue
                else:
                    allow_inference = frame_count % max(1, detect_every_n_frames) == 0
            else:
                allow_inference = frame_count % max(1, detect_every_n_frames) == 0

            # Run embedding + FAISS matching
            if allow_inference and faiss_index is not None and faiss_index.ntotal > 0:
                if frame_profiler is not None:
                    with frame_profiler.measure("embed"):
                        emb = build_query_embedding(crop, model_bundle)
                else:
                    emb = build_query_embedding(crop, model_bundle)
                query = np.expand_dims(emb, axis=0)

                if frame_profiler is not None:
                    with frame_profiler.measure("faiss"):
                        distances, indices = faiss_index.search(query, 1)
                else:
                    distances, indices = faiss_index.search(query, 1)
                best_idx = int(indices[0][0])
                best_score = float(distances[0][0])

                if best_score > match_threshold and best_idx < len(labels):
                    name = str(labels[best_idx])

                    # Store match result in detection
                    detection["label"] = name
                    detection["score"] = best_score

                    state["last_label"] = name
                    state["last_score"] = best_score
                    state["last_status"] = "매칭됨"
                    state.setdefault("item_scores", {})[name] = best_score

                    # Check if we should count this product
                    last_seen_at = state.setdefault("last_seen_at", {})
                    can_count = should_count_product(
                        last_seen_at,
                        name,
                        cooldown_seconds=cooldown_seconds,
                    )
                    if can_count:
                        billing_items = state.setdefault("billing_items", {})
                        billing_items[name] = int(billing_items.get(name, 0)) + 1
                else:
                    state["last_label"] = "미매칭"
                    state["last_score"] = best_score
                    state["last_status"] = "매칭 실패"

    else:
        # Fallback: Background subtraction (original logic)
        state["detection_boxes"] = []  # No YOLO detections

        if frame_profiler is not None:
            with frame_profiler.measure("detect"):
                fg_mask = bg_subtractor.apply(frame)
                fg_mask = cv2.erode(fg_mask, None, iterations=2)
                fg_mask = cv2.dilate(fg_mask, None, iterations=4)
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.erode(fg_mask, None, iterations=2)
            fg_mask = cv2.dilate(fg_mask, None, iterations=4)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        if roi_poly is not None:
            roi_mask = np.zeros_like(fg_mask)
            cv2.fillPoly(roi_mask, [roi_poly], 255)
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if candidates and faiss_index is not None and faiss_index.ntotal > 0:
            state["last_status"] = "탐지됨"
            main_cnt = max(candidates, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_cnt)

            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + 2 * pad)
            y2 = min(frame.shape[0], y + h + 2 * pad)

            w = x2 - x
            h = y2 - y

            if w > 20 and h > 20:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                entry_event = False
                inside_roi = False
                if roi_poly is not None:
                    cx = x + (w / 2)
                    cy = y + (h / 2)
                    inside_roi = cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0

                    if inside_roi:
                        state["roi_empty_frames"] = 0
                        entry_event = not bool(state.get("roi_occupied", False))
                        state["roi_occupied"] = True
                    else:
                        state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1

                if roi_poly is not None and int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                    state["roi_occupied"] = False

                crop = frame[y:y + h, x:x + w]

                if roi_poly is not None and roi_entry_mode:
                    periodic_slot = frame_count % max(1, detect_every_n_frames) == 0
                    allow_inference = inside_roi and (entry_event or periodic_slot)
                    if inside_roi:
                        state["last_status"] = "ROI 진입" if entry_event else "ROI 내부"
                    else:
                        state["last_status"] = "ROI 외부"
                else:
                    allow_inference = frame_count % max(1, detect_every_n_frames) == 0

                if allow_inference:
                    if frame_profiler is not None:
                        with frame_profiler.measure("embed"):
                            emb = build_query_embedding(crop, model_bundle)
                    else:
                        emb = build_query_embedding(crop, model_bundle)
                    query = np.expand_dims(emb, axis=0)

                    if frame_profiler is not None:
                        with frame_profiler.measure("faiss"):
                            distances, indices = faiss_index.search(query, 1)
                    else:
                        distances, indices = faiss_index.search(query, 1)
                    best_idx = int(indices[0][0])
                    best_score = float(distances[0][0])

                    if best_score > match_threshold and best_idx < len(labels):
                        name = str(labels[best_idx])
                        label = f"{name} ({best_score:.3f})"

                        state["last_label"] = name
                        state["last_score"] = best_score
                        state["last_status"] = "매칭됨"
                        state.setdefault("item_scores", {})[name] = best_score

                        cv2.putText(
                            display_frame,
                            label,
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        last_seen_at = state.setdefault("last_seen_at", {})
                        can_count = should_count_product(
                            last_seen_at,
                            name,
                            cooldown_seconds=cooldown_seconds,
                        )
                        if can_count:
                            billing_items = state.setdefault("billing_items", {})
                            billing_items[name] = int(billing_items.get(name, 0)) + 1
                    else:
                        state["last_label"] = "미매칭"
                        state["last_score"] = best_score
                        state["last_status"] = "매칭 실패"
        else:
            if roi_poly is not None and bool(state.get("roi_occupied", False)):
                state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1
                if int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                    state["roi_occupied"] = False

            state["last_label"] = "-"
            state["last_score"] = 0.0
            state["last_status"] = "미탐지"

    # Draw ROI polygon
    if roi_poly is not None:
        cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

    if frame_profiler is not None:
        frame_profiler.add_ms("total", (time.perf_counter() - total_start) * 1000.0)
        frame_profiler.finish()

    return display_frame
