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
    from backend.association import associate_hands_products
    from backend.utils.profiler import ProfileCollector
except Exception:  # pragma: no cover - optional in non-backend contexts
    backend_config = None
    associate_hands_products = None
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


def _reset_candidate_votes(state: MutableMapping[str, Any]) -> None:
    state["candidate_votes"] = {}
    state["candidate_history"] = []
    state["topk_candidates"] = []
    state["confidence"] = 0.0


def _update_candidate_votes(
    state: MutableMapping[str, Any],
    current_scores: dict[str, float],
    vote_window_size: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    history = state.setdefault("candidate_history", [])
    history.append(current_scores)
    max_len = max(1, int(vote_window_size))
    if len(history) > max_len:
        del history[:-max_len]

    aggregated: dict[str, float] = {}
    for sample in history:
        for label, score in sample.items():
            aggregated[label] = aggregated.get(label, 0.0) + float(score)

    state["candidate_votes"] = aggregated
    return history, aggregated


def _build_topk_response(aggregated: dict[str, float], top_k: int) -> list[dict[str, float]]:
    ranked = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_k))]
    return [{"label": label, "score": float(score)} for label, score in ranked]


def _compute_confidence(topk_candidates: list[dict[str, float]]) -> float:
    if not topk_candidates:
        return 0.0
    top1 = float(topk_candidates[0]["score"])
    top2 = float(topk_candidates[1]["score"]) if len(topk_candidates) > 1 else 0.0
    if top2 <= 1e-6:
        return top1
    return top1 / top2


def _match_with_voting(
    *,
    crop: np.ndarray,
    model_bundle,
    faiss_index,
    labels,
    state: MutableMapping[str, Any],
    match_threshold: float,
    faiss_top_k: int,
    vote_window_size: int,
    vote_min_samples: int,
    frame_profiler,
) -> tuple[str | None, float, list[dict[str, float]], float]:
    if frame_profiler is not None:
        with frame_profiler.measure("embed"):
            emb = build_query_embedding(crop, model_bundle)
    else:
        emb = build_query_embedding(crop, model_bundle)

    query = np.expand_dims(emb, axis=0)
    search_k = max(1, min(int(faiss_top_k), int(faiss_index.ntotal)))
    if frame_profiler is not None:
        with frame_profiler.measure("faiss"):
            distances, indices = faiss_index.search(query, search_k)
    else:
        distances, indices = faiss_index.search(query, search_k)

    current_scores: dict[str, float] = {}
    for idx, score in zip(indices[0], distances[0]):
        label_idx = int(idx)
        if label_idx < 0 or label_idx >= len(labels):
            continue
        name = str(labels[label_idx])
        current_scores[name] = max(current_scores.get(name, -1e9), float(score))

    history, aggregated = _update_candidate_votes(state, current_scores, vote_window_size)
    topk_candidates = _build_topk_response(aggregated, faiss_top_k)
    confidence = _compute_confidence(topk_candidates)
    state["topk_candidates"] = topk_candidates
    state["confidence"] = confidence

    if not topk_candidates:
        return None, 0.0, topk_candidates, confidence

    best_label = str(topk_candidates[0]["label"])
    best_agg_score = float(topk_candidates[0]["score"])
    avg_score = best_agg_score / max(1, len(history))
    enough_samples = len(history) >= max(1, int(vote_min_samples))
    if enough_samples and avg_score >= match_threshold:
        return best_label, avg_score, topk_candidates, confidence
    return None, avg_score, topk_candidates, confidence


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
    min_product_bbox_area: int = 2500,
    max_products_per_frame: int = 3,
    faiss_top_k: int = 3,
    vote_window_size: int = 5,
    vote_min_samples: int = 3,
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
    if "best_pair" not in state:
        state["best_pair"] = None

    # Choose detection method: YOLO or background subtraction
    if yolo_detector is not None:
        # YOLO-based detection
        if frame_profiler is not None:
            with frame_profiler.measure("detect"):
                detections = yolo_detector.detect(frame)
        else:
            detections = yolo_detector.detect(frame)
        state["detection_boxes"] = detections  # Store all detections (product + hand)
        hands = [d for d in detections if d.get("class") == "hand"]
        all_products = [d for d in detections if d.get("class") == "product"]

        if associate_hands_products is not None:
            assoc_matches = associate_hands_products(
                hands,
                all_products,
                iou_weight=getattr(backend_config, "ASSOCIATION_IOU_WEIGHT", 0.5),
                dist_weight=getattr(backend_config, "ASSOCIATION_DIST_WEIGHT", 0.5),
                max_center_dist=getattr(backend_config, "ASSOCIATION_MAX_CENTER_DIST", 0.35),
                min_score=getattr(backend_config, "ASSOCIATION_MIN_SCORE", 0.1),
            )
        else:
            assoc_matches = []

        if assoc_matches:
            state["best_pair"] = assoc_matches[0]
            hb = assoc_matches[0]["hand_box"]
            pb = assoc_matches[0]["product_box"]
            h, w = frame.shape[:2]
            hcx = int(((hb[0] + hb[2]) * 0.5) * w)
            hcy = int(((hb[1] + hb[3]) * 0.5) * h)
            pcx = int(((pb[0] + pb[2]) * 0.5) * w)
            pcy = int(((pb[1] + pb[3]) * 0.5) * h)
            cv2.line(display_frame, (hcx, hcy), (pcx, pcy), (255, 180, 0), 2)
        else:
            state["best_pair"] = None

        # Filter only product detections and keep largest K for stable throughput.
        frame_h, frame_w = frame.shape[:2]
        product_detections = list(all_products)
        if product_detections:
            product_detections = sorted(
                product_detections,
                key=lambda d: max(0.0, (d["box"][2] - d["box"][0]) * frame_w)
                * max(0.0, (d["box"][3] - d["box"][1]) * frame_h),
                reverse=True,
            )[: max(1, int(max_products_per_frame))]
        else:
            _reset_candidate_votes(state)

        # Process each detected product
        for detection in product_detections:
            box = detection["box"]

            bbox_w_px = max(0.0, (box[2] - box[0]) * frame_w)
            bbox_h_px = max(0.0, (box[3] - box[1]) * frame_h)
            bbox_area_px = bbox_w_px * bbox_h_px
            if bbox_area_px < max(1, int(min_product_bbox_area)):
                continue

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
                name, best_score, topk_candidates, confidence = _match_with_voting(
                    crop=crop,
                    model_bundle=model_bundle,
                    faiss_index=faiss_index,
                    labels=labels,
                    state=state,
                    match_threshold=match_threshold,
                    faiss_top_k=faiss_top_k,
                    vote_window_size=vote_window_size,
                    vote_min_samples=vote_min_samples,
                    frame_profiler=frame_profiler,
                )
                detection["topk_candidates"] = topk_candidates
                detection["confidence"] = confidence

                if name is not None:

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
        state["best_pair"] = None

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
                    name, best_score, topk_candidates, _ = _match_with_voting(
                        crop=crop,
                        model_bundle=model_bundle,
                        faiss_index=faiss_index,
                        labels=labels,
                        state=state,
                        match_threshold=match_threshold,
                        faiss_top_k=faiss_top_k,
                        vote_window_size=vote_window_size,
                        vote_min_samples=vote_min_samples,
                        frame_profiler=frame_profiler,
                    )
                    if name is not None:
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
                        state["topk_candidates"] = topk_candidates
        else:
            _reset_candidate_votes(state)
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
