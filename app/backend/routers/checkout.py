"""Real-time checkout endpoints: WebSocket live camera + video upload with SSE."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import tempfile
import time
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

from backend import config
from backend.dependencies import app_state
from backend.roi_warp import warp_frame
from backend.services.session_manager import (
    CheckoutSession,
    PHASE_CHECKOUT_RUNNING,
    PHASE_ROI_CALIBRATING,
)
from backend.utils.profiler import FrameProfiler, ProfileCollector

logger = logging.getLogger("backend.checkout")

router = APIRouter(tags=["checkout"])


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _resize_frame(frame: np.ndarray, target_width: int = 960) -> np.ndarray:
    """Resize frame to target width maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _mask_to_polygon_norm(mask01: np.ndarray) -> list[list[float]] | None:
    if mask01 is None or mask01.ndim != 2:
        return None
    h, w = mask01.shape[:2]
    if h <= 1 or w <= 1:
        return None
    bin255 = (mask01.astype(np.uint8) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 8.0:
        return None
    eps = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        pts = largest.reshape(-1, 2)
    poly: list[list[float]] = []
    for x, y in pts:
        xn = float(np.clip(float(x) / float(max(1, w - 1)), 0.0, 1.0))
        yn = float(np.clip(float(y) / float(max(1, h - 1)), 0.0, 1.0))
        poly.append([xn, yn])
    return poly if len(poly) >= 3 else None


def _build_state_snapshot(
    *,
    session: CheckoutSession,
    session_id: str | None,
    roi_polygon_normalized: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase = str(session.state.get("phase", ""))
    pending_ready = isinstance(session.state.get("_cart_roi_mask_pending"), np.ndarray)
    message = None
    user_message = session.state.get("_checkout_user_message")
    if isinstance(user_message, str) and user_message.strip():
        message = user_message.strip()
    if phase == PHASE_ROI_CALIBRATING:
        message = message or "ROI preview ready. Press OK to confirm or Retry."
    confirm_enabled = bool(phase == PHASE_ROI_CALIBRATING and pending_ready)
    retry_enabled = bool(phase == PHASE_ROI_CALIBRATING)
    auto_mode = session.state.get("_cart_roi_auto_enabled")
    checkout_start_mode = session.state.get("_checkout_start_mode")
    state_snapshot = {
        "type": "checkout_state",
        "session_id": session_id,
        "billing_items": dict(session.state["billing_items"]),
        "item_scores": {k: round(v, 4) for k, v in session.state["item_scores"].items()},
        "last_label": session.state["last_label"],
        "last_score": round(session.state["last_score"], 4),
        "last_status": session.state["last_status"],
        "total_count": sum(session.state["billing_items"].values()),
        "roi_polygon": roi_polygon_normalized,
        "detection_boxes": session.state.get("detection_boxes", []),
        "topk_candidates": session.state.get("topk_candidates", []),
        "confidence": round(float(session.state.get("confidence", 0.0)), 4),
        "best_pair": session.state.get("best_pair"),
        "event_state": session.state.get("event_state"),
        "occluded_by_hand": bool(session.state.get("occluded_by_hand", False)),
        "overlap_hand_iou_max": round(float(session.state.get("overlap_hand_iou_max", 0.0)), 4),
        "search_crop_min_side_before": session.state.get("search_crop_min_side_before"),
        "search_crop_min_side_after": session.state.get("search_crop_min_side_after"),
        "search_crop_padding_ratio": round(float(session.state.get("search_crop_padding_ratio", 0.0)), 4),
        "ocr_used": bool(session.state.get("ocr_used", False)),
        "ocr_attempted": bool(session.state.get("ocr_attempted", False)),
        "ocr_error": session.state.get("ocr_error"),
        "ocr_skip_reason": session.state.get("ocr_skip_reason"),
        "ocr_text": str(session.state.get("ocr_text", "")),
        "ocr_matched_keywords": session.state.get("ocr_matched_keywords", {}),
        "ocr_text_score": round(float(session.state.get("ocr_text_score", 0.0)), 4),
        "ocr_ms": round(float(session.state.get("ocr_ms", 0.0)), 2),
        "ocr_ambiguous": bool(session.state.get("ocr_ambiguous", False)),
        "ocr_chosen_slice": session.state.get("ocr_chosen_slice"),
        "ocr_chosen_psm": session.state.get("ocr_chosen_psm"),
        "ocr_chosen_thresh": session.state.get("ocr_chosen_thresh"),
        "ocr_chosen_conf_cut": session.state.get("ocr_chosen_conf_cut"),
        "ocr_chosen_lang": session.state.get("ocr_chosen_lang"),
        "ocr_token_count": int(session.state.get("ocr_token_count", 0)),
        "ocr_korean_char_count": int(session.state.get("ocr_korean_char_count", 0)),
        "ocr_avg_conf": round(float(session.state.get("ocr_avg_conf", 0.0)), 3),
        "ocr_reranked_topk": session.state.get("ocr_reranked_topk", []),
        "did_search": bool(session.state.get("did_search", False)),
        "skip_reason": str(session.state.get("skip_reason", "unknown")),
        "result_label": session.state.get("result_label", session.state.get("last_label", "-")),
        "is_unknown": bool(session.state.get("is_unknown", False)),
        "match_score_raw": (
            round(float(session.state["match_score_raw"]), 6)
            if session.state.get("match_score_raw") is not None
            else None
        ),
        "match_top2_raw": (
            round(float(session.state["match_top2_raw"]), 6)
            if session.state.get("match_top2_raw") is not None
            else None
        ),
        "match_score_percent": (
            round(float(session.state["match_score_percent"]), 3)
            if session.state.get("match_score_percent") is not None
            else None
        ),
        "match_gap": (
            round(float(session.state["match_gap"]), 6)
            if session.state.get("match_gap") is not None
            else None
        ),
        "match_gap_reason": session.state.get("match_gap_reason"),
        "gap_reason": session.state.get("match_gap_reason"),
        "unknown_reason": session.state.get("unknown_reason"),
        "last_result_name": session.state.get("last_result_name"),
        "last_result_score": (
            round(float(session.state["last_result_score"]), 4)
            if session.state.get("last_result_score") is not None
            else None
        ),
        "last_result_topk": session.state.get("last_result_topk", []),
        "last_result_confidence": (
            round(float(session.state["last_result_confidence"]), 4)
            if session.state.get("last_result_confidence") is not None
            else None
        ),
        "last_result_age_ms": (
            max(0, int(time.time() * 1000) - int(session.state["last_result_at_ms"]))
            if session.state.get("last_result_at_ms") is not None
            else None
        ),
        "warp_enabled": bool(getattr(session, "warp_enabled", False)),
        "warp_points": getattr(session, "warp_points_norm", None),
        "phase": phase,
        "message": message,
        "cart_roi_confirmed": bool(session.state.get("_cart_roi_confirmed", False)),
        "cart_roi_preview_ready": pending_ready,
        "cart_roi_pending_polygon": session.state.get("_cart_roi_pending_polygon"),
        "cart_roi_pending_ratio": round(float(session.state.get("_cart_roi_pending_ratio", 0.0)), 4),
        "confirm_enabled": confirm_enabled,
        "retry_enabled": retry_enabled,
        "cart_roi_polygon_confirmed": roi_polygon_normalized,
        "cart_roi_auto_enabled": auto_mode,
        "checkout_start_mode": checkout_start_mode,
        "cart_roi_available": bool(getattr(app_state, "cart_roi_available", False)),
        "cart_roi_unavailable_reason": getattr(app_state, "cart_roi_unavailable_reason", None),
        "last_roi_error": session.state.get("_cart_roi_last_error"),
        "cart_roi_invalid_reason": session.state.get("_cart_roi_invalid_reason"),
    }
    if extra:
        state_snapshot.update(extra)
    return state_snapshot


def _process_frame_sync(
    session: CheckoutSession,
    frame: np.ndarray,
    frame_id: int | None,
    session_id: str | None,
    model_bundle: dict,
    faiss_index: Any,
    labels: np.ndarray,
    yolo_detector: Any = None,
    cart_roi_segmenter: Any = None,
    ws_profiler: FrameProfiler | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one frame through the checkout pipeline (sync, for thread pool execution).

    IMPORTANT: This function expects FAISS index snapshot to be passed in,
    ensuring consistency during the entire frame processing.

    Args:
        yolo_detector: Optional YOLO detector for object detection.
    """
    from checkout_core.frame_processor import process_checkout_frame

    original_frame = frame
    warp_active = bool(getattr(session, "warp_enabled", False) and getattr(session, "warp_points_norm", None))
    warp_applied = False
    if warp_active:
        try:
            if ws_profiler is not None:
                with ws_profiler.measure("warp"):
                    frame = warp_frame(frame, session.warp_points_norm, session.warp_size)
            else:
                frame = warp_frame(frame, session.warp_points_norm, session.warp_size)
            warp_applied = True
            # Safety: fallback to original frame if warped output looks almost black.
            max_px = int(np.max(frame)) if frame.size else 0
            mean_px = float(np.mean(frame)) if frame.size else 0.0
            if max_px < int(config.WARP_BLACK_MAX_THRESHOLD) or mean_px < float(config.WARP_BLACK_MEAN_THRESHOLD):
                logger.warning(
                    "Warp output too dark; fallback to original frame. session=%s frame_id=%s max=%d mean=%.2f",
                    session_id,
                    frame_id,
                    max_px,
                    mean_px,
                )
                frame = original_frame
                warp_applied = False
        except Exception:
            logger.exception("Failed to apply warp; fallback to original frame")
            # Degrade session-level warp mode on hard warp failure.
            session.warp_enabled = False
            warp_applied = False

    if ws_profiler is not None:
        with ws_profiler.measure("resize"):
            frame = _resize_frame(frame, config.STREAM_TARGET_WIDTH)
    else:
        frame = _resize_frame(frame, config.STREAM_TARGET_WIDTH)
    session.frame_count += 1

    # Get ROI polygon coordinates (normalized 0-1)
    roi_polygon_normalized = getattr(session, "roi_poly_norm", None)

    auto_mode = session.state.get("_cart_roi_auto_enabled")
    if isinstance(auto_mode, bool):
        requires_calibration = bool(auto_mode and bool(getattr(app_state, "cart_roi_available", False)))
    else:
        requires_calibration = bool(getattr(app_state, "cart_roi_available", False))
    prev_phase = str(session.state.get("phase", ""))
    phase = session.start_checkout(require_roi_calibration=requires_calibration)
    if phase != prev_phase:
        logger.info(
            "Session phase changed: session=%s frame_id=%s phase=%s confirmed=%s",
            session_id,
            frame_id,
            phase,
            bool(session.state.get("_cart_roi_confirmed", False)),
        )
    if phase == PHASE_ROI_CALIBRATING:
        pending_mask = None
        segmenter_error: str | None = None
        if cart_roi_segmenter is not None:
            try:
                if ws_profiler is not None:
                    with ws_profiler.measure("cart_roi_calibration"):
                        pending_mask = cart_roi_segmenter.get_or_update_mask(
                            frame_bgr=frame,
                            frame_count=session.frame_count,
                            frame_id=frame_id,
                            session_id=session_id,
                            state=session.state,
                            mask_key="_cart_roi_mask_pending",
                            last_update_key="_cart_roi_pending_last_update_frame",
                        )
                else:
                    pending_mask = cart_roi_segmenter.get_or_update_mask(
                        frame_bgr=frame,
                        frame_count=session.frame_count,
                        frame_id=frame_id,
                        session_id=session_id,
                        state=session.state,
                            mask_key="_cart_roi_mask_pending",
                            last_update_key="_cart_roi_pending_last_update_frame",
                        )
            except Exception:
                segmenter_error = "segmenter_exception"
                session.state["_cart_roi_last_error"] = segmenter_error
                logger.exception("Cart ROI calibration update failed: session=%s frame_id=%s", session_id, frame_id)
        else:
            session.state["_cart_roi_last_error"] = "segmenter_unavailable"

        display_frame = frame.copy()
        pending_polygon = None
        ratio = 0.0
        preview_updated = False
        preview_skip_reason: str | None = None
        if isinstance(pending_mask, np.ndarray):
            pending_polygon = _mask_to_polygon_norm(pending_mask)
            ratio = float(np.count_nonzero(pending_mask)) / float(max(1, pending_mask.size))
            now_ms = int(time.time() * 1000)
            last_preview_ms = int(session.state.get("_cart_roi_preview_last_sent_ms", 0))
            # Limit preview payload updates to ~4 FPS in calibration mode.
            if now_ms - last_preview_ms >= 250:
                session.state["_cart_roi_preview_last_sent_ms"] = now_ms
                session.state["_cart_roi_pending_ratio"] = ratio
                session.state["_cart_roi_pending_polygon"] = pending_polygon
                preview_updated = True
            else:
                preview_skip_reason = "preview_rate_limited"
            current_poly = session.state.get("_cart_roi_pending_polygon")
            if isinstance(current_poly, list) and len(current_poly) >= 3:
                h, w = pending_mask.shape[:2]
                pts = np.array(
                    [[int(round(p[0] * max(1, w - 1))), int(round(p[1] * max(1, h - 1)))] for p in current_poly],
                    dtype=np.int32,
                )
                if len(pts) >= 3:
                    cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
            display_frame[pending_mask > 0] = (
                0.65 * display_frame[pending_mask > 0] + 0.35 * np.array([0, 255, 0])
            ).astype(np.uint8)
        else:
            session.state["_cart_roi_pending_polygon"] = None
            session.state["_cart_roi_pending_ratio"] = 0.0

        confirm_enabled = bool(isinstance(pending_mask, np.ndarray))
        invalid_reason: str | None = None
        if not isinstance(pending_mask, np.ndarray):
            invalid_reason = str(session.state.get("_cart_roi_last_error") or segmenter_error or "pending_mask_none")
        elif pending_polygon is None or len(pending_polygon) < 3:
            invalid_reason = "polygon_invalid"
        elif ratio <= 0.0:
            invalid_reason = "ratio_zero"
        session.state["_cart_roi_invalid_reason"] = invalid_reason

        now_ms = int(time.time() * 1000)
        last_calib_log_ms = int(session.state.get("_cart_roi_calib_log_last_ms", 0))
        if now_ms - last_calib_log_ms >= 1000:
            session.state["_cart_roi_calib_log_last_ms"] = now_ms
            seg_meta = session.state.get("_cart_roi_last_segmenter")
            if not isinstance(seg_meta, dict):
                seg_meta = {}
            seg_status = str(seg_meta.get("status", "none"))
            seg_skip_reason = str(seg_meta.get("skip_reason", "none"))
            logger.info(
                (
                    "ROI_CALIBRATING debug: session=%s phase=%s frame_count=%s frame_id=%s "
                    "segmenter=%s skip_reason=%s rf_ok=%s rf_latency_ms=%s "
                    "preview_updated=%s preview_skip_reason=%s "
                    "mask_shape=%s frame_shape=%s class_has_target=%s matched_class=%s class_map_values=%s target_id=%s "
                    "ratio=%.4f polygon_points=%s confirm_enabled=%s invalid_reason=%s last_roi_error=%s"
                ),
                session_id,
                phase,
                int(session.frame_count),
                frame_id,
                seg_status,
                seg_skip_reason,
                bool(seg_meta.get("roboflow_response_ok", False)),
                seg_meta.get("roboflow_latency_ms"),
                preview_updated,
                preview_skip_reason,
                seg_meta.get("decoded_mask_shape"),
                [int(frame.shape[0]), int(frame.shape[1])],
                bool(seg_meta.get("class_map_has_target", False)),
                seg_meta.get("matched_class_name"),
                seg_meta.get("class_map_values_sample"),
                seg_meta.get("target_id"),
                float(ratio),
                (len(pending_polygon) if isinstance(pending_polygon, list) else 0),
                confirm_enabled,
                invalid_reason,
                session.state.get("_cart_roi_last_error"),
            )

        session.state["skip_reason"] = "roi_calibrating"
        state_snapshot = _build_state_snapshot(
            session=session,
            session_id=session_id,
            roi_polygon_normalized=roi_polygon_normalized,
            extra={
                "phase": PHASE_ROI_CALIBRATING,
                "cart_roi_preview_ready": bool(isinstance(pending_mask, np.ndarray)),
                "message": "ROI preview ready. Press OK to confirm or Retry.",
            },
        )
        return display_frame, state_snapshot

    if phase != PHASE_CHECKOUT_RUNNING:
        session.force_checkout_running()

    # ROI polygon is computed on the same frame that YOLO sees:
    # - warp_applied=True: ROI is in warped-frame coordinates
    # - warp_applied=False: ROI is in original-frame coordinates
    roi_poly = session.get_roi_polygon(frame.shape)

    display_frame = process_checkout_frame(
        frame=frame,
        frame_count=session.frame_count,
        frame_id=frame_id,
        session_id=session_id,
        bg_subtractor=session.bg_subtractor,
        model_bundle=model_bundle,
        faiss_index=faiss_index,
        labels=labels,
        state=session.state,
        min_area=config.MIN_AREA,
        detect_every_n_frames=config.DETECT_EVERY_N_FRAMES,
        match_threshold=config.MATCH_THRESHOLD,
        cooldown_seconds=config.COUNT_COOLDOWN_SECONDS,
        roi_poly=roi_poly,
        roi_clear_frames=config.ROI_CLEAR_FRAMES,
        roi_entry_mode=roi_poly is not None,
        min_product_bbox_area=config.MIN_PRODUCT_BBOX_AREA,
        max_products_per_frame=config.MAX_PRODUCTS_PER_FRAME,
        faiss_top_k=config.FAISS_TOP_K,
        vote_window_size=config.VOTE_WINDOW_SIZE,
        vote_min_samples=config.VOTE_MIN_SAMPLES,
        search_every_n_frames=config.SEARCH_EVERY_N_FRAMES,
        min_box_area_ratio=config.MIN_BOX_AREA_RATIO,
        stable_frames_for_search=config.STABLE_FRAMES_FOR_SEARCH,
        search_cooldown_ms=config.SEARCH_COOLDOWN_MS,
        roi_box_min_overlap=config.ROI_BOX_MIN_OVERLAP,
        product_conf_min=config.PRODUCT_CONF_MIN,
        product_min_area_ratio=config.PRODUCT_MIN_AREA_RATIO,
        product_aspect_ratio_min=config.PRODUCT_ASPECT_RATIO_MIN,
        product_aspect_ratio_max=config.PRODUCT_ASPECT_RATIO_MAX,
        product_max_height_ratio=config.PRODUCT_MAX_HEIGHT_RATIO,
        product_max_width_ratio=config.PRODUCT_MAX_WIDTH_RATIO,
        product_edge_touch_eps=config.PRODUCT_EDGE_TOUCH_EPS,
        min_search_area_ratio=config.MIN_SEARCH_AREA_RATIO,
        min_crop_size=config.MIN_CROP_SIZE,
        search_select_w_conf=config.SEARCH_SELECT_W_CONF,
        search_select_w_area=config.SEARCH_SELECT_W_AREA,
        search_select_w_roi=config.SEARCH_SELECT_W_ROI,
        roi_iou_min=config.ROI_IOU_MIN,
        roi_center_pass=config.ROI_CENTER_PASS,
        hand_conf_min=config.HAND_CONF_MIN,
        hand_overlap_iou=config.HAND_OVERLAP_IOU,
        search_mask_hand=config.SEARCH_MASK_HAND,
        warp_on=warp_applied,
        frame_profiler_override=ws_profiler,
        yolo_detector=yolo_detector,
        cart_roi_segmenter=cart_roi_segmenter,
    )

    state_snapshot = _build_state_snapshot(
        session=session,
        session_id=session_id,
        roi_polygon_normalized=roi_polygon_normalized,
        extra={"phase": PHASE_CHECKOUT_RUNNING},
    )
    return display_frame, state_snapshot


async def _process_frame(
    session: CheckoutSession,
    frame: np.ndarray,
    frame_id: int | None = None,
    session_id: str | None = None,
    ws_profiler: FrameProfiler | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Async wrapper: acquires reader lock and delegates to sync processing.

    Uses RWLock reader lock to allow multiple concurrent inference requests
    while blocking during product updates (writer lock).
    """
    loop = asyncio.get_event_loop()

    # Acquire reader lock: allows concurrent reads, blocks if writer is active
    async with app_state.index_rwlock.reader_lock:
        # Snapshot shared state under lock for consistency
        model_bundle = app_state.model_bundle
        faiss_index = app_state.faiss_index
        labels = app_state.labels
        yolo_detector = app_state.yolo_detector
        cart_roi_segmenter = app_state.cart_roi_segmenter

        # Run CPU/GPU-intensive work in thread pool
        return await loop.run_in_executor(
            None,
            _process_frame_sync,
            session,
            frame,
            frame_id,
            session_id,
            model_bundle,
            faiss_index,
            labels,
            yolo_detector,
            cart_roi_segmenter,
            ws_profiler,
        )


# ---------------------------------------------------------------------------
# WebSocket: live camera checkout
# ---------------------------------------------------------------------------


@router.websocket("/ws/checkout/{session_id}")
async def checkout_ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time camera checkout.

    Protocol:
    - Client sends: binary JPEG frame data
    - Server responds: JSON { frame: base64_jpeg, ...state }
    """
    session = app_state.session_manager.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    # Latest-frame-only queue for minimum latency
    # Small queue + drop old frames = always process most recent frame
    # Hard latest-frame-only behavior for low latency.
    frame_queue_maxsize = 1
    frame_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=frame_queue_maxsize)
    dropped_frames_total = 0
    recv_frame_seq = 0

    async def receive_loop():
        nonlocal dropped_frames_total, recv_frame_seq
        try:
            while True:
                data = await websocket.receive_bytes()
                recv_frame_seq += 1
                item = (recv_frame_seq, data)
                try:
                    frame_queue.put_nowait(item)
                except asyncio.QueueFull:
                    try:
                        frame_queue.get_nowait()
                        dropped_frames_total += 1
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        frame_queue.put_nowait(item)
                    except asyncio.QueueFull:
                        dropped_frames_total += 1
        except (WebSocketDisconnect, RuntimeError):
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait((-1, b""))  # Sentinel to stop processing
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def process_loop():
        nonlocal dropped_frames_total
        interval_started = time.time()
        interval_frames = 0
        interval_dropped_start = dropped_frames_total
        perf_acc = {
            "yolo_ms": 0.0,
            "embed_ms": 0.0,
            "faiss_ms": 0.0,
            "ocr_ms": 0.0,
            "ws_ms": 0.0,
            "total_ms": 0.0,
        }
        ws_profiler_collector = ProfileCollector(
            kind="ws",
            enable=config.ENABLE_PROFILING,
            every_n_frames=config.PROFILE_EVERY_N_FRAMES,
            logger=logger,
        )
        try:
            while True:
                start_time = time.time()
                ws_frame_profiler = ws_profiler_collector.start_frame()

                # Always get the latest frame, discard old ones for minimum latency
                frame_id, data = await frame_queue.get()  # Wait for first frame
                while not frame_queue.empty():
                    try:
                        frame_id, data = frame_queue.get_nowait()
                        dropped_frames_total += 1
                    except asyncio.QueueEmpty:
                        break

                if data == b"":
                    break  # Disconnected

                # Decode JPEG
                with ws_frame_profiler.measure("decode_bytes"):
                    np_arr = np.frombuffer(data, np.uint8)
                with ws_frame_profiler.measure("imdecode"):
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    ws_frame_profiler.finish()
                    continue

                # Run inference with reader lock
                display_frame, state_snapshot = await _process_frame(
                    session,
                    frame,
                    frame_id=frame_id,
                    session_id=session_id,
                    ws_profiler=ws_frame_profiler,
                )

                # Conditionally encode and send image based on config
                if config.STREAM_SEND_IMAGES:
                    _, jpeg_buf = cv2.imencode(
                        ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
                    response = {"frame": frame_b64, **state_snapshot}
                else:
                    # JSON-only mode: no image, just state and ROI
                    response = state_snapshot

                if websocket.client_state != WebSocketState.CONNECTED:
                    ws_frame_profiler.finish()
                    break

                try:
                    with ws_frame_profiler.measure("serialize_send"):
                        sanitized_response = _sanitize_json_value(response)
                        payload = json.dumps(sanitized_response, allow_nan=False)
                        await websocket.send_text(payload)
                except (WebSocketDisconnect, RuntimeError):
                    logger.info("WebSocket send aborted: session=%s", session_id)
                    ws_frame_profiler.finish()
                    break
                except (TypeError, ValueError):
                    logger.exception("Failed to serialize WS payload: session=%s frame_id=%s", session_id, frame_id)
                    ws_frame_profiler.finish()
                    continue

                # 1-second performance summary for fast bottleneck diagnosis.
                frame_time = (time.time() - start_time) * 1000
                breakdown = dict(getattr(ws_frame_profiler, "breakdown_ms", {}) or {})
                interval_frames += 1
                perf_acc["yolo_ms"] += float(breakdown.get("yolo_infer", 0.0))
                perf_acc["embed_ms"] += float(breakdown.get("embed", 0.0))
                perf_acc["faiss_ms"] += float(breakdown.get("faiss", 0.0))
                perf_acc["ocr_ms"] += float(breakdown.get("ocr", state_snapshot.get("ocr_ms", 0.0)))
                perf_acc["ws_ms"] += float(breakdown.get("serialize_send", 0.0))
                perf_acc["total_ms"] += float(frame_time)
                ws_frame_profiler.finish()
                now = time.time()
                elapsed = now - interval_started
                if elapsed >= 1.0:
                    avg_total = perf_acc["total_ms"] / max(1, interval_frames)
                    fps = interval_frames / max(elapsed, 1e-6)
                    dropped_interval = int(dropped_frames_total - interval_dropped_start)
                    queue_depth = int(frame_queue.qsize())
                    logger.info(
                        (
                            "Checkout performance(%.1fs): session=%s fps=%.1f avg_ms=%.1f "
                            "yolo_ms=%.1f embed_ms=%.1f faiss_ms=%.1f ocr_ms=%.1f ws_ms=%.1f total_ms=%.1f "
                            "dropped=%d dropped_interval=%d queue_depth=%d ocr_attempted=%s"
                        ),
                        elapsed,
                        session_id,
                        fps,
                        avg_total,
                        perf_acc["yolo_ms"] / max(1, interval_frames),
                        perf_acc["embed_ms"] / max(1, interval_frames),
                        perf_acc["faiss_ms"] / max(1, interval_frames),
                        perf_acc["ocr_ms"] / max(1, interval_frames),
                        perf_acc["ws_ms"] / max(1, interval_frames),
                        perf_acc["total_ms"] / max(1, interval_frames),
                        int(dropped_frames_total),
                        dropped_interval,
                        queue_depth,
                        bool(state_snapshot.get("ocr_attempted", False)),
                    )
                    interval_started = now
                    interval_frames = 0
                    interval_dropped_start = dropped_frames_total
                    for key in perf_acc:
                        perf_acc[key] = 0.0
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Error in checkout WebSocket process loop")

    # Run receive and process concurrently
    receive_task = asyncio.create_task(receive_loop())
    process_task = asyncio.create_task(process_loop())

    try:
        await asyncio.gather(receive_task, process_task)
    except Exception:
        pass
    finally:
        receive_task.cancel()
        process_task.cancel()
        logger.info("WebSocket disconnected: session=%s", session_id)


# ---------------------------------------------------------------------------
# Video upload + SSE progress
# ---------------------------------------------------------------------------

# In-memory task status storage
_video_tasks: dict[str, dict[str, Any]] = {}


@router.post("/sessions/{session_id}/video-upload")
async def upload_video(session_id: str, file: UploadFile):
    """Upload a video file for offline inference. Returns a task_id for SSE tracking."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="checkout_video_")
    try:
        content = await file.read()
        os.write(fd, content)
    finally:
        os.close(fd)

    task_id = str(uuid.uuid4())
    _video_tasks[task_id] = {
        "done": False,
        "progress": 0.0,
        "total_frames": 0,
        "current_frame": 0,
        "error": None,
    }

    session.video_task_id = task_id

    # Launch background processing
    asyncio.create_task(_process_video_background(session, temp_path, task_id))

    return {"task_id": task_id}


async def _process_video_background(
    session: CheckoutSession, video_path: str, task_id: str
):
    """Background task: process video frame-by-frame and update task status."""
    loop = asyncio.get_event_loop()
    task_status = _video_tasks[task_id]
    session.force_checkout_running()

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            task_status["error"] = "Cannot open video file"
            task_status["done"] = True
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        task_status["total_frames"] = total_frames

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run inference with reader lock
            await _process_frame(session, frame)

            task_status["current_frame"] = frame_idx
            task_status["progress"] = round(frame_idx / max(total_frames, 1), 4)

            # Yield control to event loop periodically
            if frame_idx % 5 == 0:
                await asyncio.sleep(0)

        cap.release()
        task_status["done"] = True
        task_status["progress"] = 1.0

    except Exception as e:
        logger.exception("Video processing error: task=%s", task_id)
        task_status["error"] = str(e)
        task_status["done"] = True
    finally:
        # Cleanup temp file
        try:
            os.unlink(video_path)
        except OSError:
            pass


@router.get("/sessions/{session_id}/video-status")
async def video_status_sse(session_id: str, task_id: str):
    """SSE endpoint streaming video inference progress."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if task_id not in _video_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        while True:
            status = _video_tasks.get(task_id)
            if status is None:
                break

            payload = {
                **status,
                "billing_items": dict(session.state["billing_items"]),
                "total_count": sum(session.state["billing_items"].values()),
            }
            yield f"data: {json.dumps(_sanitize_json_value(payload), allow_nan=False)}\n\n"

            if status["done"]:
                # Cleanup task status after final send
                _video_tasks.pop(task_id, None)
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
