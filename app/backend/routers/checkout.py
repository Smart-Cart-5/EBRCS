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
from backend.services.session_manager import CheckoutSession
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


def _process_frame_sync(
    session: CheckoutSession,
    frame: np.ndarray,
    frame_id: int | None,
    session_id: str | None,
    model_bundle: dict,
    faiss_index: Any,
    labels: np.ndarray,
    yolo_detector: Any = None,
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
    roi_polygon_normalized = session.roi_polygon if hasattr(session, 'roi_polygon') else None

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
    )

    state_snapshot = {
        "billing_items": dict(session.state["billing_items"]),
        "item_scores": {k: round(v, 4) for k, v in session.state["item_scores"].items()},
        "last_label": session.state["last_label"],
        "last_score": round(session.state["last_score"], 4),
        "last_status": session.state["last_status"],
        "total_count": sum(session.state["billing_items"].values()),
        "roi_polygon": roi_polygon_normalized,  # Normalized coordinates for frontend
        "detection_boxes": session.state.get("detection_boxes", []),  # YOLO detections
        "topk_candidates": session.state.get("topk_candidates", []),
        "confidence": round(float(session.state.get("confidence", 0.0)), 4),
        "best_pair": session.state.get("best_pair"),
        "event_state": session.state.get("event_state"),
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
        # Similarity*100 scale for UI convenience (not model accuracy percentage).
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
    }

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
    frame_queue_maxsize = max(1, int(config.CHECKOUT_QUEUE_MAXSIZE))
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
        frame_times = []
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

                # Log performance every 30 frames
                frame_time = (time.time() - start_time) * 1000
                frame_times.append(frame_time)
                ws_frame_profiler.finish()
                if len(frame_times) >= 30:
                    avg_time = sum(frame_times) / len(frame_times)
                    fps = 1000 / avg_time if avg_time > 0 else 0
                    logger.info(
                        "Checkout performance: avg=%.1fms, fps=%.1f, dropped=%d",
                        avg_time, fps, dropped_frames_total
                    )
                    frame_times = []
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
