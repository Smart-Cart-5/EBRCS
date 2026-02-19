"""Real-time checkout endpoints: WebSocket live camera + video upload with SSE."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from backend import config
from backend.dependencies import app_state
from backend.services.session_manager import CheckoutSession

logger = logging.getLogger("backend.checkout")

router = APIRouter(tags=["checkout"])


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
    model_bundle: dict,
    faiss_index: Any,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one frame through the checkout pipeline (sync, for thread pool execution)."""
    from checkout_core.frame_processor import process_checkout_frame

    frame = _resize_frame(frame, config.STREAM_TARGET_WIDTH)
    session.frame_count += 1

    # ROI polygon coordinates in normalized [0, 1] space for frontend rendering
    roi_polygon_normalized = session.roi_poly_norm

    # Fast path: skip heavy processing on non-inference frames
    should_infer = session.frame_count % max(1, config.DETECT_EVERY_N_FRAMES) == 0

    if not should_infer:
        display_frame = frame.copy()
        roi_poly = session.get_roi_polygon(frame.shape)
        if roi_poly is not None:
            cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

        state_snapshot = {
            "billing_items": dict(session.state["billing_items"]),
            "item_scores": {k: round(v, 4) for k, v in session.state["item_scores"].items()},
            "last_label": session.state["last_label"],
            "last_score": round(session.state["last_score"], 4),
            "last_status": "display_only",
            "last_direction": session.state.get("last_direction", "-"),
            "total_count": sum(session.state["billing_items"].values()),
            "roi_polygon": roi_polygon_normalized,
        }
        return display_frame, state_snapshot

    roi_poly = session.get_roi_polygon(frame.shape)

    display_frame = process_checkout_frame(
        frame=frame,
        frame_count=session.frame_count,
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
        use_deepsort=config.USE_DEEPSORT,
        track_n_init=config.TRACK_N_INIT,
        track_max_age=config.TRACK_MAX_AGE,
        track_max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
        track_stale_frames=config.TRACK_STALE_FRAMES,
        direction_gate_y_norm=config.DIRECTION_GATE_Y_NORM,
        direction_min_delta_px=config.DIRECTION_MIN_DELTA_PX,
        direction_event_cooldown_sec=config.DIRECTION_EVENT_COOLDOWN_SECONDS,
        reclassify_every_n_frames=config.RECLASSIFY_EVERY_N_FRAMES,
        reclassify_area_gain=config.RECLASSIFY_AREA_GAIN,
        direction_event_band_px=config.DIRECTION_EVENT_BAND_PX,
        deepsort_embedder_mode=config.DEEPSORT_EMBEDDER_MODE,
        deepsort_max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
        deepsort_gating_only_position=config.DEEPSORT_GATING_ONLY_POSITION,
        deepsort_bbox_pad_ratio=config.DEEPSORT_BBOX_PAD_RATIO,
        deepsort_tsu_tolerance=config.DEEPSORT_TSU_TOLERANCE,
        deepsort_simple_hs_bins=config.DEEPSORT_SIMPLE_HS_BINS,
        pipeline_debug=config.PIPELINE_DEBUG,
        pipeline_log_every_n_infer=config.PIPELINE_LOG_EVERY_N_INFER,
    )

    state_snapshot = {
        "billing_items": dict(session.state["billing_items"]),
        "item_scores": {k: round(v, 4) for k, v in session.state["item_scores"].items()},
        "last_label": session.state["last_label"],
        "last_score": round(session.state["last_score"], 4),
        "last_status": session.state["last_status"],
        "last_direction": session.state.get("last_direction", "-"),
        "total_count": sum(session.state["billing_items"].values()),
        "roi_polygon": roi_polygon_normalized,
    }

    return display_frame, state_snapshot


async def _process_frame(session: CheckoutSession, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Async wrapper: acquires reader lock and delegates to sync processing."""
    loop = asyncio.get_event_loop()

    # Acquire reader lock: allows concurrent reads, blocks if writer is active
    async with app_state.index_rwlock.reader_lock:
        model_bundle = app_state.model_bundle
        faiss_index = app_state.faiss_index
        labels = app_state.labels

        return await loop.run_in_executor(
            None, _process_frame_sync, session, frame, model_bundle, faiss_index, labels
        )


# ---------------------------------------------------------------------------
# WebSocket: live camera checkout
# ---------------------------------------------------------------------------


@router.websocket("/ws/checkout/{session_id}")
async def checkout_ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time camera checkout."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    # Latest-frame-only queue for minimum latency
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)

    async def receive_loop():
        try:
            while True:
                data = await websocket.receive_bytes()
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(data)
        except WebSocketDisconnect:
            await frame_queue.put(b"")

    async def process_loop():
        frame_times = []
        perf_frames = 0
        perf_bytes = 0
        perf_decode_ms = 0.0
        perf_process_ms = 0.0
        perf_send_ms = 0.0
        perf_last_log = time.monotonic()
        try:
            while True:
                start_time = time.time()

                data = None
                dropped_frames = 0
                while True:
                    try:
                        data = frame_queue.get_nowait()
                        if not frame_queue.empty():
                            dropped_frames += 1
                    except asyncio.QueueEmpty:
                        if data is None:
                            data = await frame_queue.get()
                        break

                if data == b"":
                    break

                perf_frames += 1
                perf_bytes += len(data)

                decode_start = time.perf_counter()
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                perf_decode_ms += decode_ms
                if frame is None:
                    logger.warning(
                        "WS decode failed: session=%s bytes=%d",
                        session_id,
                        len(data),
                    )
                    continue

                process_start = time.perf_counter()
                display_frame, state_snapshot = await _process_frame(session, frame)
                process_ms = (time.perf_counter() - process_start) * 1000.0
                perf_process_ms += process_ms

                send_start = time.perf_counter()
                if config.STREAM_SEND_IMAGES:
                    _, jpeg_buf = cv2.imencode(
                        ".jpg",
                        display_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 75],
                    )
                    frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
                    response = {"frame": frame_b64, **state_snapshot}
                else:
                    response = state_snapshot

                await websocket.send_text(json.dumps(response))
                send_ms = (time.perf_counter() - send_start) * 1000.0
                perf_send_ms += send_ms

                if config.PIPELINE_DEBUG:
                    now = time.monotonic()
                    if now - perf_last_log >= 1.0:
                        n = max(1, perf_frames)
                        interval = max(1e-6, now - perf_last_log)
                        logger.info(
                            "WS debug: session=%s fps_in=%.1f recv=%d bytes=%d "
                            "decode=%.1fms process=%.1fms send=%.1fms state=%s label=%s total=%s",
                            session_id,
                            perf_frames / interval,
                            perf_frames,
                            perf_bytes,
                            perf_decode_ms / n,
                            perf_process_ms / n,
                            perf_send_ms / n,
                            state_snapshot.get("last_status"),
                            state_snapshot.get("last_label"),
                            state_snapshot.get("total_count"),
                        )
                        perf_frames = 0
                        perf_bytes = 0
                        perf_decode_ms = 0.0
                        perf_process_ms = 0.0
                        perf_send_ms = 0.0
                        perf_last_log = now

                # Log performance every 30 frames
                frame_time = (time.time() - start_time) * 1000
                frame_times.append(frame_time)
                if len(frame_times) >= 30:
                    avg_time = sum(frame_times) / len(frame_times)
                    fps = 1000 / avg_time if avg_time > 0 else 0
                    logger.info(
                        "Checkout performance: avg=%.1fms, fps=%.1f, dropped=%d",
                        avg_time,
                        fps,
                        dropped_frames,
                    )
                    frame_times = []
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Error in checkout WebSocket process loop")

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

_video_tasks: dict[str, dict[str, Any]] = {}


@router.post("/sessions/{session_id}/video-upload")
async def upload_video(session_id: str, file: UploadFile):
    """Upload a video file for offline inference and return a task id."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

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
    asyncio.create_task(_process_video_background(session, temp_path, task_id))

    return {"task_id": task_id}


async def _process_video_background(session: CheckoutSession, video_path: str, task_id: str):
    """Background task: process a video frame-by-frame and update task status."""
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
            await _process_frame(session, frame)

            task_status["current_frame"] = frame_idx
            task_status["progress"] = round(frame_idx / max(total_frames, 1), 4)

            if frame_idx % 5 == 0:
                await asyncio.sleep(0)

        cap.release()
        task_status["done"] = True
        task_status["progress"] = 1.0
    except Exception as exc:
        logger.exception("Video processing error: task=%s", task_id)
        task_status["error"] = str(exc)
        task_status["done"] = True
    finally:
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
            yield f"data: {json.dumps(payload)}\n\n"

            if status["done"]:
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
