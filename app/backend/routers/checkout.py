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
    yolo_detector: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one frame through the checkout pipeline (sync, for thread pool execution).

    IMPORTANT: This function expects FAISS index snapshot to be passed in,
    ensuring consistency during the entire frame processing.

    Args:
        yolo_detector: Optional YOLO detector for object detection.
    """
    from checkout_core.frame_processor import process_checkout_frame

    frame = _resize_frame(frame, config.STREAM_TARGET_WIDTH)
    session.frame_count += 1

    # Get ROI polygon coordinates (normalized 0-1)
    roi_polygon_normalized = session.roi_polygon if hasattr(session, 'roi_polygon') else None

    # Fast path: skip heavy processing for non-inference frames
    should_infer = session.frame_count % max(1, config.DETECT_EVERY_N_FRAMES) == 0

    if not should_infer:
        # Return frame quickly without heavy processing
        display_frame = frame.copy()
        roi_poly = session.get_roi_polygon(frame.shape)
        if roi_poly is not None:
            cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

        state_snapshot = {
            "billing_items": dict(session.state["billing_items"]),
            "item_scores": {k: round(v, 4) for k, v in session.state["item_scores"].items()},
            "last_label": session.state["last_label"],
            "last_score": round(session.state["last_score"], 4),
            "last_status": "표시중",  # Non-inference frame
            "total_count": sum(session.state["billing_items"].values()),
            "roi_polygon": roi_polygon_normalized,  # Normalized coordinates for frontend
            "detection_boxes": session.state.get("detection_boxes", []),  # YOLO detections
        }
        return display_frame, state_snapshot

    # Full processing path for inference frames
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
    }

    return display_frame, state_snapshot


async def _process_frame(session: CheckoutSession, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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
            None, _process_frame_sync, session, frame, model_bundle, faiss_index, labels, yolo_detector
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
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)

    async def receive_loop():
        try:
            while True:
                data = await websocket.receive_bytes()
                # Drop old frame if queue is full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(data)
        except WebSocketDisconnect:
            await frame_queue.put(b"")  # Sentinel to stop processing

    async def process_loop():
        loop = asyncio.get_event_loop()
        frame_times = []
        try:
            while True:
                start_time = time.time()

                # Always get the latest frame, discard old ones for minimum latency
                data = None
                dropped_frames = 0
                while True:
                    try:
                        data = frame_queue.get_nowait()
                        if not frame_queue.empty():
                            dropped_frames += 1
                    except asyncio.QueueEmpty:
                        if data is None:
                            data = await frame_queue.get()  # Wait for first frame
                        break

                if data == b"":
                    break  # Disconnected

                # Decode JPEG
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Run inference with reader lock
                display_frame, state_snapshot = await _process_frame(session, frame)

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

                await websocket.send_text(json.dumps(response))

                # Log performance every 30 frames
                frame_time = (time.time() - start_time) * 1000
                frame_times.append(frame_time)
                if len(frame_times) >= 30:
                    avg_time = sum(frame_times) / len(frame_times)
                    fps = 1000 / avg_time if avg_time > 0 else 0
                    logger.info(
                        "Checkout performance: avg=%.1fms, fps=%.1f, dropped=%d",
                        avg_time, fps, dropped_frames
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
            yield f"data: {json.dumps(payload)}\n\n"

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
