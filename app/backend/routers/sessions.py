"""Session management endpoints."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
import numpy as np
from pydantic import BaseModel

from backend import config
from backend.dependencies import app_state
from backend.roi_warp import order_points_tl_tr_br_bl
from backend.services.session_manager import PHASE_ROI_CALIBRATING

router = APIRouter(tags=["sessions"])
logger = logging.getLogger("backend.sessions")


class SessionResponse(BaseModel):
    session_id: str


class ROIRequest(BaseModel):
    points: list[list[float]]  # [[x_norm, y_norm], ...] in [0, 1]


class ROIResponse(BaseModel):
    points: list[list[float]] | None
    num_vertices: int


class WarpRequest(BaseModel):
    points: list[list[float]]
    enabled: bool | None = None
    width: int | None = None
    height: int | None = None


class WarpEnabledRequest(BaseModel):
    enabled: bool


class WarpResponse(BaseModel):
    enabled: bool
    points: list[list[float]] | None
    size: list[int]


class ROICalibrationResponse(BaseModel):
    phase: str
    confirmed: bool
    has_pending_mask: bool


class SessionStateResponse(BaseModel):
    session_id: str
    phase: str
    cart_roi_confirmed: bool
    cart_roi_preview_ready: bool
    cart_roi_pending_polygon: list[list[float]] | None
    cart_roi_pending_ratio: float
    cart_roi_auto_enabled: bool | None
    checkout_start_mode: str | None
    cart_roi_available: bool
    cart_roi_unavailable_reason: str | None
    last_roi_error: str | None
    cart_roi_invalid_reason: str | None


class ROIModeRequest(BaseModel):
    enabled: bool


class ROIModeResponse(BaseModel):
    session_id: str
    cart_roi_auto_enabled: bool
    phase: str


class CheckoutStartRequest(BaseModel):
    mode: str  # "auto_roi" | "no_roi"


class CheckoutStartResponse(BaseModel):
    session_id: str
    requested_mode: str
    effective_mode: str
    phase: str
    message: str | None = None


@router.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new checkout session."""
    session = app_state.session_manager.create()
    session.warp_enabled = bool(config.WARP_MODE)
    session.warp_size = (int(config.WARP_WIDTH), int(config.WARP_HEIGHT))
    return SessionResponse(session_id=session.session_id)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a checkout session."""
    if not app_state.session_manager.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/roi", response_model=ROIResponse)
async def set_roi(session_id: str, req: ROIRequest):
    """Set ROI polygon for the session (normalized [0,1] coordinates)."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if len(req.points) < 3:
        raise HTTPException(status_code=422, detail="ROI needs at least 3 points")

    for pt in req.points:
        if len(pt) != 2:
            raise HTTPException(status_code=422, detail="Each point must be [x, y]")

    session.roi_poly_norm = req.points
    return ROIResponse(points=session.roi_poly_norm, num_vertices=len(req.points))


@router.delete("/sessions/{session_id}/roi")
async def clear_roi(session_id: str):
    """Clear ROI polygon."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.roi_poly_norm = None
    return {"status": "cleared"}


@router.post("/sessions/{session_id}/roi/confirm", response_model=ROICalibrationResponse)
async def confirm_roi(session_id: str):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    confirmed = session.confirm_pending_cart_roi()
    if not confirmed:
        raise HTTPException(status_code=422, detail="No pending ROI mask to confirm")
    logger.info(
        "ROI confirmed: session=%s phase=%s confirmed=%s",
        session_id,
        session.state.get("phase"),
        session.state.get("_cart_roi_confirmed"),
    )

    return ROICalibrationResponse(
        phase=str(session.state.get("phase", "")),
        confirmed=bool(session.state.get("_cart_roi_confirmed", False)),
        has_pending_mask=isinstance(session.state.get("_cart_roi_mask_pending"), np.ndarray),
    )


@router.post("/sessions/{session_id}/roi/retry", response_model=ROICalibrationResponse)
async def retry_roi(session_id: str):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.retry_cart_roi()
    logger.info("ROI retry requested: session=%s phase=%s", session_id, session.state.get("phase"))
    return ROICalibrationResponse(
        phase=PHASE_ROI_CALIBRATING,
        confirmed=bool(session.state.get("_cart_roi_confirmed", False)),
        has_pending_mask=False,
    )


@router.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(session_id: str):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateResponse(
        session_id=session_id,
        phase=str(session.state.get("phase", "")),
        cart_roi_confirmed=bool(session.state.get("_cart_roi_confirmed", False)),
        cart_roi_preview_ready=isinstance(session.state.get("_cart_roi_mask_pending"), np.ndarray),
        cart_roi_pending_polygon=session.state.get("_cart_roi_pending_polygon"),
        cart_roi_pending_ratio=float(session.state.get("_cart_roi_pending_ratio", 0.0)),
        cart_roi_auto_enabled=session.state.get("_cart_roi_auto_enabled"),
        checkout_start_mode=session.state.get("_checkout_start_mode"),
        cart_roi_available=bool(app_state.cart_roi_available),
        cart_roi_unavailable_reason=app_state.cart_roi_unavailable_reason,
        last_roi_error=session.state.get("_cart_roi_last_error"),
        cart_roi_invalid_reason=session.state.get("_cart_roi_invalid_reason"),
    )


@router.post("/sessions/{session_id}/roi/mode", response_model=ROIModeResponse)
async def set_roi_mode(session_id: str, req: ROIModeRequest):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.set_cart_roi_auto_enabled(bool(req.enabled))
    logger.info(
        "ROI auto mode set: session=%s enabled=%s",
        session_id,
        bool(req.enabled),
    )
    return ROIModeResponse(
        session_id=session_id,
        cart_roi_auto_enabled=bool(session.state.get("_cart_roi_auto_enabled")),
        phase=str(session.state.get("phase", "")),
    )


@router.post("/sessions/{session_id}/checkout/start", response_model=CheckoutStartResponse)
async def checkout_start(session_id: str, req: CheckoutStartRequest):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    requested_mode = "auto_roi" if str(req.mode) == "auto_roi" else "no_roi"
    effective_mode = requested_mode
    message = None
    if requested_mode == "auto_roi" and not bool(app_state.cart_roi_available):
        effective_mode = "no_roi"
        reason = str(app_state.cart_roi_unavailable_reason or "unavailable")
        if reason == "missing_api_key":
            message = "자동 ROI를 사용할 수 없어(ROBOFLOW_API_KEY 미설정) ROI 없이 시작합니다."
        elif reason == "server_disabled":
            message = "자동 ROI가 서버 설정(CART_ROI_ENABLED=0)으로 비활성화되어 ROI 없이 시작합니다."
        else:
            message = "자동 ROI를 사용할 수 없어 ROI 없이 시작합니다."
    phase = session.start_checkout_with_mode(effective_mode)
    session.state["_checkout_user_message"] = message
    logger.info(
        "Checkout start: session=%s requested_mode=%s effective_mode=%s phase=%s reason=%s",
        session_id,
        requested_mode,
        effective_mode,
        phase,
        app_state.cart_roi_unavailable_reason,
    )
    return CheckoutStartResponse(
        session_id=session_id,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        phase=str(phase),
        message=message,
    )


@router.post("/sessions/{session_id}/warp", response_model=WarpResponse)
async def set_warp(session_id: str, req: WarpRequest):
    """Set 4-point warp configuration for the session."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if len(req.points) != 4:
        raise HTTPException(status_code=422, detail="Warp needs exactly 4 points")
    for pt in req.points:
        if len(pt) != 2:
            raise HTTPException(status_code=422, detail="Each point must be [x, y]")
    ordered = order_points_tl_tr_br_bl(req.points)
    session.warp_points_norm = ordered
    if req.width and req.height and req.width > 0 and req.height > 0:
        session.warp_size = (int(req.width), int(req.height))
    else:
        session.warp_size = (int(config.WARP_WIDTH), int(config.WARP_HEIGHT))
    if req.enabled is not None:
        session.warp_enabled = bool(req.enabled)
    return WarpResponse(
        enabled=bool(session.warp_enabled),
        points=session.warp_points_norm,
        size=[int(session.warp_size[0]), int(session.warp_size[1])],
    )


@router.post("/sessions/{session_id}/warp/enabled", response_model=WarpResponse)
async def set_warp_enabled(session_id: str, req: WarpEnabledRequest):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.warp_enabled = bool(req.enabled)
    return WarpResponse(
        enabled=bool(session.warp_enabled),
        points=session.warp_points_norm,
        size=[int(session.warp_size[0]), int(session.warp_size[1])],
    )


@router.delete("/sessions/{session_id}/warp", response_model=WarpResponse)
async def clear_warp(session_id: str):
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.warp_enabled = False
    session.warp_points_norm = None
    session.warp_size = (int(config.WARP_WIDTH), int(config.WARP_HEIGHT))
    return WarpResponse(enabled=False, points=None, size=[int(session.warp_size[0]), int(session.warp_size[1])])
