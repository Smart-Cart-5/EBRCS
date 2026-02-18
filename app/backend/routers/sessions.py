"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import config
from backend.dependencies import app_state
from backend.roi_warp import order_points_tl_tr_br_bl

router = APIRouter(tags=["sessions"])


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
