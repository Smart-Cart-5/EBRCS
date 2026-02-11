"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.dependencies import app_state

router = APIRouter(tags=["sessions"])


class SessionResponse(BaseModel):
    session_id: str


class ROIRequest(BaseModel):
    points: list[list[float]]  # [[x_norm, y_norm], ...] in [0, 1]


class ROIResponse(BaseModel):
    points: list[list[float]] | None
    num_vertices: int


@router.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new checkout session."""
    session = app_state.session_manager.create()
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
