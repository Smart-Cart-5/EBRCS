"""Billing (cart) endpoints -- read / update / confirm."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.dependencies import app_state

router = APIRouter(tags=["billing"])


class BillingState(BaseModel):
    billing_items: dict[str, int]
    item_scores: dict[str, float]
    total_count: int


class BillingUpdateRequest(BaseModel):
    billing_items: dict[str, int]


@router.get("/sessions/{session_id}/billing", response_model=BillingState)
async def get_billing(session_id: str):
    """Get current billing state for a session."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    items = session.state["billing_items"]
    scores = session.state["item_scores"]
    return BillingState(
        billing_items=items,
        item_scores=scores,
        total_count=sum(items.values()),
    )


@router.put("/sessions/{session_id}/billing", response_model=BillingState)
async def update_billing(session_id: str, req: BillingUpdateRequest):
    """Update billing items (for quantity adjustments on validate page)."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove items with count <= 0
    session.state["billing_items"] = {
        k: v for k, v in req.billing_items.items() if v > 0
    }

    items = session.state["billing_items"]
    scores = session.state["item_scores"]
    return BillingState(
        billing_items=items,
        item_scores=scores,
        total_count=sum(items.values()),
    )


@router.post("/sessions/{session_id}/billing/confirm")
async def confirm_billing(session_id: str):
    """Confirm billing and reset session state."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    confirmed_items = dict(session.state["billing_items"])
    confirmed_total = sum(confirmed_items.values())
    session.reset_billing()

    return {
        "status": "confirmed",
        "confirmed_items": confirmed_items,
        "confirmed_total": confirmed_total,
    }
