"""Billing (cart) endpoints -- read / update / confirm."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.dependencies import app_state
from backend.database import get_db
from backend.services.pricing import quote_named_items

router = APIRouter(tags=["billing"])


class BillingState(BaseModel):
    billing_items: dict[str, int]
    item_scores: dict[str, float]
    total_count: int
    item_unit_prices: dict[str, int | None] = Field(default_factory=dict)
    item_line_totals: dict[str, int] = Field(default_factory=dict)
    total_amount: int = 0
    currency: str = "KRW"
    unpriced_items: list[str] = Field(default_factory=list)


class BillingUpdateRequest(BaseModel):
    billing_items: dict[str, int]


def _build_billing_state(
    billing_items: dict[str, int], item_scores: dict[str, float], db: Session
) -> BillingState:
    quote = quote_named_items(
        db,
        [{"name": name, "count": count} for name, count in billing_items.items()],
    )

    return BillingState(
        billing_items=billing_items,
        item_scores=item_scores,
        total_count=sum(billing_items.values()),
        item_unit_prices=quote["item_unit_prices"],
        item_line_totals=quote["item_line_totals"],
        total_amount=quote["total_amount"],
        currency=quote["currency"],
        unpriced_items=quote["unpriced_items"],
    )


@router.get("/sessions/{session_id}/billing", response_model=BillingState)
async def get_billing(session_id: str, db: Session = Depends(get_db)):
    """Get current billing state for a session."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    items = session.state["billing_items"]
    scores = session.state["item_scores"]
    return _build_billing_state(items, scores, db)


@router.put("/sessions/{session_id}/billing", response_model=BillingState)
async def update_billing(
    session_id: str,
    req: BillingUpdateRequest,
    db: Session = Depends(get_db),
):
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
    return _build_billing_state(items, scores, db)


@router.post("/sessions/{session_id}/billing/confirm")
async def confirm_billing(session_id: str, db: Session = Depends(get_db)):
    """Confirm billing and reset session state."""
    session = app_state.session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    confirmed_items = dict(session.state["billing_items"])
    quote = quote_named_items(
        db,
        [{"name": name, "count": count} for name, count in confirmed_items.items()],
    )
    confirmed_total = sum(confirmed_items.values())
    session.reset_billing()

    return {
        "status": "confirmed",
        "confirmed_items": confirmed_items,
        "confirmed_total": confirmed_total,
        "confirmed_total_amount": quote["total_amount"],
        "currency": quote["currency"],
        "unpriced_items": quote["unpriced_items"],
    }
