"""Purchase history endpoints."""

from datetime import datetime, timedelta
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend import models
from backend.database import get_db
from backend.routers.auth import get_current_user

router = APIRouter(prefix="/purchases", tags=["purchases"])


# Pydantic models
class PurchaseItem(BaseModel):
    name: str
    count: int


class PurchaseCreate(BaseModel):
    session_id: str
    items: List[PurchaseItem]
    notes: str | None = None


class PurchaseResponse(BaseModel):
    id: int
    user_id: int
    username: str
    items: List[dict]
    total_amount: int
    timestamp: str
    notes: str | None

    class Config:
        from_attributes = True


class PopularProduct(BaseModel):
    name: str
    total_count: int


class DashboardStats(BaseModel):
    total_purchases: int
    total_customers: int
    today_purchases: int
    total_products_sold: int
    popular_products: List[PopularProduct]
    recent_purchases: List[PurchaseResponse]


@router.get("/my", response_model=List[PurchaseResponse])
def get_my_purchases(
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get current user's purchase history."""
    purchases = (
        db.query(models.PurchaseHistory)
        .filter(models.PurchaseHistory.user_id == current_user.id)
        .order_by(models.PurchaseHistory.timestamp.desc())
        .all()
    )

    return [
        {
            "id": p.id,
            "user_id": p.user_id,
            "username": current_user.username,
            "items": p.items,
            "total_amount": p.total_amount,
            "timestamp": p.timestamp.isoformat(),
            "notes": p.notes,
        }
        for p in purchases
    ]


@router.get("/all", response_model=List[PurchaseResponse])
def get_all_purchases(
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get all purchases (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    purchases = (
        db.query(models.PurchaseHistory)
        .join(models.User)
        .order_by(models.PurchaseHistory.timestamp.desc())
        .all()
    )

    return [
        {
            "id": p.id,
            "user_id": p.user_id,
            "username": p.user.username,
            "items": p.items,
            "total_amount": p.total_amount,
            "timestamp": p.timestamp.isoformat(),
            "notes": p.notes,
        }
        for p in purchases
    ]


@router.post("", response_model=PurchaseResponse)
def create_purchase(
    purchase_data: PurchaseCreate,
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Create a new purchase record."""
    # Convert items to dict format for JSON storage
    items_list = [{"name": item.name, "count": item.count} for item in purchase_data.items]

    new_purchase = models.PurchaseHistory(
        user_id=current_user.id,
        items=items_list,
        total_amount=0,  # TODO: Calculate based on product prices
        notes=purchase_data.notes,
    )

    db.add(new_purchase)
    db.commit()
    db.refresh(new_purchase)

    return {
        "id": new_purchase.id,
        "user_id": new_purchase.user_id,
        "username": current_user.username,
        "items": new_purchase.items,
        "total_amount": new_purchase.total_amount,
        "timestamp": new_purchase.timestamp.isoformat(),
        "notes": new_purchase.notes,
    }


@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard_stats(
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get dashboard statistics (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    # Total purchases
    total_purchases = db.query(models.PurchaseHistory).count()

    # Total customers (users with role 'user')
    total_customers = db.query(models.User).filter(models.User.role == "user").count()

    # Today's purchases
    today = datetime.utcnow().date()
    today_purchases = (
        db.query(models.PurchaseHistory)
        .filter(models.PurchaseHistory.timestamp >= today)
        .count()
    )

    # Total products sold and popular products
    all_purchases = db.query(models.PurchaseHistory).all()

    product_counts = {}
    total_products_sold = 0

    for purchase in all_purchases:
        for item in purchase.items:
            name = item["name"]
            count = item["count"]
            product_counts[name] = product_counts.get(name, 0) + count
            total_products_sold += count

    # Popular products TOP 5
    popular_products = [
        {"name": name, "total_count": count}
        for name, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Recent purchases (last 5)
    recent_purchases_db = (
        db.query(models.PurchaseHistory)
        .join(models.User)
        .order_by(models.PurchaseHistory.timestamp.desc())
        .limit(5)
        .all()
    )

    recent_purchases = [
        {
            "id": p.id,
            "user_id": p.user_id,
            "username": p.user.username,
            "items": p.items,
            "total_amount": p.total_amount,
            "timestamp": p.timestamp.isoformat(),
            "notes": p.notes,
        }
        for p in recent_purchases_db
    ]

    return {
        "total_purchases": total_purchases,
        "total_customers": total_customers,
        "today_purchases": today_purchases,
        "total_products_sold": total_products_sold,
        "popular_products": popular_products,
        "recent_purchases": recent_purchases,
    }
