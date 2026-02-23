"""Purchase history endpoints."""

from datetime import datetime, timedelta
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import models
from backend.database import get_db
from backend.routers.auth import get_current_user
from backend.services.pricing import quote_named_items

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


class DailyStat(BaseModel):
    date: str
    purchase_count: int
    revenue: int


class DashboardStats(BaseModel):
    total_purchases: int
    total_customers: int
    today_purchases: int
    total_products_sold: int
    popular_products: List[PopularProduct]
    recent_purchases: List[PurchaseResponse]
    daily_stats: List[DailyStat]
    total_revenue: int
    average_order_value: float
    today_revenue: int


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
    normalized_items = [
        {"name": item.name.strip(), "count": int(item.count)}
        for item in purchase_data.items
        if item.name.strip() and item.count > 0
    ]

    if not normalized_items:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one valid item is required",
        )

    quote = quote_named_items(db, normalized_items)

    new_purchase = models.PurchaseHistory(
        user_id=current_user.id,
        items=quote["items"],
        total_amount=quote["total_amount"],
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
    period_days: int = 7,
):
    """Get dashboard statistics (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    period_days = max(1, min(period_days, 365))
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=period_days - 1)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    # Total purchases
    total_purchases = db.query(models.PurchaseHistory).count()

    # Total customers (users with role 'user')
    total_customers = db.query(models.User).filter(models.User.role == "user").count()

    # Today's purchases
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_purchases = (
        db.query(models.PurchaseHistory)
        .filter(models.PurchaseHistory.timestamp >= today_start)
        .count()
    )

    total_revenue = (
        db.query(func.coalesce(func.sum(models.PurchaseHistory.total_amount), 0)).scalar()
        or 0
    )
    today_revenue = (
        db.query(func.coalesce(func.sum(models.PurchaseHistory.total_amount), 0))
        .filter(models.PurchaseHistory.timestamp >= today_start)
        .scalar()
        or 0
    )
    average_order_value = (
        float(total_revenue) / float(total_purchases) if total_purchases > 0 else 0.0
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

    # Daily stats (period_days)
    daily_map = {}
    cursor = start_date
    while cursor <= end_date:
        daily_map[cursor] = {"purchase_count": 0, "revenue": 0}
        cursor += timedelta(days=1)

    range_purchases = (
        db.query(models.PurchaseHistory)
        .filter(models.PurchaseHistory.timestamp >= start_dt)
        .filter(models.PurchaseHistory.timestamp < end_dt)
        .all()
    )

    for purchase in range_purchases:
        day = purchase.timestamp.date()
        if day in daily_map:
            daily_map[day]["purchase_count"] += 1
            daily_map[day]["revenue"] += int(purchase.total_amount or 0)

    daily_stats = [
        {
            "date": day.isoformat(),
            "purchase_count": daily_map[day]["purchase_count"],
            "revenue": daily_map[day]["revenue"],
        }
        for day in sorted(daily_map.keys())
    ]

    return {
        "total_purchases": total_purchases,
        "total_customers": total_customers,
        "today_purchases": today_purchases,
        "total_products_sold": total_products_sold,
        "popular_products": popular_products,
        "recent_purchases": recent_purchases,
        "daily_stats": daily_stats,
        "total_revenue": int(total_revenue),
        "average_order_value": average_order_value,
        "today_revenue": int(today_revenue),
    }
