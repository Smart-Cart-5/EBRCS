"""Purchase history endpoints."""

from datetime import datetime, timedelta
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import bindparam, func, text
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
    picture: str | None = None


class DiscountProduct(BaseModel):
    item_no: str
    product_name: str
    discount_rate: float
    discount_amount: int
    picture: str | None = None


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


def _aggregate_product_counts(db: Session) -> dict[str, int]:
    """Aggregate product counts from purchase_history.items JSON."""
    product_counts: dict[str, int] = {}
    all_purchases = db.query(models.PurchaseHistory.items).all()

    for purchase in all_purchases:
        items = purchase.items if isinstance(purchase.items, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            try:
                count = int(item.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            product_counts[name] = product_counts.get(name, 0) + count

    return product_counts


def _extract_item_no_from_label(label: str) -> str | None:
    text_label = (label or "").strip()
    if not text_label:
        return None
    if text_label.isdigit():
        return text_label
    if "_" not in text_label:
        return None
    prefix, _ = text_label.split("_", 1)
    prefix = prefix.strip()
    return prefix if prefix.isdigit() else None


def _latest_picture_by_item_no(db: Session, item_nos: list[str]) -> dict[str, str]:
    if not item_nos:
        return {}

    stmt = text(
        """
        SELECT p.item_no, p.picture
        FROM products p
        JOIN (
            SELECT item_no, MAX(id) AS max_id
            FROM products
            WHERE item_no IN :item_nos
            GROUP BY item_no
        ) latest
            ON latest.item_no = p.item_no
           AND latest.max_id = p.id
        """
    ).bindparams(bindparam("item_nos", expanding=True))

    rows = db.execute(stmt, {"item_nos": item_nos}).mappings().all()
    picture_map: dict[str, str] = {}
    for row in rows:
        item_no = str(row.get("item_no") or "").strip()
        picture = str(row.get("picture") or "").strip()
        if item_no and picture:
            picture_map[item_no] = picture
    return picture_map


def _top_popular_products(db: Session, product_counts: dict[str, int], limit: int = 5) -> list[dict]:
    safe_limit = max(1, min(limit, 20))
    top_items = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:safe_limit]

    unique_item_nos: list[str] = []
    for name, _ in top_items:
        item_no = _extract_item_no_from_label(name)
        if item_no and item_no not in unique_item_nos:
            unique_item_nos.append(item_no)

    picture_by_item_no = _latest_picture_by_item_no(db, unique_item_nos)

    result: list[dict] = []
    for name, count in top_items:
        item_no = _extract_item_no_from_label(name)
        result.append(
            {
                "name": name,
                "total_count": count,
                "picture": picture_by_item_no.get(item_no) if item_no else None,
            }
        )
    return result


def _discount_categories(db: Session) -> list[str]:
    rows = db.execute(
        text(
            """
            SELECT DISTINCT category_l
            FROM products
            WHERE category_l IS NOT NULL
              AND TRIM(category_l) <> ''
            ORDER BY category_l
            """
        )
    ).mappings().all()
    return [str(row["category_l"]) for row in rows if row.get("category_l")]


def _top_discount_products_by_category(
    db: Session, category_l: str, limit: int = 5
) -> list[dict]:
    safe_limit = max(1, min(limit, 20))
    category = (category_l or "").strip()
    if not category:
        return []

    rows = db.execute(
        text(
            """
            WITH ranked AS (
                SELECT
                    p.item_no,
                    p.product_name,
                    p.picture,
                    pd.is_discounted,
                    pd.discount_rate,
                    pd.discount_amount,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.item_no
                        ORDER BY
                            pp.checked_at DESC,
                            pd.updated_at DESC,
                            pd.id DESC
                    ) AS rn_latest
                FROM products p
                JOIN product_prices pp ON pp.product_id = p.id
                JOIN product_discounts pd ON pd.product_price_id = pp.id
                WHERE p.category_l = :category_l
            )
            SELECT item_no, product_name, picture, discount_rate, discount_amount
            FROM ranked
            WHERE rn_latest = 1
              AND is_discounted = 1
              AND discount_rate IS NOT NULL
            ORDER BY discount_rate DESC, COALESCE(discount_amount, 0) DESC, product_name ASC
            LIMIT :limit
            """
        ),
        {"category_l": category, "limit": safe_limit},
    ).mappings().all()

    result: list[dict] = []
    for row in rows:
        picture = str(row.get("picture") or "").strip() or None
        result.append(
            {
                "item_no": str(row.get("item_no") or ""),
                "product_name": str(row.get("product_name") or ""),
                "discount_rate": float(row.get("discount_rate") or 0.0),
                "discount_amount": int(row.get("discount_amount") or 0),
                "picture": picture,
            }
        )
    return result


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


@router.get("/popular", response_model=List[PopularProduct])
def get_popular_products(
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    limit: int = 5,
):
    """Get popular products for authenticated users (used by user home popup)."""
    _ = current_user  # Auth required; role does not matter.
    product_counts = _aggregate_product_counts(db)
    return _top_popular_products(db, product_counts, limit=limit)


@router.get("/discount-categories", response_model=List[str])
def get_discount_categories(
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """Get available top-level product categories for discount filtering."""
    _ = current_user  # Auth required; role does not matter.
    return _discount_categories(db)


@router.get("/discounts", response_model=List[DiscountProduct])
def get_discount_products(
    current_user: Annotated[models.User, Depends(get_current_user)],
    category_l: str,
    db: Session = Depends(get_db),
    limit: int = 5,
):
    """Get top discounted products for a selected category."""
    _ = current_user  # Auth required; role does not matter.
    return _top_discount_products_by_category(db, category_l=category_l, limit=limit)


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
    product_counts = _aggregate_product_counts(db)
    total_products_sold = sum(product_counts.values())
    popular_products = _top_popular_products(db, product_counts, limit=5)

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
