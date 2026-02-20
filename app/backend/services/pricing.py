"""Pricing helpers for resolving recognized labels to catalog prices."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

_ITEM_NO_PREFIX_PATTERN = re.compile(r"^(\d{2,})(?:[_\s-](.+))?$")


def _extract_item_no_and_name(label: str) -> tuple[str | None, str | None]:
    name = (label or "").strip()
    if not name:
        return None, None

    # Most recognized labels follow "<item_no>_<product_name>" format.
    if "_" in name:
        prefix, suffix = name.split("_", 1)
        if prefix.isdigit():
            return prefix, suffix.strip() or None

    match = _ITEM_NO_PREFIX_PATTERN.match(name)
    if match:
        return match.group(1), (match.group(2) or "").strip() or None

    return None, name


def _find_product(db: Session, label: str) -> tuple[dict[str, Any] | None, str | None]:
    item_no, display_name = _extract_item_no_and_name(label)

    if item_no:
        by_item_no = db.execute(
            text(
                """
                SELECT id, item_no, barcd, product_name
                FROM products
                WHERE item_no = :item_no
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"item_no": item_no},
        ).mappings().first()
        if by_item_no:
            return dict(by_item_no), "item_no"

    if display_name:
        by_name = db.execute(
            text(
                """
                SELECT id, item_no, barcd, product_name
                FROM products
                WHERE product_name = :product_name
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"product_name": display_name},
        ).mappings().first()
        if by_name:
            return dict(by_name), "product_name"

    if label and label != display_name:
        by_label = db.execute(
            text(
                """
                SELECT id, item_no, barcd, product_name
                FROM products
                WHERE product_name = :product_name
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"product_name": label},
        ).mappings().first()
        if by_label:
            return dict(by_label), "raw_label"

    return None, None


def _find_latest_price(db: Session, product_id: int) -> dict[str, Any] | None:
    price_row = db.execute(
        text(
            """
            SELECT id, price, currency, source, checked_at
            FROM product_prices
            WHERE product_id = :product_id
            ORDER BY checked_at DESC, id DESC
            LIMIT 1
            """
        ),
        {"product_id": product_id},
    ).mappings().first()

    if not price_row:
        return None

    return dict(price_row)


def quote_named_items(db: Session, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve cart items to the latest catalog price and aggregate totals."""
    quoted_items: list[dict[str, Any]] = []
    item_unit_prices: dict[str, int | None] = {}
    item_line_totals: dict[str, int] = {}
    unpriced_items: list[str] = []
    total_amount = 0
    currency = "KRW"

    catalog_available = True
    try:
        db.execute(text("SELECT 1 FROM products LIMIT 1"))
        db.execute(text("SELECT 1 FROM product_prices LIMIT 1"))
    except SQLAlchemyError:
        catalog_available = False

    for raw in items:
        label = str(raw.get("name", "")).strip()
        try:
            count = int(raw.get("count", 0) or 0)
        except (TypeError, ValueError):
            count = 0

        if not label or count <= 0:
            continue

        product: dict[str, Any] | None = None
        matched_by: str | None = None

        if catalog_available:
            try:
                product, matched_by = _find_product(db, label)
            except SQLAlchemyError:
                product, matched_by = None, None

        unit_price: int | None = None
        line_total = 0
        source: str | None = None
        checked_at: str | None = None

        if product:
            try:
                latest_price = _find_latest_price(db, int(product["id"]))
            except SQLAlchemyError:
                latest_price = None

            if latest_price:
                unit_price = int(latest_price["price"])
                line_total = unit_price * count
                currency = str(latest_price.get("currency") or currency)
                source = latest_price.get("source")
                checked_value = latest_price.get("checked_at")
                if isinstance(checked_value, datetime):
                    checked_at = checked_value.isoformat()
                elif checked_value is not None:
                    checked_at = str(checked_value)

        if unit_price is None:
            unpriced_items.append(label)

        total_amount += line_total
        item_unit_prices[label] = unit_price
        item_line_totals[label] = line_total

        quoted_items.append(
            {
                "name": label,
                "count": count,
                "product_id": int(product["id"]) if product else None,
                "item_no": str(product["item_no"]) if product else None,
                "product_name": str(product["product_name"]) if product else label,
                "match_strategy": matched_by,
                "unit_price": unit_price,
                "line_total": line_total,
                "currency": currency,
                "price_found": unit_price is not None,
                "price_source": source,
                "price_checked_at": checked_at,
            }
        )

    return {
        "items": quoted_items,
        "total_amount": total_amount,
        "currency": currency,
        "item_unit_prices": item_unit_prices,
        "item_line_totals": item_line_totals,
        "unpriced_items": unpriced_items,
    }
