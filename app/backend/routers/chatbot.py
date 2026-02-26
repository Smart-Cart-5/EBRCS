"""Chatbot router – LLM-powered shopping assistant using catalog DB tables.

Uses HuggingFace Router API (OpenAI-compatible) for natural-language answers.
Pulls cart data plus catalog metadata from products/product_prices/product_discounts.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import app_state

router = APIRouter(tags=["chatbot"])
# ---------------------------------------------------------------------------
# Cart action helpers
# ---------------------------------------------------------------------------

_ADD_TOKENS = ("추가", "담아", "더해", "넣어", "넣어줘", "담아줘")
_REMOVE_TOKENS = ("빼", "삭제", "제거", "빼줘", "빼 줘", "빼주세요")
_CLEAR_TOKENS = (
    "비워",
    "비워줘",
    "비워 줘",
    "초기화",
    "모두 삭제",
    "전부 삭제",
    "전체 삭제",
    "장바구니 비워",
    "장바구니 비워줘",
    "장바구니 초기화",
)
_SELECT_PREFIX = "__select__:"
_POLITE_TOKENS = ("줘", "주세요", "해줘", "해 주세요", "좀", "개", "개만", "개요", "개씩")
_REFERENCE_TOKENS = ("그거", "그것", "그 상품", "그 제품", "이거", "이것", "이 상품", "이 제품", "저거", "저것")


def _extract_quantity(question: str) -> tuple[int, str]:
    match = re.search(r"(\d+)\s*(개|개씩|개만|pcs|개요)", question)
    if match:
        qty = max(1, int(match.group(1)))
        cleaned = question.replace(match.group(0), " ")
        return qty, cleaned

    word_map = [
        ("한 개", 1),
        ("한개", 1),
        ("하나", 1),
        ("한", 1),
        ("두", 2),
        ("둘", 2),
        ("세", 3),
        ("셋", 3),
        ("네", 4),
        ("넷", 4),
    ]
    for key, value in word_map:
        if key in question:
            return value, question.replace(key, " ")

    return 1, question


def _normalize_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", text or "").lower()


def _label_display(label: str) -> str:
    if "_" in label:
        prefix, suffix = label.split("_", 1)
        if prefix.isdigit() and suffix:
            return suffix
    return label


def _find_label_in_cart(question: str, billing_items: dict[str, int]) -> str | None:
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return None

    for label in billing_items.keys():
        display = _label_display(label)
        normalized_display = _normalize_text(display)
        if normalized_display and (
            normalized_display in normalized_question or normalized_question in normalized_display
        ):
            return label

    return None


def _find_cart_label(
    db: Session,
    question: str,
    billing_items: dict[str, int],
    require_in_cart: bool = False,
) -> tuple[str | None, str | None]:
    normalized = re.sub(r"\s+", " ", question).strip()

    label = _find_label_in_cart(normalized, billing_items)
    if label:
        return label, _label_display(label)

    for label in billing_items.keys():
        if label and label in normalized:
            return label, label

    item_no_match = re.search(r"\b(\d{4,})\b", normalized)
    if item_no_match:
        item_no = item_no_match.group(1)
        row = db.execute(
            text("SELECT id, item_no, product_name FROM products WHERE item_no = :v LIMIT 1"),
            {"v": item_no},
        ).mappings().first()
        if row:
            for label in billing_items.keys():
                if label.startswith(item_no):
                    return label, row["product_name"]
            if require_in_cart:
                return None, None
            return f"{row['item_no']}_{row['product_name']}", row["product_name"]

    cleaned = _clean_keyword(normalized)
    if not cleaned:
        return None, None

    row = db.execute(
        text(
            "SELECT item_no, product_name FROM products "
            "WHERE product_name LIKE :kw ORDER BY LENGTH(product_name) DESC LIMIT 1"
        ),
        {"kw": f"%{cleaned}%"},
    ).mappings().first()
    if row:
        for label in billing_items.keys():
            if row["product_name"] in label:
                return label, row["product_name"]
            normalized_label = _normalize_text(label)
            normalized_name = _normalize_text(row["product_name"])
            if normalized_name and normalized_name in normalized_label:
                return label, row["product_name"]
        if require_in_cart:
            return None, None
        return f"{row['item_no']}_{row['product_name']}", row["product_name"]

    return None, None


def _find_by_item_no(db: Session, question: str) -> tuple[str | None, str | None]:
    item_no_match = re.search(r"\b(\d{4,})\b", question)
    if not item_no_match:
        return None, None

    item_no = item_no_match.group(1)
    row = db.execute(
        text("SELECT item_no, product_name FROM products WHERE item_no = :v LIMIT 1"),
        {"v": item_no},
    ).mappings().first()

    if not row:
        return None, None

    label = f"{row['item_no']}_{row['product_name']}"
    return label, str(row["product_name"])


def _find_candidate_products(db: Session, keyword: str) -> list[dict[str, str]]:
    cleaned = _clean_keyword(keyword)
    if not cleaned or len(cleaned) < 2:
        return []

    rows = db.execute(
        text(
            "SELECT item_no, product_name FROM products "
            "WHERE product_name LIKE :kw ORDER BY LENGTH(product_name) ASC LIMIT 8"
        ),
        {"kw": f"%{cleaned}%"},
    ).mappings().all()

    return [
        {
            "item_no": str(row["item_no"]),
            "product_name": str(row["product_name"]),
            "label": f"{row['item_no']}_{row['product_name']}",
        }
        for row in rows
    ]


def _find_cart_candidates(keyword: str, billing_items: dict[str, int]) -> list[dict[str, str]]:
    cleaned = _clean_keyword(keyword)
    if not cleaned or len(cleaned) < 2:
        return []

    normalized = _normalize_text(cleaned)
    if not normalized:
        return []

    candidates: list[dict[str, str]] = []
    for label in billing_items.keys():
        display = _label_display(label)
        if normalized in _normalize_text(display):
            item_no = None
            if "_" in label:
                prefix, _ = label.split("_", 1)
                if prefix.isdigit():
                    item_no = prefix
            candidates.append(
                {
                    "item_no": item_no or "",
                    "product_name": display,
                    "label": label,
                }
            )

    return candidates[:8]


def _detect_cart_action(question: str) -> str | None:
    if any(token in question for token in _CLEAR_TOKENS):
        return "clear"
    if any(token in question for token in _ADD_TOKENS):
        return "add"
    if any(token in question for token in _REMOVE_TOKENS):
        return "remove"
    return None


def _clean_keyword(text: str) -> str:
    cleaned = text
    tokens = list(_ADD_TOKENS + _REMOVE_TOKENS + _POLITE_TOKENS + _CLEAR_TOKENS)
    tokens.sort(key=len, reverse=True)
    for token in tokens:
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"[0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()



def _read_hf_token_from_env_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
                return value
    except OSError:
        return None

    return None


def _get_hf_token() -> str | None:
    """Resolve HF token from env first, then common .env fallback files."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()

    token_file = os.getenv("HF_TOKEN_FILE")
    if token_file:
        token_from_file = _read_hf_token_from_env_file(Path(token_file).expanduser())
        if token_from_file:
            return token_from_file

    backend_dir = Path(__file__).resolve().parents[1]
    app_dir = backend_dir.parent
    project_root = app_dir.parent

    env_candidates = [
        project_root / ".env",
        app_dir / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_candidates:
        token_from_file = _read_hf_token_from_env_file(env_path)
        if token_from_file:
            return token_from_file

    return None
# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatbotRequest(BaseModel):
    question: str
    session_id: str | None = None


class ProductMeta(BaseModel):
    name: str
    quantity: int
    product_name: str | None = None
    item_no: str | None = None
    unit_price: int | None = None
    line_total: int = 0
    price_found: bool = False


# ---------------------------------------------------------------------------
# DB helpers – reuse products / product_prices tables
# ---------------------------------------------------------------------------

def _find_product_row(db: Session, label: str) -> dict[str, Any] | None:
    """Try to find a product by item_no prefix or product_name."""
    name = (label or "").strip()
    if not name:
        return None

    # Try item_no_productname format (e.g. "1234_코카콜라")
    if "_" in name:
        prefix, suffix = name.split("_", 1)
        if prefix.isdigit():
            row = db.execute(
                text("SELECT id, item_no, barcd, product_name FROM products WHERE item_no = :v ORDER BY id DESC LIMIT 1"),
                {"v": prefix},
            ).mappings().first()
            if row:
                return dict(row)

    # Try exact product_name match
    row = db.execute(
        text("SELECT id, item_no, barcd, product_name FROM products WHERE product_name = :v ORDER BY id DESC LIMIT 1"),
        {"v": name},
    ).mappings().first()
    if row:
        return dict(row)

    # Try raw label as product_name
    row = db.execute(
        text("SELECT id, item_no, barcd, product_name FROM products WHERE product_name = :v ORDER BY id DESC LIMIT 1"),
        {"v": label},
    ).mappings().first()
    if row:
        return dict(row)

    return None


def _find_latest_price(db: Session, product_id: int) -> int | None:
    """Return latest unit price (KRW) for the given product, or None."""
    row = db.execute(
        text(
            "SELECT price FROM product_prices "
            "WHERE product_id = :pid ORDER BY checked_at DESC, id DESC LIMIT 1"
        ),
        {"pid": product_id},
    ).mappings().first()
    return int(row["price"]) if row else None


def _catalog_available(db: Session) -> bool:
    """Check if catalog tables exist and are queryable."""
    try:
        db.execute(text("SELECT 1 FROM products LIMIT 1"))
        db.execute(text("SELECT 1 FROM product_prices LIMIT 1"))
        return True
    except SQLAlchemyError:
        return False


def _build_cart_meta(db: Session, billing_items: dict[str, int]) -> list[ProductMeta]:
    """Resolve billing labels to product metadata via products/product_prices."""
    if not billing_items:
        return []

    catalog_ok = _catalog_available(db)
    result: list[ProductMeta] = []

    for label, qty in billing_items.items():
        product_row: dict[str, Any] | None = None
        unit_price: int | None = None

        if catalog_ok:
            try:
                product_row = _find_product_row(db, label)
            except SQLAlchemyError:
                product_row = None

            if product_row:
                try:
                    unit_price = _find_latest_price(db, int(product_row["id"]))
                except SQLAlchemyError:
                    unit_price = None

        result.append(
            ProductMeta(
                name=label,
                quantity=qty,
                product_name=str(product_row["product_name"]) if product_row else None,
                item_no=str(product_row["item_no"]) if product_row else None,
                unit_price=unit_price,
                line_total=(unit_price or 0) * qty,
                price_found=unit_price is not None,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _totals(products: list[ProductMeta]) -> dict[str, Any]:
    total_count = sum(p.quantity for p in products)
    total_price = sum(p.line_total for p in products)
    priced = sum(1 for p in products if p.price_found)
    unpriced = [p.name for p in products if not p.price_found]

    return {
        "total_count": total_count,
        "total_price": total_price,
        "priced_items": priced,
        "unpriced_items": unpriced,
    }


_CATALOG_STOPWORDS = {
    "상품",
    "정보",
    "성분",
    "현재",
    "기준",
    "관련",
    "알려줘",
    "알려",
    "보여줘",
    "보여",
    "뭐야",
    "뭐",
    "어떤",
    "있는",
    "대해",
    "가격",
    "할인",
    "할인율",
    "영양",
    "영양소",
    "영양정보",
    "영양성분",
    "칼로리",
    "열량",
    "단백질",
    "탄수화물",
    "지방",
    "포화지방",
    "트랜스지방",
    "당류",
    "당",
    "나트륨",
    "장바구니",
    "그리고",
    "또",
    "좀",
    "해줘",
    "주세요",
}
_DISCOUNT_QUERY_TOKENS = ("할인", "세일", "할인율", "할인가")
_PRICE_QUERY_TOKENS = ("가격", "비싼", "저렴", "최고가", "최저가", "금액", "얼마")
_LOCATION_QUERY_TOKENS = (
    "위치",
    "어디",
    "코너",
    "몇번",
    "몇 번",
    "어느 코너",
    "어느코너",
    "어디에",
    "location",
    "aisle",
)
_LOCATION_TERM_STOPWORDS = {
    "위치",
    "어디",
    "어디야",
    "어디에",
    "코너",
    "몇번",
    "몇",
    "번",
    "location",
    "aisle",
}
_NUTRITION_QUERY_TOKENS = (
    "영양",
    "영양소",
    "영양정보",
    "영양성분",
    "칼로리",
    "열량",
    "단백질",
    "탄수화물",
    "지방",
    "포화지방",
    "트랜스지방",
    "당류",
    "당",
    "나트륨",
)
_NUTRITION_RANK_QUERY_TOKENS = (
    "제일 높은",
    "가장 높은",
    "최고",
    "top",
    "탑",
    "순위",
    "랭킹",
)
_MAX_CATALOG_MATCH_ROWS = 8
_NUTRIENT_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "칼로리": ("칼로리", "열량", "에너지", "kcal", "calorie", "㎉"),
    "단백질": ("단백질", "protein"),
    "탄수화물": ("탄수화물", "탄수", "carbohydrate", "carb", "carbs"),
    "지방": ("지방", "총지방", "fat"),
    "포화지방": ("포화지방", "saturated"),
    "트랜스지방": ("트랜스지방", "trans"),
    "당류": ("당류", "총당류", "당", "당분", "sugar"),
    "나트륨": ("나트륨", "소듐", "sodium"),
}


def _compact_prompt_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text_value = str(value).strip()
    if not text_value:
        return None
    if len(text_value) > 180:
        return f"{text_value[:177]}..."
    return text_value


def _json_for_prompt(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except Exception:
        return str(value)


def _table_columns(db: Session, table_name: str) -> list[str]:
    try:
        db_inspector = inspect(db.get_bind())
        raw_columns = db_inspector.get_columns(table_name)
    except Exception:
        return []

    valid: list[str] = []
    for col in raw_columns:
        name = str(col.get("name") or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            valid.append(name)
    return valid


def _extract_catalog_terms(question: str, limit: int = 4) -> list[str]:
    tokens = re.findall(r"[0-9]{3,}|[A-Za-z]{2,}|[가-힣]{2,}", question or "")
    terms: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        normalized = token.lower()
        if normalized in _CATALOG_STOPWORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _query_catalog_products(
    db: Session,
    question: str,
    product_columns: list[str],
    limit: int = _MAX_CATALOG_MATCH_ROWS,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not product_columns:
        return [], []

    terms = _extract_catalog_terms(question)
    safe_limit = max(1, min(limit, 20))

    preferred = [
        "item_no",
        "product_name",
        "barcd",
        "nutrition_info",
        "nutrition",
        "nutrients",
        "kcal",
        "calories",
        "protein",
        "carbohydrate",
        "fat",
        "sodium",
        "sugar",
        "category_l",
        "category_m",
        "category_s",
        "category",
        "brand",
        "manufacturer",
        "maker",
        "origin",
        "description",
        "desc",
        "spec",
        "specification",
        "unit",
    ]
    searchable = [col for col in preferred if col in product_columns]
    if not searchable:
        searchable = [
            col
            for col in product_columns
            if any(
                key in col.lower()
                for key in (
                    "name",
                    "item",
                    "bar",
                    "category",
                    "brand",
                    "maker",
                    "origin",
                    "desc",
                    "spec",
                    "unit",
                    "nutrition",
                    "nutri",
                    "kcal",
                    "calorie",
                    "protein",
                    "carb",
                    "fat",
                    "sodium",
                    "sugar",
                )
            )
        ]
    if not searchable:
        searchable = [product_columns[0]]

    params: dict[str, Any] = {}
    groups: list[str] = []
    for idx, term in enumerate(terms):
        like_key = f"kw_{idx}"
        params[like_key] = f"%{term}%"
        clauses = [f"p.`{col}` LIKE :{like_key}" for col in searchable]
        if "product_name" in product_columns:
            normalized_term = re.sub(r"[\s\-_]+", "", term)
            if normalized_term:
                norm_key = f"kw_norm_{idx}"
                params[norm_key] = f"%{normalized_term}%"
                clauses.append(
                    "REPLACE(REPLACE(REPLACE(p.`product_name`, ' ', ''), '-', ''), '_', '') "
                    f"LIKE :{norm_key}"
                )
        if term.isdigit() and "item_no" in product_columns:
            exact_key = f"item_no_{idx}"
            clauses.append(f"p.`item_no` = :{exact_key}")
            params[exact_key] = term
        groups.append("(" + " OR ".join(clauses) + ")")

    where_sql = f"WHERE {' AND '.join(groups)}" if groups else ""
    order_parts: list[str] = []
    for col in ("updated_at", "created_at", "id"):
        if col in product_columns:
            order_parts.append(f"p.`{col}` DESC")
    if not order_parts:
        order_parts.append(f"p.`{product_columns[0]}` DESC")

    params["limit"] = safe_limit * 2 if "item_no" in product_columns else safe_limit

    def _run_catalog_query(where_clause: str) -> list[dict[str, Any]]:
        sql = (
            "SELECT p.* "
            "FROM products p "
            f"{where_clause} "
            f"ORDER BY {', '.join(order_parts)} "
            "LIMIT :limit"
        )
        rows = db.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]

    try:
        normalized_rows = _run_catalog_query(where_sql)
        if not normalized_rows and len(groups) > 1:
            fallback_where = f"WHERE {' OR '.join(groups)}"
            normalized_rows = _run_catalog_query(fallback_where)
    except SQLAlchemyError:
        return [], terms
    if "item_no" not in product_columns:
        return normalized_rows[:safe_limit], terms

    deduped: list[dict[str, Any]] = []
    seen_item_nos: set[str] = set()
    for row in normalized_rows:
        item_no = str(row.get("item_no") or "").strip()
        key = item_no if item_no else str(row.get("id") or "")
        if not key or key in seen_item_nos:
            continue
        seen_item_nos.add(key)
        deduped.append(row)
        if len(deduped) >= safe_limit:
            break

    return deduped, terms


def _latest_price_row(
    db: Session,
    product_id: int,
    price_columns: list[str],
) -> dict[str, Any]:
    if not price_columns:
        return {}
    if "product_id" not in price_columns or "price" not in price_columns:
        return {}

    select_cols = [
        col for col in ("id", "price", "currency", "source", "checked_at", "created_at")
        if col in price_columns
    ]
    if "price" not in select_cols:
        return {}

    order_cols = [col for col in ("checked_at", "created_at", "id") if col in price_columns]
    if not order_cols:
        order_cols = ["price"]

    sql = (
        f"SELECT {', '.join(f'`{col}`' for col in select_cols)} "
        "FROM product_prices "
        "WHERE `product_id` = :pid "
        f"ORDER BY {', '.join(f'`{col}` DESC' for col in order_cols)} "
        "LIMIT 1"
    )
    row = db.execute(text(sql), {"pid": int(product_id)}).mappings().first()
    return dict(row) if row else {}


def _latest_discount_row(
    db: Session,
    product_price_id: int,
    discount_columns: list[str],
) -> dict[str, Any]:
    if not discount_columns:
        return {}
    if "product_price_id" not in discount_columns:
        return {}
    if (
        "is_discounted" not in discount_columns
        and "discount_rate" not in discount_columns
        and "discount_amount" not in discount_columns
    ):
        return {}

    select_cols = [
        col
        for col in ("id", "is_discounted", "discount_rate", "discount_amount", "started_at", "ended_at", "updated_at", "created_at")
        if col in discount_columns
    ]
    order_cols = [col for col in ("updated_at", "created_at", "id") if col in discount_columns]
    if not order_cols:
        order_cols = [select_cols[0]]

    sql = (
        f"SELECT {', '.join(f'`{col}`' for col in select_cols)} "
        "FROM product_discounts "
        "WHERE `product_price_id` = :ppid "
        f"ORDER BY {', '.join(f'`{col}` DESC' for col in order_cols)} "
        "LIMIT 1"
    )
    row = db.execute(text(sql), {"ppid": int(product_price_id)}).mappings().first()
    return dict(row) if row else {}


def _catalog_summary(
    db: Session,
    product_columns: list[str],
    discount_columns: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not product_columns:
        return summary

    try:
        if "item_no" in product_columns:
            total_products = db.execute(
                text("SELECT COUNT(DISTINCT `item_no`) AS cnt FROM products")
            ).mappings().first()
        else:
            total_products = db.execute(
                text("SELECT COUNT(*) AS cnt FROM products")
            ).mappings().first()
        summary["total_products"] = int(total_products["cnt"]) if total_products else 0
    except SQLAlchemyError:
        pass

    for category_col in ("category_l", "category_m", "category_s", "category"):
        if category_col not in product_columns:
            continue
        try:
            category_row = db.execute(
                text(
                    f"SELECT COUNT(DISTINCT `{category_col}`) AS cnt "
                    "FROM products "
                    f"WHERE `{category_col}` IS NOT NULL AND TRIM(`{category_col}`) <> ''"
                )
            ).mappings().first()
            summary[f"{category_col}_count"] = int(category_row["cnt"]) if category_row else 0
        except SQLAlchemyError:
            continue

    if "is_discounted" in discount_columns:
        try:
            discount_row = db.execute(
                text(
                    "SELECT COUNT(*) AS cnt "
                    "FROM product_discounts "
                    "WHERE `is_discounted` = 1"
                )
            ).mappings().first()
            summary["discount_rows"] = int(discount_row["cnt"]) if discount_row else 0
        except SQLAlchemyError:
            pass

    return summary


def _category_corner_snapshot(db: Session) -> dict[str, dict[str, Any]]:
    map_columns = _table_columns(db, "category_corner_map")
    corner_columns = _table_columns(db, "store_corners")
    if not map_columns or not corner_columns:
        return {}
    if not {"category_l", "corner_id"}.issubset(set(map_columns)):
        return {}
    if not {"id", "corner_no"}.issubset(set(corner_columns)):
        return {}

    try:
        rows = db.execute(
            text(
                "SELECT m.category_l, c.corner_no, c.corner_name "
                "FROM category_corner_map m "
                "JOIN store_corners c ON c.id = m.corner_id "
                "ORDER BY c.corner_no ASC, m.category_l ASC"
            )
        ).mappings().all()
    except SQLAlchemyError:
        return {}

    snapshot: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = str(row.get("category_l") or "").strip()
        corner_no = row.get("corner_no")
        if not category or corner_no is None:
            continue
        corner_name = str(row.get("corner_name") or "").strip()
        snapshot[category] = {
            "corner_no": int(corner_no),
            "corner_name": corner_name or f"{int(corner_no)}번 코너",
        }
    return snapshot


def _discount_snapshot(
    db: Session,
    question: str,
    product_columns: list[str],
    price_columns: list[str],
    discount_columns: list[str],
) -> list[dict[str, Any]]:
    if not any(token in question for token in _DISCOUNT_QUERY_TOKENS):
        return []
    required = {"id", "item_no", "product_name"}
    if not required.issubset(set(product_columns)):
        return []
    if not {"id", "product_id"}.issubset(set(price_columns)):
        return []
    if "product_price_id" not in discount_columns:
        return []

    select_cols = ["p.`item_no` AS item_no", "p.`product_name` AS product_name"]
    if "discount_rate" in discount_columns:
        select_cols.append("pd.`discount_rate` AS discount_rate")
    if "discount_amount" in discount_columns:
        select_cols.append("pd.`discount_amount` AS discount_amount")
    if "is_discounted" in discount_columns:
        where_clause = "WHERE pd.`is_discounted` = 1"
    else:
        where_clause = ""

    order_terms: list[str] = []
    if "discount_rate" in discount_columns:
        order_terms.append("COALESCE(pd.`discount_rate`, 0) DESC")
    if "discount_amount" in discount_columns:
        order_terms.append("COALESCE(pd.`discount_amount`, 0) DESC")
    order_terms.append("p.`id` DESC")

    sql = (
        f"SELECT {', '.join(select_cols)} "
        "FROM products p "
        "JOIN product_prices pp ON pp.`product_id` = p.`id` "
        "JOIN product_discounts pd ON pd.`product_price_id` = pp.`id` "
        f"{where_clause} "
        f"ORDER BY {', '.join(order_terms)} "
        "LIMIT :limit"
    )
    try:
        rows = db.execute(text(sql), {"limit": 8}).mappings().all()
    except SQLAlchemyError:
        return []

    deduped: list[dict[str, Any]] = []
    seen_item_nos: set[str] = set()
    for row in rows:
        item_no = str(row.get("item_no") or "").strip()
        if item_no and item_no in seen_item_nos:
            continue
        if item_no:
            seen_item_nos.add(item_no)
        compact = {
            key: _compact_prompt_value(value)
            for key, value in dict(row).items()
            if _compact_prompt_value(value) is not None
        }
        if compact:
            deduped.append(compact)
    return deduped[:5]


def _price_snapshot(
    db: Session,
    question: str,
    product_columns: list[str],
    price_columns: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not any(token in question for token in _PRICE_QUERY_TOKENS):
        return {}
    required_product_cols = {"id", "item_no", "product_name"}
    if not required_product_cols.issubset(set(product_columns)):
        return {}
    if not {"product_id", "price"}.issubset(set(price_columns)):
        return {}

    select_cols = ["p.`item_no` AS item_no", "p.`product_name` AS product_name", "pp.`price` AS price"]
    if "currency" in price_columns:
        select_cols.append("pp.`currency` AS currency")

    def _fetch(order: str) -> list[dict[str, Any]]:
        sql = (
            f"SELECT {', '.join(select_cols)} "
            "FROM products p "
            "JOIN product_prices pp ON pp.`product_id` = p.`id` "
            f"ORDER BY pp.`price` {order}, p.`id` DESC "
            "LIMIT :limit"
        )
        try:
            rows = db.execute(text(sql), {"limit": 8}).mappings().all()
        except SQLAlchemyError:
            return []

        deduped: list[dict[str, Any]] = []
        seen_item_nos: set[str] = set()
        for row in rows:
            item_no = str(row.get("item_no") or "").strip()
            if item_no and item_no in seen_item_nos:
                continue
            if item_no:
                seen_item_nos.add(item_no)
            compact = {
                key: _compact_prompt_value(value)
                for key, value in dict(row).items()
                if _compact_prompt_value(value) is not None
            }
            if compact:
                deduped.append(compact)
        return deduped[:5]

    expensive = _fetch("DESC")
    cheap = _fetch("ASC")
    result: dict[str, list[dict[str, Any]]] = {}
    if expensive:
        result["expensive_top5"] = expensive
    if cheap:
        result["cheap_top5"] = cheap
    return result


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text_value = value.strip()
    if not text_value:
        return value
    if not (
        (text_value.startswith("{") and text_value.endswith("}"))
        or (text_value.startswith("[") and text_value.endswith("]"))
    ):
        return value
    try:
        return json.loads(text_value)
    except Exception:
        return value


def _nutrition_snapshot(
    db: Session,
    question: str,
    product_columns: list[str],
) -> dict[str, Any]:
    if not any(token in question for token in _NUTRITION_QUERY_TOKENS):
        return {}
    if not product_columns:
        return {}

    nutrition_columns = [
        col
        for col in product_columns
        if any(
            key in col.lower()
            for key in (
                "nutrition",
                "nutri",
                "kcal",
                "calorie",
                "protein",
                "carb",
                "fat",
                "sodium",
                "sugar",
            )
        )
    ]
    if not nutrition_columns:
        return {}

    matched_rows, terms = _query_catalog_products(
        db=db,
        question=question,
        product_columns=product_columns,
        limit=_MAX_CATALOG_MATCH_ROWS,
    )
    if not matched_rows:
        return {"nutrition_columns": nutrition_columns, "search_terms": terms, "products": []}

    payload_rows: list[dict[str, Any]] = []
    for row in matched_rows:
        compact: dict[str, Any] = {}
        for base_col in ("item_no", "product_name"):
            if base_col in product_columns:
                base_val = _compact_prompt_value(row.get(base_col))
                if base_val is not None:
                    compact[base_col] = base_val

        for col in nutrition_columns:
            raw_value = row.get(col)
            if raw_value is None:
                continue
            parsed = _maybe_json(raw_value)
            if isinstance(parsed, (dict, list)):
                compact[col] = parsed
                continue
            value = _compact_prompt_value(parsed)
            if value is not None:
                compact[col] = value

        if compact:
            payload_rows.append(compact)

    return {
        "nutrition_columns": nutrition_columns,
        "search_terms": terms,
        "products": payload_rows,
    }


def _build_catalog_context(db: Session, question: str) -> dict[str, Any]:
    product_columns = _table_columns(db, "products")
    if not product_columns:
        return {"available": False}

    price_columns = _table_columns(db, "product_prices")
    discount_columns = _table_columns(db, "product_discounts")

    product_rows, terms = _query_catalog_products(db, question, product_columns, _MAX_CATALOG_MATCH_ROWS)
    matched_rows: list[dict[str, Any]] = []

    for row in product_rows:
        compact_row: dict[str, Any] = {}
        for col in product_columns:
            value = _compact_prompt_value(row.get(col))
            if value is not None:
                compact_row[col] = value

        product_id = row.get("id")
        if product_id is not None and price_columns:
            try:
                price_row = _latest_price_row(db, int(product_id), price_columns)
            except SQLAlchemyError:
                price_row = {}
            if price_row:
                compact_row["latest_price"] = _compact_prompt_value(price_row.get("price"))
                compact_row["latest_currency"] = _compact_prompt_value(price_row.get("currency"))
                compact_row["latest_price_source"] = _compact_prompt_value(price_row.get("source"))
                compact_row["latest_price_checked_at"] = _compact_prompt_value(price_row.get("checked_at"))

                if discount_columns and price_row.get("id") is not None:
                    try:
                        discount_row = _latest_discount_row(db, int(price_row["id"]), discount_columns)
                    except SQLAlchemyError:
                        discount_row = {}
                    if discount_row:
                        compact_row["is_discounted"] = _compact_prompt_value(discount_row.get("is_discounted"))
                        compact_row["discount_rate"] = _compact_prompt_value(discount_row.get("discount_rate"))
                        compact_row["discount_amount"] = _compact_prompt_value(discount_row.get("discount_amount"))
                        compact_row["discount_updated_at"] = _compact_prompt_value(discount_row.get("updated_at"))

        matched_rows.append({k: v for k, v in compact_row.items() if v is not None})

    context: dict[str, Any] = {
        "available": True,
        "schema": {
            "products": product_columns,
            "product_prices": price_columns,
            "product_discounts": discount_columns,
        },
        "summary": _catalog_summary(db, product_columns, discount_columns),
        "search_terms": terms,
        "matched_products": matched_rows[:_MAX_CATALOG_MATCH_ROWS],
        "corner_map": _category_corner_snapshot(db),
    }

    discount_rows = _discount_snapshot(db, question, product_columns, price_columns, discount_columns)
    if discount_rows:
        context["discount_snapshot"] = discount_rows

    price_rows = _price_snapshot(db, question, product_columns, price_columns)
    if price_rows:
        context["price_snapshot"] = price_rows

    nutrition_rows = _nutrition_snapshot(db, question, product_columns)
    if nutrition_rows:
        context["nutrition_snapshot"] = nutrition_rows

    nutrition_rank_rows = _nutrition_rank_snapshot(db, question, product_columns)
    if nutrition_rank_rows:
        context["nutrition_rank_snapshot"] = nutrition_rank_rows

    return context


# ---------------------------------------------------------------------------
# LLM answer generator
# ---------------------------------------------------------------------------

def _normalize_query_token(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", (value or "").lower())


def _detect_requested_nutrients(question: str) -> list[str]:
    normalized_question = _normalize_query_token((question or "").lower())
    requested_nutrients: list[str] = []
    for nutrient_name, aliases in _NUTRIENT_ALIAS_MAP.items():
        normalized_aliases = [
            _normalize_query_token(alias)
            for alias in aliases
            if _normalize_query_token(alias)
        ]
        if normalized_aliases and any(alias in normalized_question for alias in normalized_aliases):
            requested_nutrients.append(nutrient_name)
    return requested_nutrients


def _is_nutrition_rank_query(question: str) -> bool:
    lowered = (question or "").lower()
    if not any(token in lowered for token in _NUTRITION_QUERY_TOKENS):
        return False
    return any(token in lowered for token in _NUTRITION_RANK_QUERY_TOKENS)


def _extract_category_terms_for_rank(question: str, max_terms: int = 3) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{2,}|[가-힣]{2,}", question or "")
    nutrient_tokens = {
        _normalize_query_token(token)
        for token in _NUTRITION_QUERY_TOKENS
        if _normalize_query_token(token)
    }
    for aliases in _NUTRIENT_ALIAS_MAP.values():
        nutrient_tokens.update(
            _normalize_query_token(alias)
            for alias in aliases
            if _normalize_query_token(alias)
        )

    ranking_stopwords = {
        "제일",
        "가장",
        "높은",
        "낮은",
        "최고",
        "최저",
        "top",
        "탑",
        "순위",
        "랭킹",
        "카테고리",
        "카테고리별",
        "별",
        "상품",
        "제품",
        "알려",
        "알려줘",
        "뭐야",
        "무엇",
        "정보",
    }
    ranking_stopwords.update(_CATALOG_STOPWORDS)
    ranking_stopwords_norm = {
        _normalize_query_token(token)
        for token in ranking_stopwords
        if _normalize_query_token(token)
    }
    ranking_stopwords_norm.update(nutrient_tokens)

    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = _normalize_query_token(token)
        if not normalized:
            continue
        if normalized in ranking_stopwords_norm:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token.strip())
        if len(terms) >= max_terms:
            break
    return terms


def _numeric_from_text(value: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _flatten_nutrition_pairs(value: Any, parent_key: str = "") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).strip()
            if isinstance(v, (dict, list)):
                pairs.extend(_flatten_nutrition_pairs(v, key or parent_key))
            else:
                pairs.append((key or parent_key, str(v).strip()))
        return pairs
    if isinstance(value, list):
        for item in value:
            pairs.extend(_flatten_nutrition_pairs(item, parent_key))
        return pairs
    text_value = str(value).strip()
    if text_value:
        pairs.append((parent_key, text_value))
    return pairs


def _collect_nutrition_pairs(product_row: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key, raw_value in product_row.items():
        if key in {"item_no", "product_name"}:
            continue
        parsed = _maybe_json(raw_value)
        if isinstance(parsed, (dict, list)):
            pairs.extend(_flatten_nutrition_pairs(parsed, str(key)))
            continue
        value = str(parsed).strip()
        if value:
            pairs.append((str(key), value))
    return pairs


def _nutrition_rank_snapshot(
    db: Session,
    question: str,
    product_columns: list[str],
) -> dict[str, Any]:
    if not _is_nutrition_rank_query(question):
        return {}
    if not product_columns or "product_name" not in product_columns:
        return {}

    requested_nutrients = _detect_requested_nutrients(question)
    if not requested_nutrients:
        return {}
    target_nutrient = requested_nutrients[0]

    nutrition_columns = [
        col
        for col in product_columns
        if any(
            key in col.lower()
            for key in (
                "nutrition",
                "nutri",
                "kcal",
                "calorie",
                "protein",
                "carb",
                "fat",
                "sodium",
                "sugar",
            )
        )
    ]
    if not nutrition_columns:
        return {}

    category_cols = [col for col in ("category_s", "category_m", "category_l", "category") if col in product_columns]

    select_cols: list[str] = []
    for col in ("item_no", "product_name"):
        if col in product_columns:
            select_cols.append(f"p.`{col}`")
    for col in category_cols:
        select_cols.append(f"p.`{col}`")
    for col in nutrition_columns:
        alias = f"p.`{col}`"
        if alias not in select_cols:
            select_cols.append(alias)

    nutrition_exists = " OR ".join(
        f"(p.`{col}` IS NOT NULL AND TRIM(CAST(p.`{col}` AS CHAR)) <> '')"
        for col in nutrition_columns
    )
    where_parts = [f"({nutrition_exists})"]

    category_terms = _extract_category_terms_for_rank(question, max_terms=3)
    params: dict[str, Any] = {}
    if category_terms:
        for idx, term in enumerate(category_terms):
            key = f"cat_kw_{idx}"
            params[key] = f"%{term}%"
            term_clauses: list[str] = [f"p.`product_name` LIKE :{key}"]
            for col in category_cols:
                term_clauses.append(f"p.`{col}` LIKE :{key}")
            where_parts.append("(" + " OR ".join(term_clauses) + ")")

    order_parts: list[str] = []
    for col in ("updated_at", "created_at", "id"):
        if col in product_columns:
            order_parts.append(f"p.`{col}` DESC")
    if not order_parts:
        order_parts.append(f"p.`{product_columns[0]}` DESC")

    params["limit"] = 5000 if category_terms else 2000
    sql = (
        f"SELECT {', '.join(select_cols)} "
        "FROM products p "
        f"WHERE {' AND '.join(where_parts)} "
        f"ORDER BY {', '.join(order_parts)} "
        "LIMIT :limit"
    )

    try:
        rows = db.execute(text(sql), params).mappings().all()
    except SQLAlchemyError:
        return {}

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        nutrition_pairs = _collect_nutrition_pairs(row_dict)
        if not nutrition_pairs:
            continue

        nutrient_value = _extract_nutrient_value_from_pairs(
            nutrient_name=target_nutrient,
            aliases=_NUTRIENT_ALIAS_MAP[target_nutrient],
            pairs=nutrition_pairs,
        )
        if nutrient_value is None:
            continue
        nutrient_numeric = _numeric_from_text(nutrient_value)
        if nutrient_numeric is None:
            continue

        category_label = ""
        for col in ("category_s", "category_m", "category_l", "category"):
            value = str(row_dict.get(col) or "").strip()
            if value:
                category_label = value
                break

        scored_rows.append(
            {
                "item_no": str(row_dict.get("item_no") or "").strip(),
                "product_name": str(row_dict.get("product_name") or "").strip(),
                "category": category_label or "기타",
                "nutrient_value": nutrient_value,
                "nutrient_numeric": nutrient_numeric,
            }
        )

    if not scored_rows:
        return {}

    scored_rows.sort(key=lambda item: item.get("nutrient_numeric", 0), reverse=True)
    category_wise = "카테고리별" in (question or "") or "카테고리 별" in (question or "")

    if category_wise:
        best_by_category: dict[str, dict[str, Any]] = {}
        for row in scored_rows:
            category_key = str(row.get("category") or "기타").strip() or "기타"
            if category_key not in best_by_category:
                best_by_category[category_key] = row
        results = list(best_by_category.values())[:8]
    else:
        deduped: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for row in scored_rows:
            key = str(row.get("item_no") or "").strip() or str(row.get("product_name") or "").strip()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)
            if len(deduped) >= 5:
                break
        results = deduped

    return {
        "target_nutrient": target_nutrient,
        "category_terms": category_terms,
        "category_wise": category_wise,
        "results": results,
    }


def _format_nutrient_value(key: str, value: str) -> str | None:
    cleaned = (value or "").strip().strip('"').strip("'")
    if not cleaned:
        return None
    number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(kcal|㎉|g|mg|ml|%)?", cleaned, flags=re.IGNORECASE)
    if not number_match:
        return cleaned

    number = number_match.group(1)
    unit = number_match.group(2)
    if unit:
        return f"{number}{unit}"

    key_unit = re.search(r"\(([^)]+)\)", key or "")
    if key_unit and key_unit.group(1).strip():
        return f"{number}{key_unit.group(1).strip()}"
    return number


def _extract_nutrient_value_from_pairs(
    nutrient_name: str,
    aliases: tuple[str, ...],
    pairs: list[tuple[str, str]],
) -> str | None:
    normalized_aliases = [_normalize_query_token(alias) for alias in aliases if _normalize_query_token(alias)]
    if not normalized_aliases:
        return None

    for key, value in pairs:
        key_norm = _normalize_query_token(key)
        if key_norm and any(alias in key_norm for alias in normalized_aliases):
            formatted = _format_nutrient_value(key, value)
            if formatted:
                return formatted

    combined = " | ".join(f"{k}:{v}" for k, v in pairs if k or v)
    for alias in aliases:
        alias_pattern = re.escape(alias)
        patterns = (
            rf"{alias_pattern}\s*(?:\([^)]*\))?\s*\"?\s*[:：]\s*\"?([0-9]+(?:\.[0-9]+)?\s*(?:kcal|㎉|g|mg|ml|%)?)",
            rf"{alias_pattern}\s*\"?\s*([0-9]+(?:\.[0-9]+)?\s*(?:kcal|㎉|g|mg|ml|%)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, combined, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return None


def _product_match_score(product_row: dict[str, Any], terms: list[str]) -> int:
    if not terms:
        return 0
    product_name = _normalize_query_token(str(product_row.get("product_name") or ""))
    item_no = _normalize_query_token(str(product_row.get("item_no") or ""))
    score = 0
    for term in terms:
        norm_term = _normalize_query_token(term)
        if not norm_term:
            continue
        if product_name.startswith(norm_term):
            score += 5
        elif norm_term in product_name:
            score += 3
        if item_no and norm_term == item_no:
            score += 4
    return score


def _is_location_query(question: str) -> bool:
    lowered = (question or "").lower()
    return any(token in lowered for token in _LOCATION_QUERY_TOKENS)


def _format_corner_label(corner_no: Any, corner_name: Any) -> str | None:
    try:
        no = int(corner_no)
    except (TypeError, ValueError):
        return None
    default_name = f"{no}번 코너"
    cleaned_name = str(corner_name or "").strip()
    if cleaned_name and cleaned_name != default_name:
        return f"{default_name} ({cleaned_name})"
    return default_name


def _answer_corner_location_direct(question: str, catalog_context: dict[str, Any] | None) -> str | None:
    if not _is_location_query(question):
        return None
    if not catalog_context or not catalog_context.get("available"):
        return "상품 위치 정보는 현재 DB에서 확인되지 않습니다."

    raw_corner_map = catalog_context.get("corner_map")
    if not isinstance(raw_corner_map, dict) or not raw_corner_map:
        return "카테고리-코너 매핑이 아직 등록되지 않았습니다."

    corner_map: dict[str, dict[str, Any]] = {
        str(k).strip(): v
        for k, v in raw_corner_map.items()
        if str(k).strip() and isinstance(v, dict)
    }
    if not corner_map:
        return "카테고리-코너 매핑이 아직 등록되지 않았습니다."

    normalized_question = _normalize_query_token(question)
    for category, info in corner_map.items():
        normalized_category = _normalize_query_token(category)
        if not normalized_category or normalized_category not in normalized_question:
            continue
        corner_label = _format_corner_label(info.get("corner_no"), info.get("corner_name"))
        if corner_label:
            return f"{category} 카테고리는 {corner_label}에 있습니다."

    raw_products = catalog_context.get("matched_products")
    if not isinstance(raw_products, list):
        return "질문하신 상품을 찾지 못했습니다."

    product_rows = [row for row in raw_products if isinstance(row, dict)]
    if not product_rows:
        return "질문하신 상품을 찾지 못했습니다."

    raw_terms = catalog_context.get("search_terms")
    terms = raw_terms if isinstance(raw_terms, list) else _extract_catalog_terms(question, limit=6)
    terms = [
        str(term).strip()
        for term in terms
        if _normalize_query_token(str(term)) not in _LOCATION_TERM_STOPWORDS
    ]
    scored_products = sorted(
        [
            (row, _product_match_score(row, terms))
            for row in product_rows
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    resolved: list[str] = []
    seen_keys: set[str] = set()
    for row, score in scored_products:
        if terms and score <= 0:
            continue
        category = str(
            row.get("category_l")
            or row.get("category")
            or row.get("category_m")
            or row.get("category_s")
            or ""
        ).strip()
        if not category:
            continue

        mapping = corner_map.get(category)
        if not mapping:
            normalized_category = _normalize_query_token(category)
            for key, value in corner_map.items():
                if _normalize_query_token(key) == normalized_category:
                    mapping = value
                    category = key
                    break
        if not mapping:
            continue

        corner_label = _format_corner_label(mapping.get("corner_no"), mapping.get("corner_name"))
        if not corner_label:
            continue

        item_no = str(row.get("item_no") or "").strip()
        product_name = str(row.get("product_name") or item_no or "해당 상품").strip()
        dedup_key = item_no or product_name
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        resolved.append(f"{product_name}: {corner_label} (카테고리: {category})")
        if len(resolved) >= 3:
            break

    if not resolved:
        return "해당 상품의 코너 정보가 아직 등록되지 않았습니다."

    if len(resolved) == 1:
        first = resolved[0]
        product_part, right = first.split(": ", 1)
        return f"{product_part}은(는) {right}에 있습니다."

    return "확인된 상품 위치입니다.\n- " + "\n- ".join(resolved)


def _answer_nutrition_direct(question: str, catalog_context: dict[str, Any] | None) -> str | None:
    if not any(token in question for token in _NUTRITION_QUERY_TOKENS):
        return None
    if not catalog_context or not catalog_context.get("available"):
        return None

    nutrition_snapshot = catalog_context.get("nutrition_snapshot")
    if not isinstance(nutrition_snapshot, dict):
        return None

    raw_products = nutrition_snapshot.get("products")
    if not isinstance(raw_products, list):
        return "요청하신 상품의 영양 정보는 현재 DB에서 확인되지 않습니다."

    products = [row for row in raw_products if isinstance(row, dict)]
    if not products:
        return "요청하신 상품의 영양 정보는 현재 DB에서 확인되지 않습니다."

    rank_snapshot = catalog_context.get("nutrition_rank_snapshot")
    if isinstance(rank_snapshot, dict):
        rank_results = rank_snapshot.get("results")
        if isinstance(rank_results, list) and rank_results:
            target_nutrient = str(rank_snapshot.get("target_nutrient") or "영양소").strip()
            category_wise = bool(rank_snapshot.get("category_wise"))
            if category_wise:
                lines: list[str] = []
                for row in rank_results:
                    if not isinstance(row, dict):
                        continue
                    category = str(row.get("category") or "기타").strip() or "기타"
                    product_name = str(row.get("product_name") or "상품명 없음").strip()
                    nutrient_value = str(row.get("nutrient_value") or "").strip()
                    if nutrient_value:
                        lines.append(f"{category}: {product_name} ({nutrient_value})")
                if lines:
                    return f"카테고리별 {target_nutrient} 최고 상품입니다.\n- " + "\n- ".join(lines)

            top = rank_results[0] if isinstance(rank_results[0], dict) else {}
            if top:
                category_terms = rank_snapshot.get("category_terms")
                category_scope = ""
                if isinstance(category_terms, list):
                    tokens = [str(term).strip() for term in category_terms if str(term).strip()]
                    if tokens:
                        category_scope = f"{', '.join(tokens)} 기준 "
                product_name = str(top.get("product_name") or "상품명 없음").strip()
                nutrient_value = str(top.get("nutrient_value") or "").strip()
                if nutrient_value:
                    return f"{category_scope}{target_nutrient}가 가장 높은 상품은 {product_name} ({nutrient_value})입니다."

    search_terms = nutrition_snapshot.get("search_terms")
    terms = search_terms if isinstance(search_terms, list) else _extract_catalog_terms(question, limit=6)
    products = sorted(products, key=lambda row: _product_match_score(row, terms), reverse=True)

    requested_nutrients = _detect_requested_nutrients(question)

    if not requested_nutrients:
        first_product = products[0]
        product_name = str(first_product.get("product_name") or "해당 상품").strip()
        nutrition_pairs = _collect_nutrition_pairs(first_product)
        if not nutrition_pairs:
            return f"{product_name}의 영양 정보는 현재 DB에서 확인되지 않습니다."
        preview = ", ".join(
            f"{k}: {v}" if k else v
            for k, v in nutrition_pairs
            if (k or v)
        )
        if not preview:
            return f"{product_name}의 영양 정보는 현재 DB에서 확인되지 않습니다."
        return f"{product_name} 영양 정보: {preview}"

    answers: list[str] = []
    for product_row in products[:3]:
        product_name = str(product_row.get("product_name") or "해당 상품").strip()
        nutrition_pairs = _collect_nutrition_pairs(product_row)
        if not nutrition_pairs:
            continue
        found_parts: list[str] = []
        for nutrient_name in requested_nutrients:
            value = _extract_nutrient_value_from_pairs(
                nutrient_name=nutrient_name,
                aliases=_NUTRIENT_ALIAS_MAP[nutrient_name],
                pairs=nutrition_pairs,
            )
            if value is not None:
                found_parts.append(f"{nutrient_name}: {value}")
        if found_parts:
            answers.append(f"{product_name} 기준 {', '.join(found_parts)}")

    if answers:
        if len(answers) == 1:
            return f"{answers[0]}입니다."
        return "질문하신 영양 정보입니다.\n- " + "\n- ".join(answers)

    fallback_product = str(products[0].get("product_name") or "해당 상품").strip()
    return f"{fallback_product}의 {', '.join(requested_nutrients)} 정보는 현재 DB에서 확인되지 않습니다."


def _answer_question(
    question: str,
    products: list[ProductMeta],
    total: dict[str, Any],
    catalog_context: dict[str, Any] | None = None,
) -> str:
    q = question.strip()
    if not q:
        return "질문을 입력해 주세요. 예: '총 금액 얼마야?'"
    direct_location = _answer_corner_location_direct(q, catalog_context)
    if direct_location:
        return direct_location
    direct_nutrition = _answer_nutrition_direct(q, catalog_context)
    if direct_nutrition:
        return direct_nutrition

    # Build context for LLM
    cart_lines: list[str] = []
    for item in products:
        price_str = f"{item.unit_price:,}원" if item.unit_price is not None else "가격 미등록"
        display = item.product_name or item.name
        cart_lines.append(
            f"- {display}: {item.quantity}개, 단가: {price_str}, 소계: {item.line_total:,}원"
        )
    cart_text = "\n".join(cart_lines) if cart_lines else "(장바구니 비어있음)"

    unpriced_text = ", ".join(total["unpriced_items"]) if total["unpriced_items"] else "없음"
    total_text = (
        f"총 수량: {total['total_count']}개, "
        f"총 금액: {total['total_price']:,}원, "
        f"가격 등록 상품: {total['priced_items']}종, "
        f"가격 미등록: {unpriced_text}"
    )

    catalog_schema_text = "카탈로그 스키마 정보 없음"
    catalog_summary_text = "카탈로그 요약 정보 없음"
    catalog_matches_text = "질문 관련 카탈로그 데이터 없음"
    catalog_discount_text = "할인 스냅샷 없음"
    catalog_price_text = "가격 스냅샷 없음"
    catalog_nutrition_text = "영양 스냅샷 없음"
    catalog_corner_text = "코너 매핑 정보 없음"

    if catalog_context and catalog_context.get("available"):
        catalog_schema_text = _json_for_prompt(catalog_context.get("schema") or {})
        catalog_summary_text = _json_for_prompt(catalog_context.get("summary") or {})
        catalog_matches_text = _json_for_prompt(catalog_context.get("matched_products") or [])
        catalog_discount_text = _json_for_prompt(catalog_context.get("discount_snapshot") or [])
        catalog_price_text = _json_for_prompt(catalog_context.get("price_snapshot") or {})
        catalog_nutrition_text = _json_for_prompt(catalog_context.get("nutrition_snapshot") or {})
        catalog_corner_text = _json_for_prompt(catalog_context.get("corner_map") or {})

    prompt = f"""
아래는 사용자의 장바구니 정보와 카탈로그 DB 스냅샷입니다.

상품 목록:
{cart_text}

합계:
{total_text}

카탈로그 스키마:
{catalog_schema_text}

카탈로그 요약:
{catalog_summary_text}

질문 관련 카탈로그 데이터:
{catalog_matches_text}

할인 스냅샷:
{catalog_discount_text}

가격 스냅샷:
{catalog_price_text}

영양 스냅샷:
{catalog_nutrition_text}

카테고리-코너 매핑:
{catalog_corner_text}

사용자 질문:
{q}

위 정보를 참고해서 친절하게 답변해 주세요.
- 가격이 미등록인 상품에 대해서는 '가격 정보가 DB에 아직 등록되지 않았습니다'라고 안내하세요.
- 상품 목록에 없는 제품을 물어보면 '장바구니에 해당 상품이 없습니다'라고 안내하세요.
- 장바구니가 비어있으면 '현재 장바구니가 비어있습니다. 상품을 담아주세요.'라고 안내하세요.
- 질문이 장바구니가 아닌 카탈로그(DB) 정보 관련이면 카탈로그 데이터 기준으로 답변하세요.
- 카탈로그 질문에서는 장바구니 상태를 먼저 언급하지 말고, 질문한 상품 정보부터 답변하세요.
- 영양소/칼로리 질문은 nutrition 관련 DB 컬럼(예: nutrition_info) 값을 우선 사용해 답하세요.
- 카탈로그 데이터에서 확인되지 않는 내용은 '현재 DB에서 확인되지 않습니다'라고 명확히 말하세요.
"""

    # --- HuggingFace Router API (OpenAI-compatible) ---
    hf_model = os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    hf_api_url = os.getenv(
        "HF_CHAT_API", "https://router.huggingface.co/v1/chat/completions"
    )
    hf_token = _get_hf_token()

    if not hf_token:
        return "HuggingFace LLM 토큰이 설정되지 않았습니다. 환경변수 HF_TOKEN을 등록하세요."

    try:
        resp = httpx.post(
            hf_api_url,
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": hf_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "당신은 한국어 쇼핑 도우미입니다. "
                            "반드시 제공된 DB/장바구니 정보만 근거로 답하고, "
                            "없는 정보는 없다고 말하세요."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.2,
            },
            timeout=30.0,
        )

        if resp.status_code in {404, 410}:
            return (
                f"LLM 모델({hf_model})을 찾을 수 없습니다. "
                "HF_CHAT_MODEL 환경변수를 다른 공개 모델로 바꿔주세요."
            )
        if resp.status_code == 503:
            return "LLM 모델이 아직 로딩 중입니다. 잠시 후 다시 시도해 주세요."
        if resp.status_code == 401:
            return "HF_TOKEN 인증에 실패했습니다. 토큰 값/만료 상태를 확인해 주세요."
        if resp.status_code == 403:
            return (
                "HF 토큰은 인식되었지만 Inference Providers 호출 권한이 없습니다. "
                "HuggingFace 토큰 권한에서 Inference Providers 권한을 활성화해 주세요."
            )
        if resp.status_code == 402:
            return "현재 선택한 모델은 과금이 필요할 수 있습니다. 다른 무료 모델로 변경해 주세요."

        resp.raise_for_status()
        data = resp.json()

        if (
            isinstance(data, dict)
            and isinstance(data.get("choices"), list)
            and data["choices"]
            and isinstance(data["choices"][0], dict)
            and isinstance(data["choices"][0].get("message"), dict)
        ):
            content = data["choices"][0]["message"].get("content", "")
            return str(content).strip() or "LLM 응답이 비어 있습니다."

        if "error" in data:
            return f"LLM 오류: {data['error']}"

        return "LLM 응답을 해석할 수 없습니다."

    except Exception as e:
        return f"LLM 호출 오류: {e}"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/chatbot/suggestions")
async def get_chatbot_suggestions(session_id: str | None = None):
    """Return a set of quick-action suggestion chips."""
    suggestions = [
        "과자 중에서 할인율이 제일 큰 상품이 뭐야?",
        "가장 많이 담긴 상품은 뭐야?",
        "코카콜라 위치가 어디야?",
    ]

    if session_id:
        session = app_state.session_manager.get(session_id)
        if session:
            items = list(session.state["billing_items"].keys())
            if items:
                suggestions.append(f"{items[0]} 가격 알려줘")

    return {"suggestions": suggestions[:4]}


@router.post("/chatbot/query")
async def query_chatbot(req: ChatbotRequest, db: Session = Depends(get_db)):
    """Process a natural-language question about the current cart."""
    billing_items: dict[str, int] = {}
    cart_update: dict[str, Any] | None = None

    if req.session_id:
        session = app_state.session_manager.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        billing_items = dict(session.state["billing_items"])

        if req.question.startswith(_SELECT_PREFIX):
            pending = session.state.get("chatbot_pending")
            chosen = req.question[len(_SELECT_PREFIX):].strip()
            if not pending:
                cart_update = {
                    "action": "add",
                    "item": None,
                    "quantity": 0,
                    "new_quantity": None,
                    "billing_items": billing_items,
                    "error": "선택할 상품이 없습니다.",
                }
            else:
                action = pending.get("action", "add")
                qty = int(pending.get("quantity", 1))
                session.state.pop("chatbot_pending", None)

                label = chosen
                product_name = pending.get("label_map", {}).get(chosen)

                current = billing_items.get(label, 0)
                if action == "add":
                    next_qty = current + qty
                else:
                    if current == 0:
                        cart_update = {
                            "action": action,
                            "item": product_name or label,
                            "quantity": qty,
                            "new_quantity": 0,
                            "billing_items": billing_items,
                            "error": "장바구니에 해당 상품이 없습니다.",
                        }
                        next_qty = 0
                    else:
                        next_qty = max(0, current - qty)

                if not cart_update:
                    if next_qty > 0:
                        billing_items[label] = next_qty
                    elif label in billing_items:
                        del billing_items[label]

                    session.state["billing_items"] = dict(billing_items)
                    session.state["chatbot_last_label"] = label
                    cart_update = {
                        "action": action,
                        "item": product_name or label,
                        "quantity": qty,
                        "new_quantity": next_qty,
                        "billing_items": billing_items,
                    }
        else:
            action = _detect_cart_action(req.question)
            if action == "clear":
                removed_items = dict(billing_items)
                billing_items = {}
                session.state["billing_items"] = {}
                session.state.pop("chatbot_pending", None)
                session.state.pop("chatbot_last_label", None)
                cart_update = {
                    "action": "clear",
                    "item": None,
                    "quantity": 0,
                    "new_quantity": 0,
                    "billing_items": billing_items,
                    "removed_items": removed_items,
                }
            elif action:
                qty, cleaned = _extract_quantity(req.question)
                require_in_cart = action == "remove"
                label = None
                product_name = None

                if action == "add":
                    label = _find_label_in_cart(cleaned, billing_items)
                    if label:
                        product_name = _label_display(label)
                    else:
                        label, product_name = _find_by_item_no(db, cleaned)
                else:
                    label, product_name = _find_cart_label(db, cleaned, billing_items, require_in_cart)

                if not label and action == "remove":
                    if any(token in req.question for token in _REFERENCE_TOKENS):
                        last_label = session.state.get("chatbot_last_label")
                        if last_label in billing_items:
                            label = last_label
                            product_name = _label_display(last_label)

                if label:
                    current = billing_items.get(label, 0)
                    if action == "add":
                        next_qty = current + qty
                    else:
                        next_qty = max(0, current - qty)

                    if next_qty > 0:
                        billing_items[label] = next_qty
                    elif label in billing_items:
                        del billing_items[label]

                    session.state["billing_items"] = dict(billing_items)
                    session.state["chatbot_last_label"] = label
                    cart_update = {
                        "action": action,
                        "item": product_name or label,
                        "quantity": qty,
                        "new_quantity": next_qty,
                        "billing_items": billing_items,
                    }
                else:
                    if action == "remove":
                        candidates = _find_cart_candidates(cleaned, billing_items)
                    else:
                        candidates = _find_candidate_products(db, cleaned)

                    if len(candidates) == 1:
                        chosen = candidates[0]
                        label = chosen["label"]
                        product_name = chosen["product_name"]
                        current = billing_items.get(label, 0)
                        next_qty = current + qty if action == "add" else max(0, current - qty)
                        if next_qty > 0:
                            billing_items[label] = next_qty
                        elif label in billing_items:
                            del billing_items[label]

                        session.state["billing_items"] = dict(billing_items)
                        session.state["chatbot_last_label"] = label
                        cart_update = {
                            "action": action,
                            "item": product_name,
                            "quantity": qty,
                            "new_quantity": next_qty,
                            "billing_items": billing_items,
                        }
                    elif len(candidates) > 1:
                        session.state["chatbot_pending"] = {
                            "action": action,
                            "quantity": qty,
                            "label_map": {c["label"]: c["product_name"] for c in candidates},
                        }
                        cart_update = {
                            "action": action,
                            "item": None,
                            "quantity": qty,
                            "new_quantity": None,
                            "billing_items": billing_items,
                            "candidates": candidates,
                        }
                    else:
                        error_msg = "상품을 찾을 수 없습니다."
                        if action == "remove":
                            error_msg = "장바구니에 해당 상품이 없습니다."
                        cart_update = {
                            "action": action,
                            "item": None,
                            "quantity": qty,
                            "new_quantity": None,
                            "billing_items": billing_items,
                            "error": error_msg,
                        }
            else:
                label, _ = _find_cart_label(db, req.question, billing_items, require_in_cart=True)
                if label:
                    session.state["chatbot_last_label"] = label

    products = _build_cart_meta(db, billing_items)
    total = _totals(products)
    if cart_update and cart_update.get("error"):
        answer = str(cart_update.get("error") or "해당 상품을 찾지 못했어요. 상품명을 조금 더 정확히 입력해 주세요.")
    elif cart_update and cart_update.get("action") == "clear":
        removed_items = cart_update.get("removed_items") or {}
        if removed_items:
            answer = "현재 장바구니를 비웠습니다."
        else:
            answer = "현재 장바구니가 이미 비어있습니다."
    elif cart_update and cart_update.get("candidates"):
        answer = "비슷한 상품이 여러 개 있어요. 아래에서 하나를 선택해 주세요."
    elif cart_update:
        verb = "추가" if cart_update["action"] == "add" else "제거"
        item_name = cart_update["item"] or "상품"
        answer = (
            f"{item_name} {cart_update['quantity']}개를 {verb}했습니다. "
            f"현재 총 수량은 {total['total_count']}개이고 총 금액은 {total['total_price']:,}원입니다."
        )
    else:
        catalog_context = _build_catalog_context(db, req.question)
        answer = _answer_question(req.question, products, total, catalog_context)

    return {
        "answer": answer,
        "cart": {
            "items": [p.model_dump() for p in products],
            **total,
        },
        "cart_update": cart_update,
    }
