"""Chatbot router – LLM-powered cart assistant using products/product_prices DB.

Uses HuggingFace Router API (OpenAI-compatible) for natural-language answers.
Pulls price data from the same products + product_prices tables used by billing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
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



def _get_hf_token() -> str | None:
    """Resolve HF token from env first, then PROJECT_ROOT/.env fallback."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()

    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return None

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
            return value

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


# ---------------------------------------------------------------------------
# LLM answer generator
# ---------------------------------------------------------------------------

def _answer_question(
    question: str,
    products: list[ProductMeta],
    total: dict[str, Any],
) -> str:
    q = question.strip()
    if not q:
        return "질문을 입력해 주세요. 예: '총 금액 얼마야?'"

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

    prompt = f"""
아래는 사용자의 장바구니 상품 목록과 가격 정보입니다.

상품 목록:
{cart_text}

합계:
{total_text}

사용자 질문:
{q}

위 정보를 참고해서 친절하게 답변해 주세요.
- 가격이 미등록인 상품에 대해서는 '가격 정보가 DB에 아직 등록되지 않았습니다'라고 안내하세요.
- 상품 목록에 없는 제품을 물어보면 '장바구니에 해당 상품이 없습니다'라고 안내하세요.
- 장바구니가 비어있으면 '현재 장바구니가 비어있습니다. 상품을 담아주세요.'라고 안내하세요.
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
        if resp.status_code in {401, 403}:
            return "HF_TOKEN 인증에 실패했습니다. 토큰 권한/만료 상태를 확인해 주세요."
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
        "지금 장바구니 총 금액은 얼마야?",
        "가장 많이 담긴 상품은 뭐야?",
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
        answer = _answer_question(req.question, products, total)

    return {
        "answer": answer,
        "cart": {
            "items": [p.model_dump() for p in products],
            **total,
        },
        "cart_update": cart_update,
    }
