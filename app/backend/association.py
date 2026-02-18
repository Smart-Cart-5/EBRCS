"""Hand-product association helpers."""

from __future__ import annotations

from typing import Any


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _center_distance(a: list[float], b: list[float]) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    dx = acx - bcx
    dy = acy - bcy
    return (dx * dx + dy * dy) ** 0.5


def associate_hands_products(
    hands: list[dict[str, Any]],
    products: list[dict[str, Any]],
    *,
    iou_weight: float = 0.5,
    dist_weight: float = 0.5,
    max_center_dist: float = 0.35,
    min_score: float = 0.1,
) -> list[dict[str, Any]]:
    """Greedy one-to-one hand-product association using IoU and center distance."""
    if not hands or not products:
        return []

    candidates: list[tuple[float, int, int, float, float]] = []
    for h_idx, hand in enumerate(hands):
        hbox = hand.get("box")
        if not hbox:
            continue
        for p_idx, product in enumerate(products):
            pbox = product.get("box")
            if not pbox:
                continue
            iou = _box_iou(hbox, pbox)
            center_dist = _center_distance(hbox, pbox)
            dist_score = max(0.0, 1.0 - (center_dist / max(1e-6, max_center_dist)))
            score = iou_weight * iou + dist_weight * dist_score
            if score >= min_score:
                candidates.append((score, h_idx, p_idx, iou, center_dist))

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_hands: set[int] = set()
    matched_products: set[int] = set()
    matches: list[dict[str, Any]] = []

    for score, h_idx, p_idx, iou, center_dist in candidates:
        if h_idx in matched_hands or p_idx in matched_products:
            continue
        matched_hands.add(h_idx)
        matched_products.add(p_idx)
        matches.append(
            {
                "hand_idx": h_idx,
                "product_idx": p_idx,
                "score": float(score),
                "iou": float(iou),
                "center_dist": float(center_dist),
                "hand_box": hands[h_idx]["box"],
                "product_box": products[p_idx]["box"],
            }
        )

    return matches
