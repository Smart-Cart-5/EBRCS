from __future__ import annotations

from collections.abc import MutableMapping
import logging
import os
import time
from typing import Any

import cv2
import numpy as np

from checkout_core.counting import should_count_product
from checkout_core.inference import build_query_embedding

try:
    from backend import config as backend_config
    from backend.association import associate_hands_products
    from backend.event_engine import AddEventEngine
    from backend.snapshots import SnapshotBuffer
    from backend.utils.profiler import ProfileCollector
except Exception:  # pragma: no cover - optional in non-backend contexts
    backend_config = None
    associate_hands_products = None
    AddEventEngine = None
    SnapshotBuffer = None
    ProfileCollector = None


logger = logging.getLogger("checkout_core.frame_processor")
_FRAME_PROFILER = None


def _get_frame_profiler():
    global _FRAME_PROFILER
    if backend_config is None or ProfileCollector is None:
        return None
    if not getattr(backend_config, "ENABLE_PROFILING", False):
        return None
    if _FRAME_PROFILER is None:
        _FRAME_PROFILER = ProfileCollector(
            kind="frame",
            enable=backend_config.ENABLE_PROFILING,
            every_n_frames=getattr(backend_config, "PROFILE_EVERY_N_FRAMES", 30),
            logger=logger,
        )
    return _FRAME_PROFILER.start_frame()


def _reset_candidate_votes(state: MutableMapping[str, Any]) -> None:
    state["candidate_votes"] = {}
    state["candidate_history"] = []
    state["topk_candidates"] = []
    state["confidence"] = 0.0


def _update_candidate_votes(
    state: MutableMapping[str, Any],
    top1_label: str | None,
    vote_window_size: int,
) -> tuple[list[str], dict[str, int]]:
    history = state.setdefault("candidate_history", [])
    if top1_label:
        history.append(str(top1_label))
    max_len = max(1, int(vote_window_size))
    if len(history) > max_len:
        del history[:-max_len]

    votes: dict[str, int] = {}
    for label in history:
        votes[label] = int(votes.get(label, 0)) + 1

    state["candidate_votes"] = votes
    return history, votes


def _build_topk_response(scores: dict[str, float], top_k: int) -> list[dict[str, float]]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_k))]
    return [
        {
            "label": label,
            "score": float(score),
            "raw_score": float(score),
            "percent_score": _score_to_percent(float(score)),
        }
        for label, score in ranked
    ]


def _compute_confidence(topk_candidates: list[dict[str, float]]) -> float:
    if not topk_candidates:
        return 0.0
    top1 = float(topk_candidates[0]["score"])
    top2 = float(topk_candidates[1]["score"]) if len(topk_candidates) > 1 else 0.0
    if top2 <= 1e-6:
        return top1
    return top1 / top2


def _score_to_percent(score: float) -> float:
    return float(score) * 100.0


def _apply_unknown_decision(
    state: MutableMapping[str, Any],
    *,
    name_candidate: str | None,
    topk_candidates: list[dict[str, float]],
    frame_id: int | None,
    session_id: str | None,
) -> tuple[str, bool, str | None, float, float | None]:
    score_th = float(getattr(backend_config, "UNKNOWN_SCORE_THRESHOLD", 0.45))
    gap_th = float(getattr(backend_config, "UNKNOWN_GAP_THRESHOLD", 0.02))
    stable_need = max(1, int(getattr(backend_config, "STABLE_RESULT_FRAMES", 3)))
    high_conf_th = float(getattr(backend_config, "HIGH_CONFIDENCE_THRESHOLD", 0.65))
    high_conf_th_occluded = float(getattr(backend_config, "HIGH_CONFIDENCE_THRESHOLD_OCCLUDED", max(0.65, high_conf_th + 0.1)))
    occluded_by_hand = bool(state.get("occluded_by_hand", False))
    iou_max = float(state.get("overlap_hand_iou_max", 0.0))
    effective_high_conf_th = high_conf_th_occluded if occluded_by_hand else high_conf_th

    if not topk_candidates:
        result_label = "UNKNOWN"
        is_unknown = True
        reason = "no_candidate"
        top1_raw = 0.0
        gap = None
        gap_reason = "no_top2"
        stable_count = 0
        top1_label = "UNKNOWN"
        top2_label = None
        top2_raw = None
        decision = "UNKNOWN"
        decision_reason = reason
        state["_stable_result_label"] = None
        state["_stable_result_count"] = 0
    else:
        top1 = topk_candidates[0]
        top2 = topk_candidates[1] if len(topk_candidates) > 1 else None
        top1_label = str(top1.get("label", "UNKNOWN"))
        top1_raw = float(top1.get("raw_score", top1.get("score", 0.0)))
        top2_label = str(top2.get("label", "UNKNOWN")) if top2 is not None else None
        top2_raw = float(top2.get("raw_score", top2.get("score", 0.0))) if top2 is not None else None
        gap = (top1_raw - top2_raw) if top2_raw is not None else None
        gap_reason = "top2_present" if top2_raw is not None else "no_top2"

        prev_label = state.get("_stable_result_label")
        prev_count = int(state.get("_stable_result_count", 0))
        stable_count = (prev_count + 1) if prev_label == top1_label else 1
        state["_stable_result_label"] = top1_label
        state["_stable_result_count"] = stable_count

        decision = "UNKNOWN"
        decision_reason = "score_below_threshold"
        if top1_raw > effective_high_conf_th:
            decision = "CONFIRMED"
            decision_reason = "high_confidence_fast_track_occluded" if occluded_by_hand else "high_confidence_fast_track"
        elif top1_raw < score_th:
            decision = "UNKNOWN"
            decision_reason = "score_below_threshold"
        elif gap is not None and gap < gap_th:
            decision = "UNKNOWN"
            decision_reason = "gap_below_threshold"
        elif stable_count < stable_need:
            decision = "UNKNOWN"
            decision_reason = "unstable_label"
        else:
            decision = "CONFIRMED"
            decision_reason = "stable_label_confirmed"

        is_unknown = decision != "CONFIRMED"
        reason = decision_reason
        result_label = "UNKNOWN" if is_unknown else top1_label

    if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
        top2_raw_log = f"{top2_raw:.4f}" if top2_raw is not None else "-"
        gap_log = f"{gap:.4f}" if gap is not None else "-"
        logger.info(
            "Unknown decision: session=%s frame_id=%s top1_label=%s top1_raw=%.4f top2_label=%s top2_raw=%s gap=%s gap_reason=%s stable_count=%d occluded_by_hand=%s iou_max=%.4f thresholds(score=%.4f,gap=%.4f,stable=%d,high=%.4f,high_occ=%.4f,high_eff=%.4f) decision=%s decision_reason=%s",
            session_id,
            frame_id,
            top1_label,
            top1_raw,
            top2_label or "-",
            top2_raw_log,
            gap_log,
            gap_reason,
            stable_count,
            occluded_by_hand,
            iou_max,
            score_th,
            gap_th,
            stable_need,
            high_conf_th,
            high_conf_th_occluded,
            effective_high_conf_th,
            decision,
            decision_reason,
        )

    state["result_label"] = result_label
    state["is_unknown"] = bool(is_unknown)
    state["match_score_raw"] = float(top1_raw)
    state["match_top2_raw"] = float(top2_raw) if top2_raw is not None else None
    # NOTE: percent_score is similarity*100 scale, not classification accuracy.
    state["match_score_percent"] = _score_to_percent(top1_raw)
    state["match_gap"] = float(gap) if gap is not None else None
    state["match_gap_reason"] = gap_reason
    state["unknown_reason"] = reason
    return result_label, is_unknown, reason, float(top1_raw), (float(gap) if gap is not None else None)


def _metric_name(faiss_index) -> str:
    try:
        import faiss  # type: ignore

        metric = getattr(faiss_index, "metric_type", None)
        if metric == faiss.METRIC_INNER_PRODUCT:
            return "IP"
        if metric == faiss.METRIC_L2:
            return "L2"
    except Exception:
        pass
    return "UNKNOWN"


def _get_db_norm_stats(
    faiss_index,
    state: MutableMapping[str, Any],
    sample_size: int,
) -> tuple[float, float] | None:
    cache_key = (id(faiss_index), int(getattr(faiss_index, "ntotal", 0)), int(getattr(faiss_index, "d", 0)))
    cached = state.get("_db_norm_stats_cache")
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return cached.get("value")

    ntotal = int(getattr(faiss_index, "ntotal", 0))
    if ntotal <= 0:
        return None
    count = max(1, min(ntotal, int(sample_size)))
    try:
        vectors = np.stack([faiss_index.reconstruct(i) for i in range(count)], axis=0).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        stats = (float(np.mean(norms)), float(np.std(norms)))
        state["_db_norm_stats_cache"] = {"key": cache_key, "value": stats}
        return stats
    except Exception:
        return None


def _should_debug_log(state: MutableMapping[str, Any], interval_ms: int, key: str = "_search_debug_last_log_ms") -> bool:
    now_ms = int(time.time() * 1000)
    last_ms = int(state.get(key, 0))
    if now_ms - last_ms < max(0, int(interval_ms)):
        return False
    state[key] = now_ms
    return True


def _match_with_voting(
    *,
    crop: np.ndarray,
    model_bundle,
    faiss_index,
    labels,
    state: MutableMapping[str, Any],
    match_threshold: float,
    faiss_top_k: int,
    vote_window_size: int,
    vote_min_samples: int,
    frame_profiler,
    box_area_ratio: float | None = None,
    frame_id: int | None = None,
    session_id: str | None = None,
) -> tuple[str | None, float, list[dict[str, float]], float]:
    if frame_profiler is not None:
        with frame_profiler.measure("embed"):
            emb = build_query_embedding(crop, model_bundle)
    else:
        emb = build_query_embedding(crop, model_bundle)

    query = np.expand_dims(emb, axis=0)
    query_norm = float(np.linalg.norm(query[0]))
    effective_top_k = max(5, int(faiss_top_k))
    search_k = max(1, min(int(effective_top_k), int(faiss_index.ntotal)))
    if frame_profiler is not None:
        with frame_profiler.measure("faiss"):
            distances, indices = faiss_index.search(query, search_k)
    else:
        distances, indices = faiss_index.search(query, search_k)

    debug_enabled = bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False))
    if debug_enabled and _should_debug_log(
        state,
        int(getattr(backend_config, "SEARCH_DEBUG_LOG_INTERVAL_MS", 5000)),
    ):
        metric = _metric_name(faiss_index)
        db_stats = _get_db_norm_stats(
            faiss_index,
            state,
            int(getattr(backend_config, "SEARCH_DEBUG_DB_NORM_SAMPLE", 2048)),
        )
        sim_min = float(np.min(distances[0])) if distances.size > 0 else 0.0
        sim_max = float(np.max(distances[0])) if distances.size > 0 else 0.0
        sim_range = "-1~1" if sim_min < 0 else "0~1"
        crop_h, crop_w = crop.shape[:2]
        logger.info(
            "Search debug: session=%s frame_id=%s faiss_index=%s metric=%s query_norm=%.4f db_norm_mean=%.4f db_norm_std=%.4f sim_range=%s sim_min=%.4f sim_max=%.4f crop=%dx%d box_area_ratio=%.4f",
            session_id,
            frame_id,
            type(faiss_index).__name__,
            metric,
            query_norm,
            db_stats[0] if db_stats is not None else -1.0,
            db_stats[1] if db_stats is not None else -1.0,
            sim_range,
            sim_min,
            sim_max,
            crop_w,
            crop_h,
            float(box_area_ratio or 0.0),
        )

    if bool(getattr(backend_config, "SEARCH_DEBUG_SAVE_CROP", False)):
        now_ms = int(time.time() * 1000)
        last_save = int(state.get("_search_debug_last_crop_save_ms", 0))
        if now_ms - last_save >= max(0, int(getattr(backend_config, "SEARCH_DEBUG_SAVE_INTERVAL_MS", 1000))):
            state["_search_debug_last_crop_save_ms"] = now_ms
            crop_dir = str(getattr(backend_config, "SEARCH_DEBUG_CROP_DIR", "debug_crops"))
            try:
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, f"crop_{now_ms}.jpg")
                cv2.imwrite(crop_path, crop)
            except Exception:
                logger.exception("Failed to save debug crop")

    def _postprocess_scores() -> tuple[str | None, float, list[dict[str, float]], float]:
        current_scores: dict[str, float] = {}
        for idx, score in zip(indices[0], distances[0]):
            label_idx = int(idx)
            if label_idx < 0 or label_idx >= len(labels):
                continue
            name = str(labels[label_idx])
            current_scores[name] = max(current_scores.get(name, -1e9), float(score))

        if debug_enabled and current_scores:
            ranked_debug = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            debug_topk = [
                {
                    "label": label,
                    "raw_score": float(raw),
                    "percent_score": _score_to_percent(float(raw)),
                }
                for label, raw in ranked_debug
            ]
            logger.info("Search debug topk(raw): session=%s frame_id=%s topk=%s", session_id, frame_id, debug_topk)

        topk_candidates = _build_topk_response(current_scores, search_k)
        confidence = _compute_confidence(topk_candidates)
        state["topk_candidates"] = topk_candidates
        state["confidence"] = confidence

        if not topk_candidates:
            _save_search_crop_if_needed(
                crop,
                state=state,
                session_id=session_id,
                frame_id=frame_id,
                top1_label="UNKNOWN",
                top1_raw=0.0,
            )
            return None, 0.0, topk_candidates, confidence

        top1_label = str(topk_candidates[0]["label"])
        top1_raw = float(topk_candidates[0]["raw_score"])
        _save_search_crop_if_needed(
            crop,
            state=state,
            session_id=session_id,
            frame_id=frame_id,
            top1_label=top1_label,
            top1_raw=top1_raw,
        )
        history, vote_counts = _update_candidate_votes(state, top1_label, vote_window_size)

        voted_label = max(
            vote_counts.items(),
            key=lambda x: (
                int(x[1]),
                1 if x[0] == top1_label else 0,
                float(current_scores.get(x[0], -1e9)),
            ),
        )[0] if vote_counts else top1_label

        enough_samples = len(history) >= max(1, int(vote_min_samples))
        if enough_samples and voted_label == top1_label and top1_raw >= match_threshold:
            return top1_label, top1_raw, topk_candidates, confidence
        return None, top1_raw, topk_candidates, confidence

    if frame_profiler is not None:
        with frame_profiler.measure("voting_postprocess"):
            return _postprocess_scores()
    return _postprocess_scores()


def _classify_snapshots_with_votes(
    *,
    crops: list[np.ndarray],
    model_bundle,
    faiss_index,
    labels,
    match_threshold: float,
    faiss_top_k: int,
    frame_profiler,
) -> tuple[str | None, float, list[dict[str, float]], float]:
    if not crops:
        return None, 0.0, [], 0.0

    label_history: list[str] = []
    vote_counts: dict[str, int] = {}
    last_scores: dict[str, float] = {}
    last_topk_candidates: list[dict[str, float]] = []
    effective_top_k = max(5, int(faiss_top_k))
    search_k = max(1, min(int(effective_top_k), int(faiss_index.ntotal)))
    for crop in crops:
        if frame_profiler is not None:
            with frame_profiler.measure("embed"):
                emb = build_query_embedding(crop, model_bundle)
        else:
            emb = build_query_embedding(crop, model_bundle)
        query = np.expand_dims(emb, axis=0)
        if frame_profiler is not None:
            with frame_profiler.measure("faiss"):
                distances, indices = faiss_index.search(query, search_k)
        else:
            distances, indices = faiss_index.search(query, search_k)

        current_scores: dict[str, float] = {}
        for idx, score in zip(indices[0], distances[0]):
            label_idx = int(idx)
            if label_idx < 0 or label_idx >= len(labels):
                continue
            name = str(labels[label_idx])
            current_scores[name] = max(current_scores.get(name, -1e9), float(score))

        if not current_scores:
            continue

        last_scores = current_scores
        last_topk_candidates = _build_topk_response(current_scores, search_k)
        if not last_topk_candidates:
            continue
        top1_label = str(last_topk_candidates[0]["label"])
        label_history.append(top1_label)
        vote_counts[top1_label] = int(vote_counts.get(top1_label, 0)) + 1

    def _finalize() -> tuple[str | None, float, list[dict[str, float]], float]:
        topk_candidates = last_topk_candidates
        confidence = _compute_confidence(topk_candidates)
        if not topk_candidates:
            return None, 0.0, [], confidence

        top1_label = str(topk_candidates[0]["label"])
        top1_raw = float(topk_candidates[0]["raw_score"])
        voted_label = max(
            vote_counts.items(),
            key=lambda x: (
                int(x[1]),
                1 if x[0] == top1_label else 0,
                float(last_scores.get(x[0], -1e9)),
            ),
        )[0] if vote_counts else top1_label

        enough_samples = len(label_history) >= 1
        if enough_samples and voted_label == top1_label and top1_raw >= match_threshold:
            return top1_label, top1_raw, topk_candidates, confidence
        return None, top1_raw, topk_candidates, confidence

    if frame_profiler is not None:
        with frame_profiler.measure("voting_postprocess"):
            return _finalize()
    return _finalize()


def _select_remove_label(state: MutableMapping[str, Any]) -> str | None:
    billing_items = state.get("billing_items", {})
    if not billing_items:
        return None

    sequence = state.get("in_cart_sequence", [])
    while sequence:
        candidate = sequence[-1]
        if int(billing_items.get(candidate, 0)) > 0:
            return candidate
        sequence.pop()

    ranked = sorted(billing_items.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return None
    return str(ranked[0][0])


def _box_iou(box_a, box_b) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _roi_overlap_ratio(box, roi_poly: np.ndarray | None, frame_shape: tuple[int, ...]) -> float:
    if roi_poly is None or len(roi_poly) < 3:
        return 0.0
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    box_area = max(0.0, (x2 - x1) * w) * max(0.0, (y2 - y1) * h)
    if box_area <= 1e-6:
        return 0.0

    rect = np.array(
        [[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]],
        dtype=np.float32,
    )
    try:
        inter_area, _ = cv2.intersectConvexConvex(roi_poly.astype(np.float32), rect)
        inter = max(0.0, float(inter_area))
    except Exception:
        inter = 0.0
    return inter / box_area


def _roi_iou(box, roi_poly: np.ndarray | None, frame_shape: tuple[int, ...]) -> float:
    if roi_poly is None or len(roi_poly) < 3:
        return 0.0
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    rect = np.array(
        [[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]],
        dtype=np.float32,
    )
    box_area = max(0.0, (x2 - x1) * w) * max(0.0, (y2 - y1) * h)
    roi_area = abs(float(cv2.contourArea(roi_poly.astype(np.float32))))
    if box_area <= 1e-6 or roi_area <= 1e-6:
        return 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(roi_poly.astype(np.float32), rect)
        inter = max(0.0, float(inter_area))
    except Exception:
        inter = 0.0
    union = box_area + roi_area - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def _mask_hand_regions_in_crop(
    crop: np.ndarray,
    *,
    product_box: list[float],
    hand_boxes: list[list[float]],
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    h, w = frame_shape[:2]
    px1 = int(product_box[0] * w)
    py1 = int(product_box[1] * h)
    px2 = int(product_box[2] * w)
    py2 = int(product_box[3] * h)
    crop_w = max(1, px2 - px1)
    crop_h = max(1, py2 - py1)
    masked = crop.copy()
    for hb in hand_boxes:
        hx1 = int(hb[0] * w)
        hy1 = int(hb[1] * h)
        hx2 = int(hb[2] * w)
        hy2 = int(hb[3] * h)
        ix1 = max(px1, hx1)
        iy1 = max(py1, hy1)
        ix2 = min(px2, hx2)
        iy2 = min(py2, hy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        cx1 = max(0, min(crop_w, ix1 - px1))
        cy1 = max(0, min(crop_h, iy1 - py1))
        cx2 = max(0, min(crop_w, ix2 - px1))
        cy2 = max(0, min(crop_h, iy2 - py1))
        if cx2 > cx1 and cy2 > cy1:
            masked[cy1:cy2, cx1:cx2] = 0
    return masked


def _expand_search_box(
    box: list[float],
    frame_shape: tuple[int, ...],
    *,
    target_min_side: int,
    pad_ratio: float,
    edge_pad_ratio: float,
) -> tuple[list[float], float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1_px = x1 * w
    y1_px = y1 * h
    x2_px = x2 * w
    y2_px = y2 * h

    bw = max(1.0, x2_px - x1_px)
    bh = max(1.0, y2_px - y1_px)
    cx = (x1_px + x2_px) * 0.5
    cy = (y1_px + y2_px) * 0.5

    new_w = bw * (1.0 + max(0.0, float(pad_ratio)))
    new_h = bh * (1.0 + max(0.0, float(pad_ratio)))
    min_side = float(max(1, int(target_min_side)))
    cur_min = min(new_w, new_h)
    if cur_min < min_side:
        scale = min_side / max(1.0, cur_min)
        new_w *= scale
        new_h *= scale

    edge_px = float(max(0.0, edge_pad_ratio) * min(w, h))
    edge_touch = bool(x1_px <= edge_px or y1_px <= edge_px or x2_px >= (w - edge_px) or y2_px >= (h - edge_px))
    if edge_touch:
        new_w *= 1.12
        new_h *= 1.12

    nx1 = max(0.0, cx - (new_w * 0.5))
    ny1 = max(0.0, cy - (new_h * 0.5))
    nx2 = min(float(w), cx + (new_w * 0.5))
    ny2 = min(float(h), cy + (new_h * 0.5))

    effective_padding_ratio = max((new_w / max(1.0, bw)) - 1.0, (new_h / max(1.0, bh)) - 1.0)
    return ([
        float(nx1 / max(1.0, float(w))),
        float(ny1 / max(1.0, float(h))),
        float(nx2 / max(1.0, float(w))),
        float(ny2 / max(1.0, float(h))),
    ], float(effective_padding_ratio))


def _pad_crop_for_embedding(
    crop: np.ndarray,
    *,
    target_min_side: int,
    aspect_min_hw: float,
    aspect_max_hw: float,
) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    target_h = max(int(h), int(target_min_side))
    target_w = max(int(w), int(target_min_side))

    aspect_min_hw = max(0.1, float(aspect_min_hw))
    aspect_max_hw = max(aspect_min_hw, float(aspect_max_hw))
    hw_ratio = target_h / max(1, target_w)
    if hw_ratio < aspect_min_hw:
        target_h = int(np.ceil(aspect_min_hw * target_w))
    elif hw_ratio > aspect_max_hw:
        target_w = int(np.ceil(target_h / max(1e-6, aspect_max_hw)))

    pad_top = max(0, (target_h - h) // 2)
    pad_bottom = max(0, target_h - h - pad_top)
    pad_left = max(0, (target_w - w) // 2)
    pad_right = max(0, target_w - w - pad_left)
    if pad_top <= 0 and pad_bottom <= 0 and pad_left <= 0 and pad_right <= 0:
        return crop
    return cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )


def _parse_debug_frame_ids(raw: str) -> set[int]:
    frame_ids: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            frame_ids.add(int(token))
        except ValueError:
            continue
    return frame_ids


def _save_search_crop_if_needed(
    crop: np.ndarray,
    *,
    state: MutableMapping[str, Any],
    session_id: str | None,
    frame_id: int | None,
    top1_label: str,
    top1_raw: float,
) -> None:
    if not bool(getattr(backend_config, "SEARCH_DEBUG_SAVE_SEARCH_CROP", False)):
        return
    save_frame_ids = state.get("_search_debug_save_frame_ids")
    if not isinstance(save_frame_ids, set):
        save_frame_ids = _parse_debug_frame_ids(getattr(backend_config, "SEARCH_DEBUG_SAVE_FRAME_IDS", ""))
        state["_search_debug_save_frame_ids"] = save_frame_ids

    low_score_th = float(getattr(backend_config, "SEARCH_DEBUG_SAVE_LOW_SCORE_THRESHOLD", 0.35))
    forced_frame = frame_id is not None and int(frame_id) in save_frame_ids
    low_score = float(top1_raw) < low_score_th
    if not (forced_frame or low_score):
        return
    if crop is None or crop.size == 0:
        return

    crop_dir = str(getattr(backend_config, "SEARCH_DEBUG_CROP_DIR", "debug_crops"))
    os.makedirs(crop_dir, exist_ok=True)
    safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id or "na"))
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(top1_label or "UNKNOWN"))
    safe_frame = int(frame_id) if frame_id is not None else -1
    reason = "frameid" if forced_frame else "lowraw"
    out_name = f"search_{safe_session}_f{safe_frame}_{safe_label}_{float(top1_raw):.4f}_{reason}.jpg"
    out_path = os.path.join(crop_dir, out_name)
    try:
        cv2.imwrite(out_path, crop)
    except Exception:
        logger.exception("Failed to save search crop: %s", out_path)


def _record_search_debug_stats(
    state: MutableMapping[str, Any],
    *,
    frame_id: int | None = None,
    session_id: str | None = None,
) -> None:
    now = time.time()
    interval = state.setdefault("_search_debug_interval", {"start_ts": now, "search_count": 0, "skip_count": 0, "reasons": {}})
    if not isinstance(interval, dict):
        interval = {"start_ts": now, "search_count": 0, "skip_count": 0, "reasons": {}}
        state["_search_debug_interval"] = interval

    did_search = bool(state.get("did_search", False))
    reason = str(state.get("skip_reason", "unknown"))
    if did_search:
        interval["search_count"] = int(interval.get("search_count", 0)) + 1
    else:
        interval["skip_count"] = int(interval.get("skip_count", 0)) + 1

    reasons = interval.setdefault("reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
        interval["reasons"] = reasons
    reasons[reason] = int(reasons.get(reason, 0)) + 1

    start_ts = float(interval.get("start_ts", now))
    elapsed = now - start_ts
    if elapsed >= 5.0:
        logger.info(
            "Search gating stats(%.1fs): session=%s last_frame_id=%s search_count=%d skip_count=%d reasons=%s",
            elapsed,
            session_id,
            frame_id,
            int(interval.get("search_count", 0)),
            int(interval.get("skip_count", 0)),
            dict(sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)),
        )
        interval["start_ts"] = now
        interval["search_count"] = 0
        interval["skip_count"] = 0
        interval["reasons"] = {}


def _update_last_result_cache(
    state: MutableMapping[str, Any],
    *,
    name: str | None,
    score: float,
    topk_candidates: list[dict[str, float]],
    confidence: float,
) -> None:
    if not topk_candidates:
        return
    cached_name = name if name is not None else str(topk_candidates[0].get("label", ""))
    state["last_result_name"] = cached_name
    state["last_result_score"] = float(score)
    state["last_result_topk"] = topk_candidates
    state["last_result_confidence"] = float(confidence)
    state["last_result_at_ms"] = int(time.time() * 1000)


def create_bg_subtractor():
    return cv2.createBackgroundSubtractorKNN(
        history=300,
        dist2Threshold=500,
        detectShadows=False,
    )


def process_checkout_frame(
    *,
    frame: np.ndarray,
    frame_count: int,
    frame_id: int | None = None,
    session_id: str | None = None,
    bg_subtractor,
    model_bundle,
    faiss_index,
    labels,
    state: MutableMapping[str, Any],
    min_area: int,
    detect_every_n_frames: int,
    match_threshold: float,
    cooldown_seconds: float,
    roi_poly: np.ndarray | None = None,
    roi_clear_frames: int = 8,
    roi_entry_mode: bool = False,
    min_product_bbox_area: int = 2500,
    max_products_per_frame: int = 3,
    faiss_top_k: int = 3,
    vote_window_size: int = 5,
    vote_min_samples: int = 3,
    search_every_n_frames: int = 10,
    min_box_area_ratio: float = 0.05,
    stable_frames_for_search: int = 5,
    search_cooldown_ms: int = 1500,
    roi_box_min_overlap: float = 0.2,
    product_conf_min: float = 0.65,
    product_min_area_ratio: float = 0.06,
    product_aspect_ratio_min: float = 0.35,
    product_aspect_ratio_max: float = 3.0,
    product_max_height_ratio: float = 0.95,
    product_max_width_ratio: float = 0.95,
    product_edge_touch_eps: float = 0.01,
    min_search_area_ratio: float = 0.12,
    min_crop_size: int = 224,
    search_select_w_conf: float = 0.4,
    search_select_w_area: float = 0.4,
    search_select_w_roi: float = 0.2,
    roi_iou_min: float = 0.15,
    roi_center_pass: bool = True,
    hand_conf_min: float = 0.5,
    hand_overlap_iou: float = 0.25,
    search_mask_hand: bool = False,
    warp_on: bool = False,
    frame_profiler_override=None,
    yolo_detector=None,
) -> np.ndarray:
    """Process a single frame and update checkout state in-place.

    Args:
        yolo_detector: Optional YOLODetector instance. If provided, uses YOLO for detection
                      instead of background subtraction.
    """
    owns_profiler = frame_profiler_override is None
    frame_profiler = frame_profiler_override or _get_frame_profiler()
    total_start = time.perf_counter() if frame_profiler is not None else 0.0
    display_frame = frame.copy()

    # Initialize detection_boxes in state if not present
    if "detection_boxes" not in state:
        state["detection_boxes"] = []
    if "best_pair" not in state:
        state["best_pair"] = None
    state["did_search"] = False
    state["skip_reason"] = "pending"

    # Choose detection method: YOLO or background subtraction
    if yolo_detector is not None:
        if warp_on and bool(getattr(backend_config, "WARP_DEBUG_LOG", False)):
            if _should_debug_log(
                state,
                int(getattr(backend_config, "WARP_DEBUG_LOG_INTERVAL_MS", 1000)),
                key="_warp_debug_last_log_ms",
            ):
                h, w = frame.shape[:2]
                min_px = int(np.min(frame)) if frame.size else 0
                max_px = int(np.max(frame)) if frame.size else 0
                mean_px = float(np.mean(frame)) if frame.size else 0.0
                roi_bounds = None
                if roi_poly is not None and len(roi_poly) >= 3:
                    xs = roi_poly[:, 0]
                    ys = roi_poly[:, 1]
                    roi_bounds = (
                        int(np.min(xs)),
                        int(np.min(ys)),
                        int(np.max(xs)),
                        int(np.max(ys)),
                    )
                logger.info(
                    "Warp YOLO input: session=%s frame_id=%s warp_on=%s shape=%s dtype=%s min=%d max=%d mean=%.2f roi_bounds=%s",
                    session_id,
                    frame_id,
                    warp_on,
                    (h, w, frame.shape[2] if frame.ndim == 3 else 1),
                    str(frame.dtype),
                    min_px,
                    max_px,
                    mean_px,
                    roi_bounds,
                )

        if warp_on and bool(getattr(backend_config, "WARP_DEBUG_SAVE", False)):
            now_ms = int(time.time() * 1000)
            last_save_ms = int(state.get("_warp_debug_last_save_ms", 0))
            if now_ms - last_save_ms >= 1000:
                state["_warp_debug_last_save_ms"] = now_ms
                out_dir = str(getattr(backend_config, "WARP_DEBUG_DIR", "warp_debug"))
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(out_dir, f"warp_input_s{session_id}_f{frame_id}_{now_ms}.jpg"),
                        frame,
                    )
                except Exception:
                    logger.exception("Failed to save warp debug frame")

        # YOLO-based detection
        if frame_profiler is not None:
            with frame_profiler.measure("yolo_infer"):
                detections = yolo_detector.detect(frame)
        else:
            detections = yolo_detector.detect(frame)
        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1.0, float(frame_h * frame_w))
        reject_stats: dict[str, int] = {
            "conf_low": 0,
            "too_small": 0,
            "aspect": 0,
            "too_tall": 0,
            "too_wide": 0,
            "edge_touch": 0,
            "roi_out": 0,
            "hand_conf_low": 0,
        }
        roi_eval_debug: list[dict[str, float | bool]] = []
        filtered_hands: list[dict[str, Any]] = []
        filtered_products: list[dict[str, Any]] = []

        for det in detections:
            cls = det.get("class")
            conf = float(det.get("confidence", 0.0))
            box = det.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue

            if cls == "hand":
                if conf < float(hand_conf_min):
                    reject_stats["hand_conf_low"] += 1
                    continue
                filtered_hands.append(det)
                continue

            if cls != "product":
                continue
            if conf < float(product_conf_min):
                reject_stats["conf_low"] += 1
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            bw_px = max(0.0, (x2 - x1) * frame_w)
            bh_px = max(0.0, (y2 - y1) * frame_h)
            bbox_area_px = bw_px * bh_px
            area_ratio = bbox_area_px / frame_area
            box_w_ratio = max(0.0, float(x2 - x1))
            box_h_ratio = max(0.0, float(y2 - y1))
            if bbox_area_px < max(1, int(min_product_bbox_area)) or area_ratio < float(product_min_area_ratio):
                reject_stats["too_small"] += 1
                continue

            if box_h_ratio > float(product_max_height_ratio):
                reject_stats["too_tall"] += 1
                continue
            if box_w_ratio > float(product_max_width_ratio):
                reject_stats["too_wide"] += 1
                continue
            if float(y1) <= float(product_edge_touch_eps) and float(y2) >= (1.0 - float(product_edge_touch_eps)):
                reject_stats["edge_touch"] += 1
                continue

            aspect = bw_px / max(1e-6, bh_px)
            if aspect < float(product_aspect_ratio_min) or aspect > float(product_aspect_ratio_max):
                reject_stats["aspect"] += 1
                continue

            if roi_poly is not None:
                cx = (x1 + x2) * 0.5 * frame_w
                cy = (y1 + y2) * 0.5 * frame_h
                inside_by_center = bool(cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0)
                roi_iou = _roi_iou(box, roi_poly, frame.shape)
                passed_roi = bool(roi_iou >= float(roi_iou_min) or (bool(roi_center_pass) and inside_by_center))
                if len(roi_eval_debug) < 8:
                    roi_eval_debug.append(
                        {
                            "cx": round(float(cx), 1),
                            "cy": round(float(cy), 1),
                            "inside_roi_by_center": inside_by_center,
                            "roi_iou": round(float(roi_iou), 4),
                            "passed_roi": passed_roi,
                        }
                    )
                if not passed_roi:
                    reject_stats["roi_out"] += 1
                    continue

            filtered_products.append(det)

        state["detection_boxes"] = filtered_hands + filtered_products
        hands = filtered_hands
        all_products = filtered_products

        if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)) and _should_debug_log(
            state,
            int(getattr(backend_config, "SEARCH_DEBUG_LOG_INTERVAL_MS", 5000)),
            key="_yolo_filter_debug_last_log_ms",
        ):
            raw_products = len([d for d in detections if d.get("class") == "product"])
            raw_hands = len([d for d in detections if d.get("class") == "hand"])
            logger.info(
                "YOLO filter: session=%s frame_id=%s warp_on=%s raw_total=%d raw_products=%d raw_hands=%d pass_products=%d pass_hands=%d reject=%s roi_eval=%s",
                session_id,
                frame_id,
                warp_on,
                len(detections),
                raw_products,
                raw_hands,
                len(filtered_products),
                len(filtered_hands),
                reject_stats,
                roi_eval_debug,
            )

        if associate_hands_products is not None:
            assoc_matches = associate_hands_products(
                hands,
                all_products,
                iou_weight=getattr(backend_config, "ASSOCIATION_IOU_WEIGHT", 0.5),
                dist_weight=getattr(backend_config, "ASSOCIATION_DIST_WEIGHT", 0.5),
                max_center_dist=getattr(backend_config, "ASSOCIATION_MAX_CENTER_DIST", 0.35),
                min_score=getattr(backend_config, "ASSOCIATION_MIN_SCORE", 0.1),
            )
        else:
            assoc_matches = []

        if assoc_matches:
            state["best_pair"] = assoc_matches[0]
            hb = assoc_matches[0]["hand_box"]
            pb = assoc_matches[0]["product_box"]
            h, w = frame.shape[:2]
            hcx = int(((hb[0] + hb[2]) * 0.5) * w)
            hcy = int(((hb[1] + hb[3]) * 0.5) * h)
            pcx = int(((pb[0] + pb[2]) * 0.5) * w)
            pcy = int(((pb[1] + pb[3]) * 0.5) * h)
            cv2.line(display_frame, (hcx, hcy), (pcx, pcy), (255, 180, 0), 2)
        else:
            state["best_pair"] = None

        event_mode_enabled = bool(getattr(backend_config, "EVENT_MODE", False))
        if event_mode_enabled and AddEventEngine is not None and SnapshotBuffer is not None:
            event_engine = state.get("_event_engine")
            if event_engine is None:
                event_engine = AddEventEngine(
                    t_grasp_min_frames=getattr(backend_config, "T_GRASP_MIN_FRAMES", 4),
                    t_place_stable_frames=getattr(backend_config, "T_PLACE_STABLE_FRAMES", 12),
                    t_remove_confirm_frames=getattr(backend_config, "T_REMOVE_CONFIRM_FRAMES", 45),
                    roi_hysteresis_inset_ratio=getattr(backend_config, "ROI_HYSTERESIS_INSET_RATIO", 0.05),
                    roi_hysteresis_outset_ratio=getattr(backend_config, "ROI_HYSTERESIS_OUTSET_RATIO", 0.05),
                )
                state["_event_engine"] = event_engine

            snapshot_buffer = state.get("_snapshot_buffer")
            if snapshot_buffer is None:
                snapshot_buffer = SnapshotBuffer(max_frames=getattr(backend_config, "SNAPSHOT_MAX_FRAMES", 8))
                state["_snapshot_buffer"] = snapshot_buffer

            event_update = event_engine.update(
                best_pair=state.get("best_pair"),
                products=all_products,
                roi_poly=roi_poly,
                frame_shape=frame.shape,
            )
            state["event_state"] = event_update.state
            state["last_status"] = event_update.status

            if event_update.track_box is not None and event_update.state in (event_engine.GRASP, event_engine.PLACE_CHECK):
                snapshot_buffer.add(frame, event_update.track_box)
            elif event_update.state == event_engine.IDLE:
                snapshot_buffer.clear()

            if event_update.add_confirmed and faiss_index is not None and faiss_index.ntotal > 0:
                state["did_search"] = True
                state["skip_reason"] = "searched"
                crops = snapshot_buffer.best_crops(limit=getattr(backend_config, "SNAPSHOT_MAX_FRAMES", 8))
                name, best_score, topk_candidates, confidence = _classify_snapshots_with_votes(
                    crops=crops,
                    model_bundle=model_bundle,
                    faiss_index=faiss_index,
                    labels=labels,
                    match_threshold=match_threshold,
                    faiss_top_k=faiss_top_k,
                    frame_profiler=frame_profiler,
                )
                state["topk_candidates"] = topk_candidates
                state["confidence"] = confidence
                _update_last_result_cache(
                    state,
                    name=name,
                    score=best_score,
                    topk_candidates=topk_candidates,
                    confidence=confidence,
                )
                result_label, is_unknown, _, top1_raw, _ = _apply_unknown_decision(
                    state,
                    name_candidate=name,
                    topk_candidates=topk_candidates,
                    frame_id=frame_id,
                    session_id=session_id,
                )

                if not is_unknown:
                    state["last_label"] = result_label
                    state["last_score"] = top1_raw
                    state["last_status"] = "ADD 확정"
                    state.setdefault("item_scores", {})[result_label] = top1_raw
                    billing_items = state.setdefault("billing_items", {})
                    billing_items[result_label] = int(billing_items.get(result_label, 0)) + 1
                    state["_last_confirmed_ms"] = int(time.time() * 1000)
                    state.setdefault("in_cart_sequence", []).append(result_label)
                else:
                    state["last_label"] = "UNKNOWN"
                    state["last_score"] = top1_raw
                    state["last_status"] = "ADD 미확정(UNKNOWN)"
                snapshot_buffer.clear()
            elif event_update.add_confirmed:
                state["skip_reason"] = "faiss_unavailable"
            else:
                state["skip_reason"] = "event_not_confirmed"

            if event_update.remove_confirmed:
                remove_label = _select_remove_label(state)
                billing_items = state.setdefault("billing_items", {})
                if remove_label and int(billing_items.get(remove_label, 0)) > 0:
                    billing_items[remove_label] = int(billing_items.get(remove_label, 0)) - 1
                    if billing_items[remove_label] <= 0:
                        billing_items.pop(remove_label, None)
                    sequence = state.setdefault("in_cart_sequence", [])
                    for i in range(len(sequence) - 1, -1, -1):
                        if sequence[i] == remove_label:
                            sequence.pop(i)
                            break
                    state["last_label"] = remove_label
                    state["last_score"] = 0.0
                    state["last_status"] = "REMOVE 확정"
                else:
                    state["last_status"] = "REMOVE 미확정"

            # In EVENT_MODE, defer embedding/FAISS to event confirmation only.
            if roi_poly is not None:
                cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)
            if frame_profiler is not None:
                frame_profiler.add_ms("total", (time.perf_counter() - total_start) * 1000.0)
                if owns_profiler:
                    frame_profiler.finish()
            _record_search_debug_stats(state, frame_id=frame_id, session_id=session_id)
            return display_frame

        # Event-gated search: choose exactly one product candidate for search.
        product_detections = list(all_products)

        candidate = None
        has_too_small = False
        has_conf_low_search = False
        has_crop_small = False
        has_outside_roi = False
        state["occluded_by_hand"] = False
        state["overlap_hand_iou_max"] = 0.0
        state["search_crop_min_side_before"] = None
        state["search_crop_min_side_after"] = None
        state["search_crop_padding_ratio"] = 0.0
        if roi_poly is None:
            state["skip_reason"] = "roi_not_set"
            state["last_status"] = "ROI 미설정"
        elif not product_detections:
            state["skip_reason"] = "no_product_detected"
            state["last_status"] = "미탐지"
            _reset_candidate_votes(state)
        else:
            ranked_candidates: list[dict[str, Any]] = []
            for detection in product_detections:
                box = detection["box"]
                conf = float(detection.get("confidence", 0.0))
                if conf < float(product_conf_min):
                    has_conf_low_search = True
                    continue
                bw = max(0.0, (box[2] - box[0]) * frame_w)
                bh = max(0.0, (box[3] - box[1]) * frame_h)
                bbox_area_px = bw * bh
                area_ratio = bbox_area_px / frame_area
                min_ratio = max(float(min_box_area_ratio), float(product_min_area_ratio))
                if bbox_area_px < max(1, int(min_product_bbox_area)) or area_ratio < min_ratio:
                    has_too_small = True
                    continue
                crop_min_side = min(int(round(bw)), int(round(bh)))
                if crop_min_side < max(1, int(min_crop_size)):
                    has_crop_small = True

                x1, y1, x2, y2 = box
                cx = (x1 + x2) * 0.5 * frame_w
                cy = (y1 + y2) * 0.5 * frame_h
                inside_roi = bool(cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0)
                roi_iou_val = _roi_iou(box, roi_poly, frame.shape)
                overlap_ratio = _roi_overlap_ratio(box, roi_poly, frame.shape)
                if not inside_roi and overlap_ratio < float(roi_box_min_overlap):
                    has_outside_roi = True
                    continue

                ranked_candidates.append(
                    {
                        "detection": detection,
                        "inside_roi": inside_roi,
                        "conf": conf,
                        "area_ratio": float(area_ratio),
                        "roi_iou": float(roi_iou_val),
                        "crop_min_side": int(crop_min_side),
                        "needs_padding": bool(crop_min_side < max(1, int(min_crop_size))),
                        "selection_score": (
                            float(search_select_w_conf) * conf
                            + float(search_select_w_area) * float(area_ratio)
                            + float(search_select_w_roi) * float(roi_iou_val)
                        ),
                    }
                )

            ranked_candidates.sort(key=lambda x: float(x["selection_score"]), reverse=True)

            if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)) and _should_debug_log(
                state,
                int(getattr(backend_config, "SEARCH_DEBUG_LOG_INTERVAL_MS", 5000)),
                key="_candidate_select_debug_last_log_ms",
            ):
                summary = [
                    {
                        "conf": round(float(c["conf"]), 4),
                        "area_ratio": round(float(c["area_ratio"]), 4),
                        "inside_roi": bool(c["inside_roi"]),
                        "roi_iou": round(float(c["roi_iou"]), 4),
                        "crop_min_side": int(c["crop_min_side"]),
                        "needs_padding": bool(c["needs_padding"]),
                        "selection_score": round(float(c["selection_score"]), 4),
                    }
                    for c in ranked_candidates[:5]
                ]
                logger.info(
                    "Search candidate ranking: session=%s frame_id=%s weights=(conf=%.2f,area=%.2f,roi=%.2f) candidates=%s",
                    session_id,
                    frame_id,
                    float(search_select_w_conf),
                    float(search_select_w_area),
                    float(search_select_w_roi),
                    summary,
                )

            if ranked_candidates:
                # If multiple candidates exist, prioritize the largest area among conf-qualified boxes.
                if len(ranked_candidates) >= 2:
                    max_area = max(float(c["area_ratio"]) for c in ranked_candidates)
                    area_priority = [c for c in ranked_candidates if float(c["area_ratio"]) >= max_area - 1e-9]
                    chosen = max(area_priority, key=lambda c: float(c["selection_score"]))
                    choose_reason = "largest_area_priority_then_score"
                else:
                    chosen = ranked_candidates[0]
                    choose_reason = "single_candidate_by_score"
                detection = chosen["detection"]
                candidate = (detection, float(chosen["area_ratio"]))
                if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)) and _should_debug_log(
                    state,
                    int(getattr(backend_config, "SEARCH_DEBUG_LOG_INTERVAL_MS", 5000)),
                    key="_candidate_chosen_debug_last_log_ms",
                ):
                    logger.info(
                        "Search candidate chosen: session=%s frame_id=%s reason=%s selection_score=%.4f conf=%.4f area_ratio=%.4f roi_iou=%.4f crop_min_side=%d inside_roi=%s box=%s",
                        session_id,
                        frame_id,
                        choose_reason,
                        float(chosen["selection_score"]),
                        float(chosen["conf"]),
                        float(chosen["area_ratio"]),
                        float(chosen["roi_iou"]),
                        int(chosen["crop_min_side"]),
                        bool(chosen["inside_roi"]),
                        detection.get("box"),
                    )

            if candidate is None:
                if has_conf_low_search:
                    state["skip_reason"] = "conf_low_search"
                    state["last_status"] = "신뢰도 낮음"
                elif has_crop_small:
                    state["skip_reason"] = "crop_too_small"
                    state["last_status"] = "크롭 작음"
                elif has_too_small:
                    state["skip_reason"] = "box_too_small"
                    state["last_status"] = "박스 작음"
                elif has_outside_roi:
                    state["skip_reason"] = "outside_roi"
                    state["last_status"] = "ROI 외부"
                else:
                    state["skip_reason"] = "no_eligible_product"
                    state["last_status"] = "후보 없음"
                if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
                    logger.info(
                        "Search skip: session=%s frame_id=%s reason=%s occluded_by_hand=%s iou_max=%.4f crop_min_side_before=%s crop_min_side_after=%s padding_ratio=%.4f",
                        session_id,
                        frame_id,
                        state["skip_reason"],
                        bool(state.get("occluded_by_hand", False)),
                        float(state.get("overlap_hand_iou_max", 0.0)),
                        state.get("search_crop_min_side_before"),
                        state.get("search_crop_min_side_after"),
                        float(state.get("search_crop_padding_ratio", 0.0)),
                    )

        if candidate is None:
            state["_search_track_box"] = None
            state["_search_track_count"] = 0
            state["_search_track_frame"] = frame_count
        else:
            detection, area_ratio = candidate
            box = detection["box"]
            prev_box = state.get("_search_track_box")
            prev_count = int(state.get("_search_track_count", 0))
            prev_frame = int(state.get("_search_track_frame", -1))
            contiguous = frame_count == (prev_frame + 1)
            iou_ok = _box_iou(box, prev_box) >= 0.5
            stable_count = (prev_count + 1) if (contiguous and iou_ok) else 1
            state["_search_track_box"] = list(box)
            state["_search_track_count"] = stable_count
            state["_search_track_frame"] = frame_count

            state["search_track_stable"] = bool(stable_count >= max(1, int(stable_frames_for_search)))
            now_ms = int(time.time() * 1000)
            last_search_ms = int(state.get("_last_search_ms", 0))
            last_confirmed_ms = int(state.get("_last_confirmed_ms", 0))
            post_confirm_window_ms = max(0, int(getattr(backend_config, "SEARCH_POST_CONFIRM_WINDOW_MS", 2500)))
            preconfirm_cooldown_ms = max(0, int(getattr(backend_config, "SEARCH_COOLDOWN_MS_PRECONFIRM", 50)))
            strong_cooldown_ms = max(0, int(search_cooldown_ms))
            confirmed_recently = last_confirmed_ms > 0 and (now_ms - last_confirmed_ms) <= post_confirm_window_ms
            cooldown_ms = strong_cooldown_ms if confirmed_recently else preconfirm_cooldown_ms
            if now_ms - last_search_ms < cooldown_ms:
                state["skip_reason"] = "cooldown"
                state["last_status"] = "쿨다운"
                if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
                    logger.info(
                        "Search skip: session=%s frame_id=%s reason=%s occluded_by_hand=%s iou_max=%.4f crop_min_side_before=%s crop_min_side_after=%s padding_ratio=%.4f confirmed_recently=%s cooldown_ms=%d",
                        session_id,
                        frame_id,
                        state["skip_reason"],
                        bool(state.get("occluded_by_hand", False)),
                        float(state.get("overlap_hand_iou_max", 0.0)),
                        state.get("search_crop_min_side_before"),
                        state.get("search_crop_min_side_after"),
                        float(state.get("search_crop_padding_ratio", 0.0)),
                        confirmed_recently,
                        int(cooldown_ms),
                    )
            elif faiss_index is None or faiss_index.ntotal <= 0:
                state["skip_reason"] = "faiss_unavailable"
                state["last_status"] = "검색 불가"
                if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
                    logger.info(
                        "Search skip: session=%s frame_id=%s reason=%s occluded_by_hand=%s iou_max=%.4f crop_min_side_before=%s crop_min_side_after=%s padding_ratio=%.4f",
                        session_id,
                        frame_id,
                        state["skip_reason"],
                        bool(state.get("occluded_by_hand", False)),
                        float(state.get("overlap_hand_iou_max", 0.0)),
                        state.get("search_crop_min_side_before"),
                        state.get("search_crop_min_side_after"),
                        float(state.get("search_crop_padding_ratio", 0.0)),
                    )
            else:
                bw = max(0.0, (box[2] - box[0]) * frame_w)
                bh = max(0.0, (box[3] - box[1]) * frame_h)
                crop_min_side_before = int(min(round(bw), round(bh)))
                state["search_crop_min_side_before"] = crop_min_side_before
                expanded_box, padding_ratio = _expand_search_box(
                    box,
                    frame.shape,
                    target_min_side=int(getattr(backend_config, "SEARCH_CROP_MIN_SIDE", 320)),
                    pad_ratio=float(getattr(backend_config, "SEARCH_CROP_PAD_RATIO", 0.20)),
                    edge_pad_ratio=float(getattr(backend_config, "SEARCH_CROP_EDGE_PAD_RATIO", 0.02)),
                )
                state["search_crop_padding_ratio"] = float(padding_ratio)
                crop = yolo_detector.extract_crop(frame, expanded_box)
                if crop is None:
                    state["skip_reason"] = "crop_failed"
                    state["last_status"] = "크롭 실패"
                    if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
                        logger.info(
                            "Search skip: session=%s frame_id=%s reason=%s occluded_by_hand=%s iou_max=%.4f crop_min_side_before=%s crop_min_side_after=%s padding_ratio=%.4f",
                            session_id,
                            frame_id,
                            state["skip_reason"],
                            bool(state.get("occluded_by_hand", False)),
                            float(state.get("overlap_hand_iou_max", 0.0)),
                            state.get("search_crop_min_side_before"),
                            state.get("search_crop_min_side_after"),
                            float(state.get("search_crop_padding_ratio", 0.0)),
                        )
                else:
                    crop_min_side_after = int(min(crop.shape[0], crop.shape[1]))
                    state["search_crop_min_side_after"] = crop_min_side_after
                    min_after_expand = max(1, int(getattr(backend_config, "SEARCH_CROP_MIN_SIDE_AFTER_EXPAND", 240)))
                    if crop_min_side_after < min_after_expand:
                        state["skip_reason"] = "crop_too_small_after_expand"
                        state["last_status"] = "크롭 작음(확장후)"
                        if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
                            logger.info(
                                "Search skip: session=%s frame_id=%s reason=%s occluded_by_hand=%s iou_max=%.4f crop_min_side_before=%s crop_min_side_after=%s padding_ratio=%.4f",
                                session_id,
                                frame_id,
                                state["skip_reason"],
                                bool(state.get("occluded_by_hand", False)),
                                float(state.get("overlap_hand_iou_max", 0.0)),
                                state.get("search_crop_min_side_before"),
                                state.get("search_crop_min_side_after"),
                                float(state.get("search_crop_padding_ratio", 0.0)),
                            )
                        crop = None
                    if crop is None:
                        pass
                    else:
                        overlap_hands: list[list[float]] = []
                        max_hand_iou = 0.0
                        for hand in hands:
                            hbox = hand.get("box")
                            if not isinstance(hbox, list) or len(hbox) != 4:
                                continue
                            iou = _box_iou(box, hbox)
                            if iou > max_hand_iou:
                                max_hand_iou = iou
                            if iou > float(hand_overlap_iou):
                                overlap_hands.append(hbox)
                        state["overlap_hand_iou_max"] = float(max_hand_iou)
                        state["occluded_by_hand"] = bool(overlap_hands)

                        if overlap_hands and bool(search_mask_hand):
                            crop = _mask_hand_regions_in_crop(
                                crop,
                                product_box=expanded_box,
                                hand_boxes=overlap_hands,
                                frame_shape=frame.shape,
                            )
                            state["_overlap_hand_last_iou"] = float(max_hand_iou)
                        crop = _pad_crop_for_embedding(
                            crop,
                            target_min_side=int(getattr(backend_config, "SEARCH_CROP_MIN_SIDE", 320)),
                            aspect_min_hw=float(getattr(backend_config, "SEARCH_CROP_ASPECT_MIN", 0.5)),
                            aspect_max_hw=float(getattr(backend_config, "SEARCH_CROP_ASPECT_MAX", 2.0)),
                        )

                        state["did_search"] = True
                        state["skip_reason"] = "searched"
                        state["_last_search_ms"] = now_ms
                        if bool(getattr(backend_config, "SEARCH_DEBUG_LOG", False)):
                            logger.info(
                                "Search run: session=%s frame_id=%s occluded_by_hand=%s iou_max=%.4f crop_min_side_before=%s crop_min_side_after=%s padding_ratio=%.4f",
                                session_id,
                                frame_id,
                                bool(state.get("occluded_by_hand", False)),
                                float(state.get("overlap_hand_iou_max", 0.0)),
                                state.get("search_crop_min_side_before"),
                                state.get("search_crop_min_side_after"),
                                float(state.get("search_crop_padding_ratio", 0.0)),
                            )
                        name, best_score, topk_candidates, confidence = _match_with_voting(
                            crop=crop,
                            model_bundle=model_bundle,
                            faiss_index=faiss_index,
                            labels=labels,
                            state=state,
                            match_threshold=match_threshold,
                            faiss_top_k=faiss_top_k,
                            vote_window_size=vote_window_size,
                            vote_min_samples=vote_min_samples,
                            frame_profiler=frame_profiler,
                            box_area_ratio=float(area_ratio),
                            frame_id=frame_id,
                            session_id=session_id,
                        )
                        detection["topk_candidates"] = topk_candidates
                        detection["confidence"] = confidence
                        _update_last_result_cache(
                            state,
                            name=name,
                            score=best_score,
                            topk_candidates=topk_candidates,
                            confidence=confidence,
                        )
                        result_label, is_unknown, _, top1_raw, _ = _apply_unknown_decision(
                            state,
                            name_candidate=name,
                            topk_candidates=topk_candidates,
                            frame_id=frame_id,
                            session_id=session_id,
                        )

                        if not is_unknown:
                            detection["label"] = result_label
                            detection["score"] = top1_raw
                            state["last_label"] = result_label
                            state["last_score"] = top1_raw
                            state["last_status"] = "매칭됨"
                            state.setdefault("item_scores", {})[result_label] = top1_raw
                            last_seen_at = state.setdefault("last_seen_at", {})
                            can_count = should_count_product(
                                last_seen_at,
                                result_label,
                                cooldown_seconds=cooldown_seconds,
                            )
                            if can_count:
                                billing_items = state.setdefault("billing_items", {})
                                billing_items[result_label] = int(billing_items.get(result_label, 0)) + 1
                                state["_last_confirmed_ms"] = now_ms
                        else:
                            state["last_label"] = "UNKNOWN"
                            state["last_score"] = top1_raw
                            state["last_status"] = "매칭 실패(UNKNOWN)"

    else:
        # Fallback: Background subtraction (original logic)
        state["detection_boxes"] = []  # No YOLO detections
        state["best_pair"] = None
        state["skip_reason"] = "yolo_not_available"

        if frame_profiler is not None:
            with frame_profiler.measure("detect"):
                fg_mask = bg_subtractor.apply(frame)
                fg_mask = cv2.erode(fg_mask, None, iterations=2)
                fg_mask = cv2.dilate(fg_mask, None, iterations=4)
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.erode(fg_mask, None, iterations=2)
            fg_mask = cv2.dilate(fg_mask, None, iterations=4)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        if roi_poly is not None:
            roi_mask = np.zeros_like(fg_mask)
            cv2.fillPoly(roi_mask, [roi_poly], 255)
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if False and candidates and faiss_index is not None and faiss_index.ntotal > 0:
            state["last_status"] = "탐지됨"
            main_cnt = max(candidates, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_cnt)

            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + 2 * pad)
            y2 = min(frame.shape[0], y + h + 2 * pad)

            w = x2 - x
            h = y2 - y

            if w > 20 and h > 20:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                entry_event = False
                inside_roi = False
                if roi_poly is not None:
                    cx = x + (w / 2)
                    cy = y + (h / 2)
                    inside_roi = cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0

                    if inside_roi:
                        state["roi_empty_frames"] = 0
                        entry_event = not bool(state.get("roi_occupied", False))
                        state["roi_occupied"] = True
                    else:
                        state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1

                if roi_poly is not None and int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                    state["roi_occupied"] = False

                crop = frame[y:y + h, x:x + w]
                area_ratio_bg = float((w * h) / max(1, frame.shape[0] * frame.shape[1]))

                if roi_poly is not None and roi_entry_mode:
                    periodic_slot = frame_count % max(1, int(search_every_n_frames)) == 0
                    allow_inference = inside_roi and (entry_event or periodic_slot)
                    if inside_roi:
                        state["last_status"] = "ROI 진입" if entry_event else "ROI 내부"
                    else:
                        state["last_status"] = "ROI 외부"
                else:
                    allow_inference = False
                    if roi_poly is None:
                        state["last_status"] = "ROI 미설정"

                if allow_inference:
                    state["did_search"] = True
                    name, best_score, topk_candidates, confidence = _match_with_voting(
                        crop=crop,
                        model_bundle=model_bundle,
                        faiss_index=faiss_index,
                        labels=labels,
                        state=state,
                        match_threshold=match_threshold,
                        faiss_top_k=faiss_top_k,
                        vote_window_size=vote_window_size,
                        vote_min_samples=vote_min_samples,
                        frame_profiler=frame_profiler,
                        box_area_ratio=area_ratio_bg,
                        frame_id=frame_id,
                        session_id=session_id,
                    )
                    _update_last_result_cache(
                        state,
                        name=name,
                        score=best_score,
                        topk_candidates=topk_candidates,
                        confidence=confidence,
                    )
                    result_label, is_unknown, _, top1_raw, _ = _apply_unknown_decision(
                        state,
                        name_candidate=name,
                        topk_candidates=topk_candidates,
                        frame_id=frame_id,
                        session_id=session_id,
                    )
                    if not is_unknown:
                        label = f"{result_label} ({top1_raw:.3f})"

                        state["last_label"] = result_label
                        state["last_score"] = top1_raw
                        state["last_status"] = "매칭됨"
                        state.setdefault("item_scores", {})[result_label] = top1_raw

                        cv2.putText(
                            display_frame,
                            label,
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        last_seen_at = state.setdefault("last_seen_at", {})
                        can_count = should_count_product(
                            last_seen_at,
                            result_label,
                            cooldown_seconds=cooldown_seconds,
                        )
                        if can_count:
                            billing_items = state.setdefault("billing_items", {})
                            billing_items[result_label] = int(billing_items.get(result_label, 0)) + 1
                    else:
                        state["last_label"] = "UNKNOWN"
                        state["last_score"] = top1_raw
                        state["last_status"] = "매칭 실패(UNKNOWN)"
                        state["topk_candidates"] = topk_candidates
        else:
            _reset_candidate_votes(state)
            if roi_poly is not None and bool(state.get("roi_occupied", False)):
                state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1
                if int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                    state["roi_occupied"] = False

            state["last_label"] = "-"
            state["last_score"] = 0.0
            state["last_status"] = "미탐지"

    # Draw ROI polygon
    if roi_poly is not None:
        cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

    if frame_profiler is not None:
        frame_profiler.add_ms("total", (time.perf_counter() - total_start) * 1000.0)
        if owns_profiler:
            frame_profiler.finish()
    _record_search_debug_stats(state, frame_id=frame_id, session_id=session_id)

    return display_frame
