from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import MutableMapping
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from checkout_core.counting import should_count_track
from checkout_core.inference import build_query_embedding

try:
    from checkout_core.tracker import ObjectTracker
except ImportError:
    ObjectTracker = None


def _calculate_movement_direction(
    centroid_history: list[tuple[float, float]],
    add_direction: str = "down",  # "down" = y 증가(위→아래 진입) = 담기
    min_movement: float = 0.05,
    min_history_points: int = 6,
) -> str:
    """Y축 기준으로만 방향을 판정하여 'add', 'remove', 'unknown'을 반환.

    카메라 좌표계: y=0이 화면 상단, y가 증가할수록 아래
    - add_direction="down": y 증가(위→아래 이동, 카트에 넣는 모션) → "add"
    - 반대(아래→위, 카트에서 꺼내는 모션) → "remove"
    - 이동량 < min_movement 또는 히스토리 부족 → "unknown" (카운트 보류)
    """
    min_history_points = max(2, int(min_history_points))
    if len(centroid_history) < min_history_points:
        # 히스토리가 너무 짧으면 방향 판정 불가 → 보류
        return "unknown"

    n = max(2, len(centroid_history) // 4)
    early_y = sum(p[1] for p in centroid_history[:n]) / n
    late_y = sum(p[1] for p in centroid_history[-n:]) / n

    dy = late_y - early_y  # 양수 = 아래로 이동, 음수 = 위로 이동

    if abs(dy) < min_movement:
        return "unknown"

    # add_direction="down": dy > 0 = 아래로 이동 = 담기
    # add_direction="up":   dy < 0 = 위로 이동 = 담기
    if add_direction == "down":
        return "add" if dy > 0 else "remove"
    else:  # "up"
        return "add" if dy < 0 else "remove"


def _vertical_sign_consistency_ratio(
    centroid_history: list[tuple[float, float]],
    *,
    action: str,
    add_direction: str,
    min_step_epsilon: float = 0.003,
) -> float:
    """연속 프레임 y 증감 부호가 기대 방향과 얼마나 일관적인지 [0,1]로 반환."""
    if len(centroid_history) < 2:
        return 0.0

    eps = max(0.0, float(min_step_epsilon))
    deltas: list[float] = []
    for i in range(1, len(centroid_history)):
        dy = centroid_history[i][1] - centroid_history[i - 1][1]
        if abs(dy) >= eps:
            deltas.append(dy)

    if not deltas:
        return 0.0

    # add_direction="down": add => dy>0, remove => dy<0
    # add_direction="up":   add => dy<0, remove => dy>0
    if add_direction == "down":
        expected_positive = action == "add"
    else:
        expected_positive = action == "remove"

    matches = 0
    for dy in deltas:
        if (dy > 0) == expected_positive:
            matches += 1

    return matches / len(deltas)


def _release_counted_track(
    counted_tracks: MutableMapping[int | str, str],
    *,
    product_name: str,
    track_id: int | None,
) -> None:
    """REMOVE 성공 시 Track 잠금을 해제해서 같은 물체의 재투입(ADD)을 허용."""
    if track_id is not None:
        counted_tracks.pop(track_id, None)
        return

    # Track ID 매칭 실패 시에도 같은 상품으로 등록된 Track 1개를 해제
    for existing_track_id, existing_product in list(counted_tracks.items()):
        if existing_product == product_name:
            counted_tracks.pop(existing_track_id, None)
            break


def create_bg_subtractor():
    return cv2.createBackgroundSubtractorKNN(
        history=300,
        dist2Threshold=500,
        detectShadows=False,
    )


def _get_or_init_track_state(
    state: MutableMapping[str, Any], track_id: int | str
) -> tuple[str, dict[str, Any]]:
    track_states = state.setdefault("track_states", {})
    key = str(track_id)
    ts = track_states.get(key)
    if ts is None:
        ts = {
            "centroid_history": [],
            "label_window": [],
            "stable_label": "",
            "last_match_frame": -1,
            "direction_committed": False,
            "age": 0,
            "missing_frames": 0,
            "last_seen_frame": -1,
        }
        track_states[key] = ts
    return key, ts


def _update_track_label_consensus(
    track_state: MutableMapping[str, Any],
    label: str,
    *,
    frame_count: int,
    label_window: int,
    label_min_votes: int,
) -> None:
    window = track_state.setdefault("label_window", [])
    window.append(label)
    max_len = max(1, int(label_window))
    if len(window) > max_len:
        del window[:-max_len]

    top_label, top_votes = Counter(window).most_common(1)[0]
    if top_votes >= max(1, int(label_min_votes)):
        track_state["stable_label"] = top_label

    if track_state.get("stable_label", "") == label:
        track_state["last_match_frame"] = frame_count


def _update_track_lifecycle(
    state: MutableMapping[str, Any],
    *,
    active_track_ids: set[str],
    soft_reentry_frames: int,
    track_state_ttl_frames: int,
) -> None:
    track_states = state.setdefault("track_states", {})
    counted_tracks = state.setdefault("counted_tracks", {})
    soft_reset = max(1, int(soft_reentry_frames))
    ttl = max(soft_reset + 1, int(track_state_ttl_frames))

    for key, ts in list(track_states.items()):
        if key in active_track_ids:
            ts["missing_frames"] = 0
            continue

        missing = int(ts.get("missing_frames", 0)) + 1
        ts["missing_frames"] = missing

        if missing >= soft_reset:
            ts["direction_committed"] = False
            ts["centroid_history"] = []
            ts["stable_label"] = ""
            ts["label_window"] = []
            ts["last_match_frame"] = -1

        if missing > ttl:
            # stale track id lock도 함께 제거
            counted_tracks.pop(key, None)
            try:
                counted_tracks.pop(int(key), None)
            except (ValueError, TypeError):
                pass
            track_states.pop(key, None)


def process_checkout_frame(
    *,
    frame: np.ndarray,
    frame_count: int,
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
    tracker: ObjectTracker | None = None,
    use_tracking: bool = False,
    ignore_labels: set[str] | None = None,
    add_direction: str = "down",
    direction_min_movement: float = 0.05,
    direction_history_frames: int = 20,
    direction_min_history_points: int = 6,
    inline_direction_min_movement: float = 0.08,
    inline_direction_min_history_points: int = 8,
    direction_sign_consistency: float = 0.7,
    direction_sign_epsilon: float = 0.003,
    fast_decision_min_movement: float = 0.12,
    fast_decision_min_history_points: int = 4,
    fast_decision_sign_consistency: float = 0.85,
    infer_burst_track_age: int = 6,
    track_min_age_frames: int = 6,
    track_label_window: int = 6,
    track_label_min_votes: int = 2,
    track_state_ttl_frames: int = 18,
    soft_reentry_frames: int = 5,
    opposite_action_cooldown_seconds: float = 1.5,
    label_stale_frames: int = 4,
    warmup_frames: int = 24,
) -> np.ndarray:
    """Process a single frame and update checkout state in-place.

    기존 인식 로직(배경차분 → 가장 큰 contour → 임베딩 → FAISS)을 그대로 유지.
    DeepSORT가 활성화되면 백그라운드에서 모든 bbox를 트래킹하고,
    가장 큰 contour에 매칭된 Track ID로 중복 카운트를 방지합니다.
    """
    display_frame = frame.copy()

    # 카운트 이벤트 초기화 (매 프레임 리셋)
    state["count_event"] = None
    state["current_track_id"] = None
    counting_enabled = frame_count > max(0, int(warmup_frames))

    # Warm-up 중에는 잔상 학습만 수행하고 카운트 상태는 초기화해 오검출을 억제
    if not counting_enabled:
        state["centroid_history"] = []
        state["direction_committed"] = False
        state["last_matched_label"] = ""
        state["last_match_frame"] = -1
        state["last_active_track_id"] = None
        state["track_states"] = {}

    # ── 1. 배경차분 + contour 탐지 ──
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

    # ── 2. DeepSORT 업데이트 (빈 detection 포함) ──
    tracks: list[dict[str, Any]] = []
    if use_tracking and tracker is not None:
        all_bboxes = []
        for cnt in candidates:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            pad = 10
            bx = max(0, bx - pad)
            by = max(0, by - pad)
            bx2 = min(frame.shape[1], bx + bw + 2 * pad)
            by2 = min(frame.shape[0], by + bh + 2 * pad)
            bw, bh = bx2 - bx, by2 - by
            if bw > 20 and bh > 20:
                all_bboxes.append((bx, by, bw, bh))
        try:
            tracks = tracker.update(frame, all_bboxes)
        except Exception:
            logger.exception("tracker update failed")
            tracks = []

    active_track_ids = {str(t["track_id"]) for t in tracks}
    _update_track_lifecycle(
        state,
        active_track_ids=active_track_ids,
        soft_reentry_frames=soft_reentry_frames,
        track_state_ttl_frames=track_state_ttl_frames,
    )

    # ── 3. 가장 큰 contour 중심 처리 + 트랙 상태 누적 ──
    if candidates and faiss_index is not None and faiss_index.ntotal > 0:
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
            track_id: int | None = None
            track_state: dict[str, Any] | None = None

            if tracks and tracker is not None:
                track_id = tracker.find_track_for_bbox(tracks, (x, y, w, h))
                state["current_track_id"] = track_id
                if track_id is not None:
                    _, track_state = _get_or_init_track_state(state, track_id)
                    track_state["age"] = int(track_state.get("age", 0)) + 1
                    track_state["last_seen_frame"] = frame_count
                    track_state["missing_frames"] = 0
                    state["last_active_track_id"] = track_id

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ROI 상태 갱신
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
                    if entry_event and track_state is not None:
                        # 새 진입에서는 이 트랙의 이전 궤적/라벨을 리셋
                        track_state["centroid_history"] = []
                        track_state["direction_committed"] = False
                        track_state["stable_label"] = ""
                        track_state["label_window"] = []
                        track_state["last_match_frame"] = -1
                        state["last_matched_label"] = ""
                        state["last_match_frame"] = -1
                else:
                    state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1

            if roi_poly is not None and int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                state["roi_occupied"] = False
                state["counted_tracks"] = {}
                state["track_states"] = {}
                state["last_active_track_id"] = None
                state["last_matched_label"] = ""
                state["last_match_frame"] = -1

            # 트랙 기반 centroid 누적 (track_id 없는 경우 NO-OP)
            if track_state is not None:
                frame_h, frame_w = frame.shape[:2]
                ncx = (x + w / 2) / frame_w
                ncy = (y + h / 2) / frame_h
                thist = track_state.setdefault("centroid_history", [])
                thist.append((ncx, ncy))
                if len(thist) > direction_history_frames:
                    track_state["centroid_history"] = thist[-direction_history_frames:]

            # Crop 생성
            fh, fw = frame.shape[:2]
            cx1 = max(0, int(x))
            cy1 = max(0, int(y))
            cx2 = min(fw, int(x + w))
            cy2 = min(fh, int(y + h))
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                if roi_poly is not None:
                    cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)
                return display_frame

            # 추론 주기
            burst_infer = (
                track_state is not None
                and int(track_state.get("age", 0)) <= max(0, int(infer_burst_track_age))
            )
            if roi_poly is not None and roi_entry_mode:
                periodic_slot = frame_count % max(1, detect_every_n_frames) == 0
                allow_inference = counting_enabled and inside_roi and (
                    entry_event or periodic_slot or burst_infer
                )
                if inside_roi:
                    state["last_status"] = "ROI 진입" if entry_event else "ROI 내부"
                else:
                    state["last_status"] = "ROI 외부"
            else:
                allow_inference = counting_enabled and (
                    frame_count % max(1, detect_every_n_frames) == 0 or burst_infer
                )

            # ── 4. 임베딩 + FAISS 매칭 ──
            if allow_inference:
                emb = build_query_embedding(crop, model_bundle)
                query = np.expand_dims(emb, axis=0)

                distances, indices = faiss_index.search(query, 1)
                best_idx = int(indices[0][0])
                best_score = float(distances[0][0])

                if best_score > match_threshold and best_idx < len(labels):
                    name = str(labels[best_idx])

                    # 무시 라벨 필터링: 2순위로 fallback
                    if ignore_labels and name in ignore_labels:
                        distances2, indices2 = faiss_index.search(query, 2)
                        fallback_ok = False
                        if indices2.shape[1] >= 2:
                            idx2 = int(indices2[0][1])
                            score2 = float(distances2[0][1])
                            name2 = str(labels[idx2]) if idx2 < len(labels) else ""
                            if score2 > match_threshold and name2 not in (ignore_labels or set()):
                                name = name2
                                best_score = score2
                                fallback_ok = True
                        if not fallback_ok:
                            state["last_label"] = "무시됨"
                            state["last_score"] = best_score
                            state["last_status"] = "필터링됨"
                            if roi_poly is not None:
                                cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)
                            return display_frame

                    state["last_label"] = name
                    state["last_score"] = best_score
                    state["last_status"] = "매칭됨"
                    state.setdefault("item_scores", {})[name] = best_score

                    # Track 라벨 컨센서스 업데이트
                    if track_state is not None:
                        _update_track_label_consensus(
                            track_state,
                            name,
                            frame_count=frame_count,
                            label_window=track_label_window,
                            label_min_votes=track_label_min_votes,
                        )
                        stable_label = str(track_state.get("stable_label", "") or "")
                        state["last_matched_label"] = stable_label
                        state["last_match_frame"] = int(track_state.get("last_match_frame", -1))

                    cv2.putText(
                        display_frame,
                        f"{name} ({best_score:.3f})",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                else:
                    state["last_label"] = "미매칭"
                    state["last_score"] = best_score
                    state["last_status"] = "매칭 실패"

            # ── 5. 트랙 기반 방향 판정/카운트 ──
            # 불확실하면 NO-OP(카운트 안 함)
            if counting_enabled and track_state is not None and track_id is not None:
                if not track_state.get("direction_committed", False):
                    _name = str(track_state.get("stable_label", "") or "")
                    _last_match_frame = int(track_state.get("last_match_frame", -10**9))
                    _label_fresh = (frame_count - _last_match_frame) <= max(
                        1, int(label_stale_frames)
                    )
                    _skip_labels = ("-", "미매칭", "무시됨", "필터링됨", "")
                    if _name and _name not in _skip_labels and _label_fresh:
                        _hist = track_state.get("centroid_history", [])
                        _track_age = int(track_state.get("age", 0))
                        _normal_age_ok = _track_age >= max(1, int(track_min_age_frames))

                        _action = "unknown"
                        _required_ratio = max(0.0, min(1.0, float(direction_sign_consistency)))
                        _decision_mode = "none"

                        # Normal path: 충분한 트랙 나이 + 기존 인라인 조건
                        if _normal_age_ok:
                            _action = _calculate_movement_direction(
                                _hist,
                                add_direction=add_direction,
                                min_movement=inline_direction_min_movement,
                                min_history_points=inline_direction_min_history_points,
                            )
                            if _action != "unknown":
                                _decision_mode = "normal"
                                _required_ratio = max(
                                    0.0, min(1.0, float(direction_sign_consistency))
                                )

                        # Fast path: 아직 짧은 트랙에서도 강한 이동이면 조기 판정
                        if _action == "unknown":
                            _action = _calculate_movement_direction(
                                _hist,
                                add_direction=add_direction,
                                min_movement=fast_decision_min_movement,
                                min_history_points=fast_decision_min_history_points,
                            )
                            if _action != "unknown":
                                _decision_mode = "fast"
                                _required_ratio = max(
                                    0.0, min(1.0, float(fast_decision_sign_consistency))
                                )

                        if _action != "unknown":
                            _ratio = _vertical_sign_consistency_ratio(
                                _hist,
                                action=_action,
                                add_direction=add_direction,
                                min_step_epsilon=direction_sign_epsilon,
                            )
                            if _ratio >= _required_ratio:
                                _lam = state.setdefault("last_action_map", {})
                                _now = time.time()
                                _li = _lam.get(_name, {})
                                _elapsed = _now - float(_li.get("time", -1e9))
                                _last_act = _li.get("action")
                                _opp = (_last_act is not None and _last_act != _action)
                                _cd_ok = _elapsed >= cooldown_seconds or (
                                    _opp and _elapsed >= opposite_action_cooldown_seconds
                                )
                                _can = False
                                if _cd_ok:
                                    if _action == "add":
                                        _can = should_count_track(
                                            state.setdefault("counted_tracks", {}),
                                            track_id,
                                            _name,
                                        )
                                    else:
                                        _can = int(
                                            state.setdefault("billing_items", {}).get(_name, 0)
                                        ) > 0
                                    if _can:
                                        _lam[_name] = {"time": _now, "action": _action}
                                if _can:
                                    _bi = state.setdefault("billing_items", {})
                                    if _action == "remove":
                                        _nq = max(0, int(_bi.get(_name, 0)) - 1)
                                        if _nq == 0:
                                            _bi.pop(_name, None)
                                        else:
                                            _bi[_name] = _nq
                                        _release_counted_track(
                                            state.setdefault("counted_tracks", {}),
                                            product_name=_name,
                                            track_id=track_id,
                                        )
                                    else:
                                        _bi[_name] = int(_bi.get(_name, 0)) + 1
                                    state["count_event"] = {
                                        "product": _name,
                                        "track_id": track_id,
                                        "quantity": int(_bi.get(_name, 0)),
                                        "action": _action,
                                    }
                                    track_state["direction_committed"] = True
                                    track_state["centroid_history"] = []
                                    logger.info(
                                        "TRACK COUNT[%s]: %s %s (track=%s) → qty=%d",
                                        _decision_mode.upper(),
                                        _action.upper(),
                                        _name,
                                        track_id,
                                        int(_bi.get(_name, 0)),
                                    )
                            else:
                                logger.debug(
                                    "direction consistency reject: track=%s mode=%s action=%s ratio=%.2f required=%.2f",
                                    track_id,
                                    _decision_mode,
                                    _action,
                                    _ratio,
                                    _required_ratio,
                                )
    else:
        # 탐지 없음: 불확실 카운트는 하지 않음 (NO-OP 정책)
        if roi_poly is not None and bool(state.get("roi_occupied", False)):
            prev_empty = int(state.get("roi_empty_frames", 0))
            state["roi_empty_frames"] = prev_empty + 1

            if int(state.get("roi_empty_frames", 0)) >= max(1, int(soft_reentry_frames)):
                state["last_matched_label"] = ""
                state["last_match_frame"] = -1

            if int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                state["roi_occupied"] = False
                state["counted_tracks"] = {}
                state["track_states"] = {}
                state["last_active_track_id"] = None
                state["last_matched_label"] = ""
                state["last_match_frame"] = -1

        state["last_label"] = "-"
        state["last_score"] = 0.0
        state["last_status"] = "미탐지"

    if not counting_enabled:
        state["last_status"] = "워밍업 중"

    if roi_poly is not None:
        cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

    return display_frame
