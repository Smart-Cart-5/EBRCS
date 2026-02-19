from __future__ import annotations

import logging
import time
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

    # ── 1. 배경차분 + contour 탐지 (기존 그대로) ──
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

    # ── 2. DeepSORT 백그라운드 업데이트 (모든 bbox 넘기기) ──
    tracks: list[dict[str, Any]] = []
    if use_tracking and tracker is not None:
        all_bboxes = []
        if candidates:
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
            # 빈 detection으로도 update를 호출해 stale track이 자연 소거되도록 함
            tracks = tracker.update(frame, all_bboxes)
        except Exception:
            logger.exception("tracker update failed")
            tracks = []

    # ── 3. 기존 인식 로직 (가장 큰 contour 1개) ──
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
            # Track ID 찾기 (DeepSORT 활성화 시)
            track_id = None
            if tracks and tracker is not None:
                track_id = tracker.find_track_for_bbox(tracks, (x, y, w, h))
                state["current_track_id"] = track_id

            # Bounding box 그리기
            box_color = (0, 255, 0)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

            # ROI 체크 (기존 로직 그대로)
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
                    if entry_event:
                        # 새 진입에서는 이전 궤적을 버려 방향 판정 오염 방지
                        state["centroid_history"] = []
                        state["direction_committed"] = False   # 새 진입 시 커밋 플래그 초기화
                        state["last_matched_label"] = ""       # 이전 상품명이 새 진입에 오염되는 것 방지
                        state["last_match_frame"] = -1
                else:
                    state["roi_empty_frames"] = int(state.get("roi_empty_frames", 0)) + 1

            if roi_poly is not None and int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                state["roi_occupied"] = False
                state["counted_tracks"] = {}       # 재진입 시 다시 카운트 가능
                state["centroid_history"] = []     # 새 진입 시 궤적 초기화
                state["direction_committed"] = False
                state["last_matched_label"] = ""
                state["last_match_frame"] = -1

            # ── 방향 감지: 정규화된 centroid 추적 ──
            frame_h, frame_w = frame.shape[:2]
            ncx = (x + w / 2) / frame_w
            ncy = (y + h / 2) / frame_h  # y=0이 화면 상단, y 증가 = 아래 이동
            centroid_hist = state.setdefault("centroid_history", [])
            centroid_hist.append((ncx, ncy))
            if len(centroid_hist) > direction_history_frames:
                state["centroid_history"] = centroid_hist[-direction_history_frames:]

            # Crop 생성
            fh, fw = frame.shape[:2]
            cx1 = max(0, int(x))
            cy1 = max(0, int(y))
            cx2 = min(fw, int(x + w))
            cy2 = min(fh, int(y + h))
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                # 빈 crop이면 스킵
                if roi_poly is not None:
                    cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)
                return display_frame

            # 추론 주기 결정 (기존 ROI entry mode 그대로)
            if roi_poly is not None and roi_entry_mode:
                periodic_slot = frame_count % max(1, detect_every_n_frames) == 0
                allow_inference = counting_enabled and inside_roi and (entry_event or periodic_slot)
                if inside_roi:
                    state["last_status"] = "ROI 진입" if entry_event else "ROI 내부"
                else:
                    state["last_status"] = "ROI 외부"
            else:
                allow_inference = counting_enabled and frame_count % max(1, detect_every_n_frames) == 0

            # ── 4. 임베딩 + FAISS 매칭 (추론 주기마다 상품 식별) ──
            # 방향 판정은 FAISS와 독립적으로 매 프레임 실행 (아래 ── 5 참조)
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
                    state["last_matched_label"] = name  # 방향 판정용 (미매칭 영향 없음)
                    state["last_match_frame"] = frame_count
                    state["last_score"] = best_score
                    state["last_status"] = "매칭됨"
                    state.setdefault("item_scores", {})[name] = best_score

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

            # ── 5. 방향 판정 및 즉시 카운트 (방향 확정 시) ──
            # direction_committed=True이면 이번 진입에서 이미 카운트됨 → 스킵
            # 인라인 임계값은 exit fallback보다 엄격 → 손떨림/잔상 오검출 방지
            if counting_enabled and not state.get("direction_committed", False):
                _last_match_frame = int(state.get("last_match_frame", -10**9))
                _label_fresh = (frame_count - _last_match_frame) <= max(1, int(label_stale_frames))
                _name = state.get("last_matched_label", "") if _label_fresh else ""
                _skip_labels = ("-", "미매칭", "무시됨", "필터링됨", "")
                if _name and _name not in _skip_labels:
                    _hist = state.get("centroid_history", [])
                    _action = _calculate_movement_direction(
                        _hist,
                        add_direction=add_direction,
                        min_movement=inline_direction_min_movement,
                        min_history_points=inline_direction_min_history_points,
                    )
                    if _action != "unknown":
                        _lam = state.setdefault("last_action_map", {})
                        _now = time.time()
                        _li = _lam.get(_name, {})
                        _elapsed = _now - float(_li.get("time", -1e9))
                        _last_act = _li.get("action")
                        _opp = (_last_act is not None and _last_act != _action)
                        _cd_ok = _elapsed >= cooldown_seconds or (
                            _opp and _elapsed >= opposite_action_cooldown_seconds
                        )
                        _tid = state.get("current_track_id")
                        _can = False
                        if _cd_ok:
                            if _action == "add":
                                if _tid is not None:
                                    _can = should_count_track(
                                        state.setdefault("counted_tracks", {}), _tid, _name
                                    )
                                else:
                                    _can = True
                            else:
                                _can = int(state.setdefault("billing_items", {}).get(_name, 0)) > 0
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
                                    product_name=_name, track_id=_tid,
                                )
                            else:
                                _bi[_name] = int(_bi.get(_name, 0)) + 1
                            state["count_event"] = {
                                "product": _name,
                                "track_id": _tid,
                                "quantity": int(_bi.get(_name, 0)),
                                "action": _action,
                            }
                            state["direction_committed"] = True
                            state["centroid_history"] = []
                            logger.info(
                                "INLINE COUNT: %s %s → qty=%d",
                                _action.upper(), _name, int(_bi.get(_name, 0)),
                            )
                    else:
                        logger.debug(
                            "in-frame direction: name=%r hist=%d action=%s",
                            _name, len(_hist), _action,
                        )
    else:
        # 탐지 없음
        if roi_poly is not None and bool(state.get("roi_occupied", False)):
            prev_empty = int(state.get("roi_empty_frames", 0))
            state["roi_empty_frames"] = prev_empty + 1

            # 소프트 재진입: N프레임+ 비어있으면 direction_committed 리셋
            # → 짧은 이탈 후 복귀 시에도 새 방향 판정 가능 (전체 ROI clear 불필요)
            if (
                int(state.get("roi_empty_frames", 0)) >= max(1, int(soft_reentry_frames))
                and state.get("direction_committed", False)
            ):
                state["direction_committed"] = False
                state["centroid_history"] = []
                state["last_matched_label"] = ""
                state["last_match_frame"] = -1
                logger.debug("soft re-entry: direction_committed reset at roi_empty=%d", state["roi_empty_frames"])

            # 퇴장 첫 프레임: 인라인 카운트 미발동 시 exit fallback으로 카운트
            if prev_empty == 0 and counting_enabled and not state.get("direction_committed", False):
                exit_name = state.get("last_matched_label", "")
                _skip = ("-", "미매칭", "무시됨", "필터링됨", "")
                if exit_name and exit_name not in _skip:
                    exit_hist = state.get("centroid_history", [])
                    exit_action = _calculate_movement_direction(
                        exit_hist,
                        add_direction=add_direction,
                        min_movement=direction_min_movement,
                        min_history_points=direction_min_history_points,
                    )
                    logger.info("exit trigger: name=%r hist=%d action=%s", exit_name, len(exit_hist), exit_action)
                    if exit_action != "unknown":
                        last_action_map = state.setdefault("last_action_map", {})
                        now = time.time()
                        last_info = last_action_map.get(exit_name, {})
                        elapsed = now - float(last_info.get("time", -1e9))
                        last_act = last_info.get("action")
                        opp = (last_act is not None and last_act != exit_action)
                        cooldown_ok = elapsed >= cooldown_seconds or (
                            opp and elapsed >= opposite_action_cooldown_seconds
                        )
                        exit_track_id = state.get("current_track_id")
                        can_count = False
                        if cooldown_ok:
                            if exit_action == "add":
                                if exit_track_id is not None:
                                    ct = state.setdefault("counted_tracks", {})
                                    can_count = should_count_track(ct, exit_track_id, exit_name)
                                else:
                                    can_count = True
                            else:
                                bi = state.setdefault("billing_items", {})
                                can_count = int(bi.get(exit_name, 0)) > 0
                            if can_count:
                                last_action_map[exit_name] = {"time": now, "action": exit_action}
                        if can_count:
                            bi = state.setdefault("billing_items", {})
                            if exit_action == "remove":
                                nq = max(0, int(bi.get(exit_name, 0)) - 1)
                                if nq == 0:
                                    bi.pop(exit_name, None)
                                else:
                                    bi[exit_name] = nq
                                _release_counted_track(
                                    state.setdefault("counted_tracks", {}),
                                    product_name=exit_name, track_id=exit_track_id,
                                )
                            else:
                                bi[exit_name] = int(bi.get(exit_name, 0)) + 1
                            state["count_event"] = {
                                "product": exit_name,
                                "track_id": exit_track_id,
                                "quantity": int(bi.get(exit_name, 0)),
                                "action": exit_action,
                            }
                            state["centroid_history"] = []
                            state["direction_committed"] = True
                            logger.info("EXIT COUNT: %s %s → qty=%d", exit_action.upper(), exit_name, int(bi.get(exit_name, 0)))

            # no-detect가 이어지면 stale label은 폐기해 과거 라벨 오염을 차단
            if prev_empty > 0:
                state["last_matched_label"] = ""
                state["last_match_frame"] = -1

            if int(state.get("roi_empty_frames", 0)) >= roi_clear_frames:
                state["roi_occupied"] = False
                state["counted_tracks"] = {}
                state["centroid_history"] = []
                state["direction_committed"] = False
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
