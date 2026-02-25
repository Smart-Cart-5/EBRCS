from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import MutableMapping
from typing import Any
import cv2
import numpy as np
import traceback

logger = logging.getLogger(__name__)

from checkout_core.counting import should_count_track
from checkout_core.inference import build_query_embedding

try:
    from checkout_core.tracker import ObjectTracker
except ImportError:
    ObjectTracker = None

# ==========================================================
# EasyOCR 엔진 초기화 (동기식 - GPU 가속)
# ==========================================================
try:
    import easyocr
    ocr_engine = easyocr.Reader(['ko', 'en'], gpu=True)
    logger.info("✅ [System] EasyOCR 엔진 로드 성공 (GPU 가속 모드)")
except Exception as e:
    ocr_engine = None
    logger.error(f"❌ [System] EasyOCR 초기화 중 에러 발생: {e}")

# ==========================================================
# 누락 방지: 기본 핵심 함수들
# ==========================================================
def _calculate_movement_direction(
    centroid_history: list[tuple[float, float]],
    add_direction: str = "down",  
    min_movement: float = 0.05,
    min_history_points: int = 6,
) -> str:
    min_history_points = max(2, int(min_history_points))
    if len(centroid_history) < min_history_points:
        return "unknown"

    n = max(2, len(centroid_history) // 4)
    early_y = sum(p[1] for p in centroid_history[:n]) / n
    late_y = sum(p[1] for p in centroid_history[-n:]) / n

    dy = late_y - early_y  

    if abs(dy) < min_movement:
        return "unknown"

    if add_direction == "down":
        return "add" if dy > 0 else "remove"
    else:  
        return "add" if dy < 0 else "remove"


def _vertical_sign_consistency_ratio(
    centroid_history: list[tuple[float, float]],
    *,
    action: str,
    add_direction: str,
    min_step_epsilon: float = 0.003,
) -> float:
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
    if track_id is not None:
        counted_tracks.pop(track_id, None)
        return

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
            counted_tracks.pop(key, None)
            try:
                counted_tracks.pop(int(key), None)
            except (ValueError, TypeError):
                pass
            track_states.pop(key, None)


# ==========================================================
# 🔍 동기식 OCR 정밀 분류기 (Fuzzy 제거, Track ID 분리 + CLAHE 적용)
# ==========================================================
def _refine_label_with_ocr(crop_img: np.ndarray, base_label: str, state: dict[str, Any], track_id: int | None = None) -> str:
    print(f"\n▶️ [OCR 진입] FAISS 라벨: {base_label}, Track ID: {track_id}")

    if "ocr_track_cache" not in state:
        state["ocr_track_cache"] = {}

    if track_id is not None and track_id in state["ocr_track_cache"]:
        return state["ocr_track_cache"][track_id]

    target_keywords = ["컵밥", "컵반", "햇반", "햇밥", "혼밥"]
    if not any(k in base_label for k in target_keywords):
        return base_label

    if ocr_engine is None:
        return base_label

    try:
        h, w = crop_img.shape[:2]
        if h < 200 or w < 200:
            analysis_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            h, w = analysis_img.shape[:2]
        else:
            analysis_img = crop_img
            
        gray_img = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(gray_img)

        results = ocr_engine.readtext(enhanced_img, detail=1)
        del analysis_img, gray_img, enhanced_img 

        if not results:
            return base_label

        total_area = h * w
        min_area_threshold = total_area * 0.003
        
        filtered_results = []
        for (bbox, text, prob) in results:
            bbox_np = np.array(bbox, dtype=np.int32)
            if cv2.contourArea(bbox_np) >= min_area_threshold:
                filtered_results.append((bbox, text, prob))

        if not filtered_results:
             return base_label

        full_text = ""
        all_detected_texts = []
        for (bbox, text, prob) in filtered_results:
            clean_text = text.replace(" ", "")
            full_text += clean_text
            all_detected_texts.append(f"'{clean_text}'({prob:.2f})")
            
        print(f"  -> 🔎 [인식 텍스트]: {', '.join(all_detected_texts)}")
        
        # 💡 Fuzzy 매칭이 빠졌으므로, 딕셔너리에 오타(항태, 홤태 등)를 다시 추가했습니다.
        CUPBOP_DB_MAPPING = {
            "10240_CJ햇반컵반설렁탕밥253g": ["설렁탕"],
            "15446_CJ햇반컵반스팸마요덮밥219G": ["스팸마요", "스팸", "스햄"],
            "15447_CJ햇반컵반강된장보리비빔밥280G": ["강된장", "보리비빔", "강된"],
            "15448_오뚜기컵밥수원식갈비탕밥311G": ["수원식", "갈비탕"],
            "15449_오뚜기컵밥햄버그덮밥310G": ["햄버그", "함박"],
            "15450_오뚜기컵밥부산식돼지국밥316G": ["부산식", "돼지국밥"],
            "15834_한일식품혼면혼밥본고장바지락칼국수_밥276.5G": ["바지락", "바지락칼국수"],
            "15835_한일식품혼면혼밥본고장김치칼국수_밥274.5G": ["김치칼국수", "김치칼"],
            "20228_CJ)햇반컵반중화마파두부덮밥270G": ["마파두부", "중화"],
            "20235_CJ햇반컵반낙지콩나물비빔밥216g": ["낙지", "낙지콩나물"],
            "25417_CJ햇반컵반미역국밥167G": ["미역", "미역국"],
            "25418_CJ햇반컵반사골곰탕국밥166G": ["사골", "사골곰탕"],
            "25419_CJ햇반컵반콩나물해장국밥270G": ["콩나물해장", "해장국"],
            "25420_CJ햇반컵반황태국밥170G": ["황태", "항태", "홤태", "태국밥", "황태국"],
            "25421_CJ햇반컵반직화불고기덮밥257G": ["직화불고기", "불고기덮밥", "불고"],
            "25422_CJ햇반컵반치킨커리덮밥278G": ["치킨커리", "치킨카레"],
            "25423_CJ햇반컵반고추장나물현미비빔밥229G": ["고추장나물", "현미비빔", "고추장"],
            "25424_CJ햇반컵반볶은김치덮밥247G": ["볶은김치", "볶은"],
            "25426_오뚜기컵밥춘천닭갈비덮밥310G": ["닭갈비", "춘천"],
            "25427_오뚜기컵밥허니갈릭치밥310G": ["허니갈릭", "치밥"],
            "25428_오뚜기컵밥소시지카레덮밥315G": ["소시지카레", "소세지카레"],
            "25429_오뚜기컵밥참치마요덮밥247G": ["참치", "참치마요", "치마요"],
            "25430_오뚜기컵밥소시지야채덮밥280G": ["소시지야채", "소야"],
            "30202_CJ햇반컵반버섯곤드레비빔밥189g": ["곤드레", "버섯곤드레"],
            "40221_CJ햇반컵반닭곰탕밥261g": ["닭곰", "곰탕", "닭곰탕"],
            "50173_CJ햇반컵반육개장국밥": ["육개장", "육개"],
            "80172_CJ햇반컵반순두부찌개국밥173G": ["순두", "순부", "두부찌"],
            "80177_CJ햇반컵반김치날치알밥188g": ["날치알", "김치날치"],
            "80178_CJ햇반컵반버터장조림비빔밥216g": ["버터", "장조림", "버터장조림"],
            "90205_CJ햇반컵반직화볶음짜장덮밥280G": ["짜장", "볶음짜장", "직화볶음"]
        }

        refined = base_label
        
        is_ottogi = any(k in full_text for k in ["오뚜기", "오뚜", "뚜기"])
        is_cj = any(k in full_text for k in ["CJ", "씨제이", "햇반", "컵반"])

        if "불닭" in full_text or "불닭마요" in full_text:
            if is_ottogi:
                refined = "25425_오뚜기컵밥불닭마요덮밥277g"
            else:
                refined = "46090_씨제이)햇반컵반불닭마요덮밥"
        elif "제육" in full_text:
            if "철판" in full_text or is_cj:
                refined = "15445_CJ햇반컵반철판제육덮밥256G"
            else:
                refined = "65046_오뚜기컵밥제육덮밥"
        else:
            for db_name, keywords in CUPBOP_DB_MAPPING.items():
                if any(keyword in full_text for keyword in keywords):
                    refined = db_name
                    break

        if refined != base_label and track_id is not None:
            state["ocr_track_cache"][track_id] = refined
            print(f"  -> 🎯 [보정 완료] Track {track_id}를 '{refined}'로 캐싱합니다.")

        return refined
            
    except Exception as e:
        print(f"  -> ❌ [크리티컬 에러] OCR 구동 중 문제 발생: {e}")
        return base_label


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
    """Process a single frame and update checkout state in-place."""
    display_frame = frame.copy()

    state["count_event"] = None
    state["current_track_id"] = None
    counting_enabled = frame_count > max(0, int(warmup_frames))

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

    # ── 2. DeepSORT 업데이트 ──
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
                            
                    # =========================================================
                    # 💡 EasyOCR을 통한 최종 정밀 보정 (🚨 치명적 파라미터 버그 완벽 수정!)
                    # =========================================================
                    refined_name = _refine_label_with_ocr(crop, name, state, track_id)

                    if name != refined_name:
                        logger.info(f"💡 OCR 보정 완료: {name} -> {refined_name}")
                        name = refined_name 
                    # =========================================================

                    state["last_label"] = name
                    state["last_score"] = best_score
                    state["last_status"] = "매칭됨"
                    state.setdefault("item_scores", {})[name] = best_score

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