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

# ── EasyOCR 엔진 (모듈 레벨 lazy 초기화) ──────────────────────────────────
# gpu=True → CUDA(서버 T4) 또는 MPS(로컬 Mac) 자동 선택
_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            import easyocr
            _ocr_engine = easyocr.Reader(['ko', 'en'], gpu=True, verbose=False)
            logger.info("EasyOCR 엔진 로드 성공")
        except Exception as exc:
            logger.warning("EasyOCR 초기화 실패 (OCR 비활성화): %s", exc)
            _ocr_engine = False  # False = 초기화 시도했지만 실패
    return _ocr_engine if _ocr_engine else None


# ── 컵밥 계열 감지 키워드 ──────────────────────────────────────────────────
_CUPBOP_TARGET_KEYWORDS: frozenset[str] = frozenset({
    "컵밥", "컵반", "햇반", "햇밥", "혼밥",
})

# ── 컵밥 상품 OCR 키워드 → DB 라벨 매핑 ──────────────────────────────────
_CUPBOP_DB_MAPPING: dict[str, list[str]] = {
    "25420_CJ햇반컵반황태국밥170G":       ["황태", "항태", "홤태", "태국밥", "황태국"],
    "15446_CJ햇반컵반스팸마요덮밥219G":   ["스팸마요", "스팸", "스햄"],
    "15445_CJ햇반컵반철판제육덮밥256G":   ["철판제육", "제육"],
    "25425_오뚜기컵밥불닭마요덮밥277g":   ["불닭마요", "불닭"],
    "46090_씨제이)햇반컵반불닭마요덮밥":  ["불닭마요CJ", "불닭CJ"],
    "15447_CJ햇반컵반소불고기덮밥256G":   ["소불고기", "불고기"],
    "15443_CJ햇반컵반참치마요덮밥229G":   ["참치마요", "참치"],
    "15444_CJ햇반컵반김치찌개덮밥256G":   ["김치찌개", "김치"],
    "25419_CJ햇반컵반차돌된장국밥170G":   ["차돌된장", "된장"],
    "25421_CJ햇반컵반미역국밥170G":       ["미역국", "미역"],
    "25422_CJ햇반컵반순두부찌개국밥170G": ["순두부", "순두부찌개"],
    "25423_CJ햇반컵반육개장국밥170G":     ["육개장"],
    "25424_CJ햇반컵반갈비탕국밥170G":     ["갈비탕"],
    "46089_씨제이)햇반컵반스팸마요덮밥":  ["스팸마요CJ"],
    "46091_씨제이)햇반컵반철판제육덮밥":  ["제육CJ"],
    "46088_씨제이)햇반컵반소불고기덮밥":  ["소불고기CJ"],
    "46087_씨제이)햇반컵반참치마요덮밥":  ["참치CJ"],
    "25426_오뚜기컵밥참치마요덮밥277g":   ["오뚜기참치마요", "오뚜기참치"],
    "25427_오뚜기컵밥소불고기덮밥277g":   ["오뚜기소불고기"],
    "25428_오뚜기컵밥김치참치덮밥277g":   ["오뚜기김치참치"],
    "25429_오뚜기컵밥제육덮밥277g":       ["오뚜기제육"],
    "25430_오뚜기컵밥황태해장국밥277g":   ["오뚜기황태"],
    "25431_오뚜기컵밥순두부찌개국밥277g": ["오뚜기순두부"],
    "25432_오뚜기컵밥육개장국밥277g":     ["오뚜기육개장"],
    "25433_오뚜기컵밥갈비탕국밥277g":     ["오뚜기갈비탕"],
    "25434_오뚜기컵밥김치찌개국밥277g":   ["오뚜기김치찌개"],
    "25435_오뚜기컵밥차돌된장국밥277g":   ["오뚜기차돌"],
    "25436_오뚜기컵밥미역국밥277g":       ["오뚜기미역"],
    "25437_동원쎈쿡컵밥소불고기덮밥250g": ["쎈쿡소불고기", "동원소불고기"],
    "25438_동원쎈쿡컵밥참치마요덮밥250g": ["쎈쿡참치", "동원참치마요"],
    "25439_동원쎈쿡컵밥제육덮밥250g":     ["쎈쿡제육", "동원제육"],
    "25440_동원쎈쿡컵밥김치찌개덮밥250g": ["쎈쿡김치", "동원김치"],
    "25441_동원쎈쿡컵밥불닭마요덮밥250g": ["쎈쿡불닭", "동원불닭"],
}


def _is_cupbop_label(label: str) -> bool:
    """FAISS 결과 라벨이 컵밥 계열인지 판단."""
    return any(kw in label for kw in _CUPBOP_TARGET_KEYWORDS)


def _refine_label_with_ocr(
    crop_img: np.ndarray,
    base_label: str,
    state: MutableMapping[str, Any],
    track_id: int | None = None,
) -> str | None:
    """컵밥 계열 상품에 대해 OCR로 라벨을 정밀 보정.

    Returns:
        str : OCR로 확인된 DB 라벨 (키워드 매칭 성공, base_label과 같을 수도 있음)
        None: OCR 미확인 (텍스트 없음, 키워드 미매칭, 엔진 오류)
              → 호출 측에서 "아직 못 읽음"으로 판단해 대기 또는 취소
    """
    if not _is_cupbop_label(base_label):
        return base_label  # 컵밥 아닌 상품: 즉시 base_label 반환 (OCR 대상 아님)

    # Track 캐시 확인 (이전에 성공한 OCR 결과 재사용)
    cache: dict = state.setdefault("ocr_track_cache", {})
    cache_key = str(track_id) if track_id is not None else None
    if cache_key and cache_key in cache:
        return cache[cache_key]

    engine = _get_ocr_engine()
    if engine is None:
        return None  # 엔진 없음 → 호출 측에서 처리

    try:
        h, w = crop_img.shape[:2]

        # 작은 이미지 확대
        analysis_img = crop_img
        if h < 200 or w < 200:
            analysis_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0,
                                      interpolation=cv2.INTER_LINEAR)
            h, w = analysis_img.shape[:2]

        # CLAHE 대비 강화
        gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        results = engine.readtext(enhanced, detail=1)

        # 최소 면적 필터 (이미지 면적의 0.3%)
        min_area = h * w * 0.003
        filtered = [
            (bbox, text, prob)
            for (bbox, text, prob) in results
            if cv2.contourArea(np.array(bbox, dtype=np.int32)) >= min_area
        ]

        full_text = "".join(t.replace(" ", "") for _, t, _ in filtered)

        if not full_text:
            return None  # 텍스트 없음 → 아직 못 읽음

        # 오뚜기 브랜드 여부
        is_ottogi = "오뚜기" in full_text

        # 불닭 특수 규칙 (브랜드 구분 필요)
        if "불닭" in full_text:
            refined = (
                "25425_오뚜기컵밥불닭마요덮밥277g"
                if is_ottogi
                else "46090_씨제이)햇반컵반불닭마요덮밥"
            )
            if cache_key:
                cache[cache_key] = refined
            return refined

        # 일반 키워드 매핑
        for db_name, keywords in _CUPBOP_DB_MAPPING.items():
            if any(kw in full_text for kw in keywords):
                if cache_key:
                    cache[cache_key] = db_name
                logger.info("OCR 보정: %s → %s (text=%r)", base_label, db_name, full_text)
                return db_name

    except Exception as exc:
        logger.warning("OCR 처리 중 에러: %s", exc)

    return None  # 키워드 미매칭 → 아직 인식 못함


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

    # ── OCR pending 상태 처리 (CP5) ──────────────────────────────────────
    # 컵밥 상품 방향 판정 완료 후 OCR 확인 대기 중인 경우
    # 타임아웃을 wall-clock 기반으로 → FPS(캡처 주기) 변경에 무관하게 일정한 대기 시간 보장
    _OCR_PENDING_TIMEOUT_SEC = 6.0  # 6초 wall-clock: 모달 확인 + 상품 뒤집어 비추는 시간
    if counting_enabled and state.get("ocr_state") == "ocr_pending":
        _pending_action = state.get("ocr_pending_action", "add")
        _pending_label = state.get("ocr_pending_base_label", "")
        _pending_tid = state.get("ocr_pending_track_id")
        _pending_since_time = float(state.get("ocr_pending_since_time", time.time()))
        _elapsed_sec = time.time() - _pending_since_time

        # DeepSORT: 빈 detection으로 update → OCR 대기 중 트랙이 정상 만료되도록
        # (호출 안 하면 트랙 age가 동결되어 OCR 완료 후 동일 track_id가 새 상품에 재배정됨)
        if use_tracking and tracker is not None:
            try:
                tracker.update(frame, [])
            except Exception:
                pass

        # 현재 프레임 crop 추출 (OCR 실행용)
        _ocr_crop: np.ndarray | None = None
        _fg_for_ocr = bg_subtractor.apply(frame, learningRate=0)  # 상태 변경 없이 mask만
        _fg_for_ocr = cv2.erode(_fg_for_ocr, None, iterations=2)
        _fg_for_ocr = cv2.dilate(_fg_for_ocr, None, iterations=4)
        _, _fg_for_ocr = cv2.threshold(_fg_for_ocr, 200, 255, cv2.THRESH_BINARY)
        _cnts_ocr, _ = cv2.findContours(_fg_for_ocr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _cands_ocr = [c for c in _cnts_ocr if cv2.contourArea(c) > min_area]
        if _cands_ocr:
            _ox, _oy, _ow, _oh = cv2.boundingRect(max(_cands_ocr, key=cv2.contourArea))
            _fh, _fw = frame.shape[:2]
            _ocr_crop = frame[
                max(0, _oy): min(_fh, _oy + _oh),
                max(0, _ox): min(_fw, _ox + _ow),
            ]

        _resolved_label = _pending_label
        _do_fire = False
        _ocr_confirmed: str | None = None

        if _ocr_crop is not None and _ocr_crop.size > 0:
            _ocr_confirmed = _refine_label_with_ocr(_ocr_crop, _pending_label, state, _pending_tid)
            if _ocr_confirmed is not None:
                # OCR 키워드 매칭 성공 → 확인된 라벨로 발화
                _resolved_label = _ocr_confirmed
                _do_fire = True

        # 타임아웃 처리: OCR 미확인 상태에서 시간 초과
        if not _do_fire and _elapsed_sec >= _OCR_PENDING_TIMEOUT_SEC:
            if _pending_action == "remove":
                # REMOVE 타임아웃: FAISS 기본 라벨로 제거 시도 (베스트에포트)
                # → ADD와 달리 REMOVE는 이미 담긴 상품을 꺼내는 것이므로 기본 라벨 fallback 허용
                _bi_check = state.setdefault("billing_items", {})
                if int(_bi_check.get(_pending_label, 0)) > 0:
                    _resolved_label = _pending_label
                    _do_fire = True
                    logger.info(
                        "OCR REMOVE 타임아웃(%.1fs): FAISS 라벨로 제거 시도 (%s)",
                        _elapsed_sec, _pending_label,
                    )
                else:
                    logger.info(
                        "OCR REMOVE 타임아웃(%.1fs): 장바구니에 없음 → 취소 (%s)",
                        _elapsed_sec, _pending_label,
                    )

            if not _do_fire:
                # ADD 타임아웃 OR REMOVE이지만 장바구니에 없는 경우 → 취소 + 재시도 허용
                state["ocr_state"] = "normal"
                state["ocr_pending_action"] = None
                state["ocr_pending_track_id"] = None
                state["ocr_pending_base_label"] = ""
                state["ocr_pending_since_frame"] = -1
                state["ocr_pending_since_time"] = 0.0
                # 쿨다운·track 잠금 해제 → 취소 직후 즉시 재시도 가능
                state.get("last_action_map", {}).pop(_pending_label, None)
                _ct = state.get("counted_tracks", {})
                if _pending_tid is not None:
                    _ct.pop(_pending_tid, None)
                    try:
                        _ct.pop(int(_pending_tid), None)
                    except (TypeError, ValueError):
                        pass
                logger.info(
                    "OCR pending 타임아웃(%.1fs): OCR 미확인 → 카운트 취소, 재시도 허용 (%s)",
                    _elapsed_sec, _pending_label,
                )
                if roi_poly is not None:
                    cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)
                return display_frame

        if _do_fire:
            _bi = state.setdefault("billing_items", {})
            if _pending_action == "remove":
                if int(_bi.get(_resolved_label, 0)) > 0:
                    _nq = max(0, int(_bi.get(_resolved_label, 0)) - 1)
                    if _nq == 0:
                        _bi.pop(_resolved_label, None)
                    else:
                        _bi[_resolved_label] = _nq
                    _release_counted_track(
                        state.setdefault("counted_tracks", {}),
                        product_name=_resolved_label,
                        track_id=_pending_tid,
                    )
                    state["count_event"] = {
                        "product": _resolved_label,
                        "track_id": _pending_tid,
                        "quantity": int(_bi.get(_resolved_label, 0)),
                        "action": "remove",
                    }
            else:
                _bi[_resolved_label] = int(_bi.get(_resolved_label, 0)) + 1
                state["count_event"] = {
                    "product": _resolved_label,
                    "track_id": _pending_tid,
                    "quantity": int(_bi.get(_resolved_label, 0)),
                    "action": "add",
                }
            state.setdefault("item_scores", {})[_resolved_label] = state.get("last_score", 0.0)
            state["ocr_state"] = "normal"
            state["ocr_pending_action"] = None
            state["ocr_pending_track_id"] = None
            state["ocr_pending_base_label"] = ""
            state["ocr_pending_since_frame"] = -1
            # last_action_map을 OCR 발화 시각으로 갱신
            # (방향판정 시각 기준이면 OCR 지연 시간만큼 쿨다운이 소모되어 두 번째 스캔이 차단됨)
            state.setdefault("last_action_map", {})[_resolved_label] = {
                "time": time.time(),
                "action": _pending_action,
            }
            logger.info(
                "OCR COUNT[%s]: %s %s → qty=%d",
                "OCR" if _resolved_label != _pending_label else "FALLBACK",
                _pending_action.upper(),
                _resolved_label,
                int(state["billing_items"].get(_resolved_label, 0)),
            )

        if roi_poly is not None:
            cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)
        return display_frame
    # ── OCR pending 처리 끝 ───────────────────────────────────────────────

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
                                        _bi = state.setdefault("billing_items", {})
                                        if _is_cupbop_label(_name):
                                            # REMOVE + 컵밥: 임베딩 라벨이 장바구니 라벨과 달라도
                                            # 장바구니에 컵밥이 하나라도 있으면 OCR pending 진입 허용.
                                            # 실제 차감은 OCR 확정/최종 검증 단계에서만 수행.
                                            _can = any(
                                                int(_qty) > 0 and _is_cupbop_label(str(_label))
                                                for _label, _qty in _bi.items()
                                            )
                                        else:
                                            _can = int(_bi.get(_name, 0)) > 0
                                    if _can:
                                        _lam[_name] = {"time": _now, "action": _action}
                                if _can:
                                    # direction_committed 먼저 설정 (재판정 방지)
                                    track_state["direction_committed"] = True
                                    track_state["centroid_history"] = []

                                    # ── CP4: 컵밥 계열 → OCR pending 분기 ──
                                    # OCR 엔진이 사용 가능한 경우에만 pending 모달 진입.
                                    # 엔진이 없으면(초기화 실패) FAISS 라벨로 즉시 카운트.
                                    if _is_cupbop_label(_name) and _get_ocr_engine() is not None:
                                        state["ocr_state"] = "ocr_pending"
                                        state["ocr_pending_action"] = _action
                                        state["ocr_pending_track_id"] = track_id
                                        state["ocr_pending_base_label"] = _name
                                        state["ocr_pending_since_frame"] = frame_count
                                        state["ocr_pending_since_time"] = time.time()
                                        logger.info(
                                            "OCR PENDING: %s %s (track=%s) → 모달 표시",
                                            _action.upper(),
                                            _name,
                                            track_id,
                                        )
                                    else:
                                        # 일반 상품: 기존 즉시 카운트
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
