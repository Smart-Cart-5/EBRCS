# ROI Calibration Flow (Frontend + WS Protocol)

## Coordinate System
- `cart_roi_pending_polygon` and `cart_roi_polygon_confirmed` use **normalized coordinates** `[x, y]` in `[0.0, 1.0]`.
- Frontend must convert to pixels as:
  - `x_px = x_norm * videoWidth`
  - `y_px = y_norm * videoHeight`

## WebSocket Payload Contract
Backend sends a unified checkout state payload with these common fields on every message:

```json
{
  "type": "checkout_state",
  "session_id": "<uuid>",
  "phase": "IDLE | ROI_CALIBRATING | CHECKOUT_RUNNING",
  "message": "optional text or null",
  "cart_roi_auto_enabled": true,
  "checkout_start_mode": "auto_roi | no_roi",
  "cart_roi_available": true,
  "cart_roi_unavailable_reason": null
}
```

Always included ROI calibration control fields:

```json
{
  "cart_roi_preview_ready": true,
  "cart_roi_pending_polygon": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]],
  "cart_roi_pending_ratio": 0.2345,
  "confirm_enabled": true,
  "retry_enabled": true
}
```

Running phase may include confirmed polygon in normalized coordinates:

```json
{
  "cart_roi_polygon_confirmed": [[...], [...], ...]
}
```

Notes:
- In `ROI_CALIBRATING`, preview payload updates are rate-limited to about 4 FPS.
- In `CHECKOUT_RUNNING`, existing checkout fields (`billing_items`, `detection_boxes`, `topk_candidates`, etc.) continue as before.
- Roboflow semantic segmentation target class defaults to `cartline`.
  - `CART_ROI_CLASS_NAME` (default: `cartline`)
  - `CART_ROI_CLASS_ALIASES` (default: `cartline,cart,shopping_cart,trolley`)
  - Matching is case-insensitive with trim, and supports both map styles:
    - `{"cartline": 1}`
    - `{"0":"background","1":"cartline"}`

## API
- Start checkout mode (recommended):
  - `POST /api/sessions/{session_id}/checkout/start`
  - body: `{ "mode": "auto_roi" | "no_roi" }`
- `auto_roi` fallback behavior:
  - if `cart_roi_available=false`, server degrades to `no_roi` and returns message.
- Health:
  - `GET /api/health` includes:
    - `cart_roi_available: bool`
    - `cart_roi_unavailable_reason: string | null`
- Legacy per-session auto ROI mode:
  - `POST /api/sessions/{session_id}/roi/mode` with body `{ "enabled": true|false }`
- Confirm pending ROI:
  - `POST /api/sessions/{session_id}/roi/confirm`
- Retry ROI calibration:
  - `POST /api/sessions/{session_id}/roi/retry`
- State recovery endpoint:
  - `GET /api/sessions/{session_id}/state`

UI recommendation:
- When user clicks camera start, ask:
  - `자동 ROI로 시작`
  - `자동 ROI 없이 시작`
- Send selected mode via `/checkout/start` first, then open WS camera stream.
- Disable `자동 ROI` button when `cart_roi_available=false` and show reason tooltip.
- If `자동 ROI 사용` is selected, show ROI confirmation guidance within 3 seconds.
- If `ROI 없이 바로 시작` is selected, use legacy full-frame ROI behavior so product inference starts immediately.
- `auto_roi`를 끈 상태에서는 이벤트 ROI 소스 우선순위:
  1) calibration 4 points (`/warp` 설정값)
  2) 수동 ROI polygon (`/roi`)
  3) full fallback
- 즉, 4점 캘리브레이션이 있으면 EVENT_MODE inside/outside/crossing 판정은 해당 사다리꼴 ROI를 사용한다.
- calib4 ROI 유효성 가드:
  - `event_roi_area_ratio`를 계산해 너무 큰 ROI(기본 `>0.90`)는 `event_roi_too_large=true`로 경고
  - `event_roi_warnings`에 `roi_too_large`, `calib4_point_near_edge`, `calib4_bbox_near_full_frame` 등을 포함
  - 옵션: `EVENT_ROI_BLOCK_WHEN_TOO_LARGE=true`일 때 서버가 full fallback으로 강제 전환
- `auto_roi`에서 기본 동작은 **수동 confirm 필요**:
  - `POST /api/sessions/{session_id}/roi/confirm` 호출 전까지 phase=`ROI_CALIBRATING`
  - phase=`CHECKOUT_RUNNING` 진입 후에만 hand event engine 경로가 정상 동작
- 테스트용 자동 confirm 옵션:
  - `CHECKOUT_AUTO_CONFIRM_ROI=true`
  - `CHECKOUT_AUTO_CONFIRM_ROI_MIN_RATIO` / `CHECKOUT_AUTO_CONFIRM_ROI_MIN_POLY_POINTS` 조건 만족 시 서버가 자동 confirm 처리

## Manual Test Scenarios
1. API_KEY 있음 → auto_roi 정상:
   - Set env:
     - `CART_ROI_ENABLED=1`
     - `ROBOFLOW_API_KEY=<your_key>`
     - optional `CART_ROI_EVERY_N_FRAMES=10`
   - `카메라 시작` -> `자동 ROI로 시작` 선택.
   - Verify phase becomes `ROI_CALIBRATING`, guide overlay and preview polygon appear.
   - Click `OK`.
   - Verify phase changes to `CHECKOUT_RUNNING`, then product inference starts.

2. API_KEY 없음 → auto_roi fallback:
   - Set env:
     - `CART_ROI_ENABLED=1`
     - `ROBOFLOW_API_KEY` unset
   - `카메라 시작` -> `자동 ROI로 시작` 선택.
   - Verify server degrades to `no_roi`, phase goes `CHECKOUT_RUNNING`, message explains missing key.

3. CART_ROI_ENABLED=0 (guardrail) → auto_roi fallback:
   - Set env:
     - `CART_ROI_ENABLED=0`
     - `ROBOFLOW_API_KEY=<any>`
   - `카메라 시작` -> `자동 ROI로 시작` 선택.
   - Verify server degrades to `no_roi`, phase goes `CHECKOUT_RUNNING`, message explains server guardrail.

4. no_roi 선택:
   - `카메라 시작` -> `자동 ROI 없이 시작` 선택.
   - Verify direct transition to `CHECKOUT_RUNNING`.

## Hand Crossing Event Engine
- `EVENT_MODE=true`일 때 장바구니 반영은 hand crossing 기반 확정 이벤트만 사용한다.
- 입력: 프레임별 `hand`/`object(product)` bbox, ROI polygon, frame_id.
- hand↔object association:
  - IoU + 중심점 거리 스코어 기반 top-k 산출.
  - `HAND_EVENT_ASSOC_STABLE_FRAMES` 연속 유지된 후보만 안정 연결로 사용.
- 상태 전이:
  - `OUT_TO_IN` crossing -> `CANDIDATE_ADD`
  - `IN_TO_OUT` crossing -> `CANDIDATE_REMOVE`
  - crossing 시점에 후보가 없어도 candidate 상태 동안 late binding으로 후보 채택 가능
  - ADD 확정: 후보 object가 ROI 내부에서 `HAND_EVENT_ADD_EVIDENCE_FRAMES` 프레임 유지
  - REMOVE 확정: 후보 object가 ROI 내부에서 `HAND_EVENT_REMOVE_EVIDENCE_FRAMES` 프레임 연속 소실
  - timeout/cancel: `HAND_EVENT_CANDIDATE_TIMEOUT_S`
  - cooldown: `HAND_EVENT_COOLDOWN_S`
- 성능 정책:
  - 임베딩/FAISS는 매 프레임이 아니라 **ADD 확정 이벤트 시점**에만 호출한다.
  - 기존 분류/FAISS 함수는 그대로 재사용한다.

### 주요 설정값
- `EVENT_MODE` (default `true`)
- `HAND_EVENT_ASSOC_TOP_K`
- `HAND_EVENT_ASSOC_STABLE_FRAMES`
- `HAND_EVENT_MIN_ASSOC_SCORE`
- `HAND_EVENT_MIN_ASSOC_IOU`
- `HAND_EVENT_ALLOW_DET_FALLBACK`
- `HAND_EVENT_DET_FALLBACK_MIN_SCORE`
- `HAND_EVENT_CANDIDATE_ROI_RELAX`
- `HAND_EVENT_CANDIDATE_HAND_NEAR_DIST`
- `HAND_EVENT_ADD_EVIDENCE_FRAMES`
- `HAND_EVENT_REMOVE_EVIDENCE_FRAMES`
- `HAND_EVENT_CANDIDATE_TIMEOUT_S`
- `HAND_EVENT_COOLDOWN_S`
- `HAND_EVENT_CANDIDATE_SWITCH_MIN_DELTA`
- `HAND_EVENT_TRACK_IOU_MATCH_THRESHOLD`
- `HAND_EVENT_TRACK_MAX_MISSED_FRAMES`
- `HAND_EVENT_HAND_REPRESENTATIVE_POINT` (`center` | `bottom_center`)
- `HAND_EVENT_OBJECT_REPRESENTATIVE_POINT` (`bottom_center` | `center`)
- `HAND_EVENT_DEBUG_LOG`
- `HAND_EVENT_DEBUG_OVERLAY`
- `HAND_EVENT_WS_DEBUG`
