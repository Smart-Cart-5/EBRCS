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
