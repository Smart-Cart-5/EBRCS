# EBRCS_streaming Project Context (Detailed)

## 1) Project Goal
- Streamlit 기반 임베딩 검색형 리테일 체크아웃 데모.
- 입력 소스(라이브 카메라/업로드 영상)에서 상품을 인식하고 장바구니를 누적한다.
- 마지막에 영수증 검수 단계에서 수량 수정/삭제/확정한다.

## 2) Primary Stack
- App/UI: Streamlit, streamlit-drawable-canvas
- CV: OpenCV
- Embedding: Hugging Face Transformers (`DINOv3`, `CLIP`)
- Adapter: PEFT LoRA (optional)
- ANN Search: FAISS (`IndexFlatIP`)
- Data: numpy, pandas, Pillow, pyarrow

## 3) Entry Points and Navigation
- Desktop entry: `app.py`
- Mobile entry: `mobile_app.py`
- Desktop home page: `pages/0_Desktop_Home.py`
- Mobile checkout wrappers:
  - `pages/4_Checkout_Mobile.py`
  - `pages_mobile/2_Checkout_Mobile.py`
- Mobile checkout implementation: `pages_mobile/checkout_mobile_page.py`
- Common theme/sidebar: `ui_theme.py`
- Mobile nav map: `mobile_nav.py`

## 4) Core Inference Modules

### `checkout_core/inference.py`
- Loads HF token from env/secrets (`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`).
- Loads models (cached):
  - DINO: `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - DINO processor id: `facebook/dinov2-large`
  - CLIP: `openai/clip-vit-base-patch32`
- Optionally loads LoRA adapter from `data/adapter_model.safetensors`.
- Loads embedding DB (`data/embeddings.npy`, `data/labels.npy`) with shape validation.
- Builds weighted embeddings:
  - `DINO_WEIGHT = 0.7`
  - `CLIP_WEIGHT = 0.3`
- Builds/loads FAISS index (`data/faiss_index.bin`).
- Builds query embedding from a crop for top-1 matching.

### `checkout_core/frame_processor.py`
- Creates background subtractor (`KNN`).
- Per-frame pipeline:
  - Foreground mask -> morphology -> threshold
  - Optional ROI mask application
  - Contour candidates filtered by area
  - Main bbox crop
  - Inference slot control by frame interval
  - Optional ROI entry-based gating
  - FAISS top-1 search + threshold check
  - Count update with cooldown
- Draws bbox and ROI overlay on output frame.

### `checkout_core/counting.py`
- Uses time-based dedup state `last_seen_at`.
- `should_count_product()` compares current time vs last timestamp.
- Legacy frame-based key is intentionally removed.

### `checkout_core/video_input.py`
- Persists Streamlit uploaded video to tempfile for OpenCV reading.

## 5) Desktop Checkout (`pages/2_Checkout.py`)
- Constants:
  - `MIN_AREA = 2500`
  - `DETECT_EVERY_N_FRAMES = 5`
  - `MATCH_THRESHOLD = 0.62`
  - `COUNT_COOLDOWN_SECONDS = 1.0`
  - `ROI_CLEAR_FRAMES = 8`
- Source mode radio:
  - `라이브 카메라`
  - `업로드 영상`
- Live mode:
  - Polygon ROI editor via `st_canvas(drawing_mode="polygon")`.
  - ROI setup mode pauses main loop (`stop_when_roi_setup=True`).
- Upload mode:
  - Runs synchronous video inference loop.
  - Uses playback FPS pacing.
  - Current behavior: desktop upload uses `use_roi=False`.

## 6) Mobile Checkout (`pages_mobile/checkout_mobile_page.py`)
- Constants:
  - `MIN_AREA = 2500`
  - `DETECT_EVERY_N_FRAMES = 5` (UI slider)
  - `MATCH_THRESHOLD = 0.62`
  - `COUNT_COOLDOWN_SECONDS = 1.0` (UI slider)
- Input source:
  - `라이브 카메라`
  - `업로드 영상`
- Camera handling:
  - camera probing + index select (Iriun usage path)
  - macOS backend fallback logic
- Live ROI:
  - Rectangle ROI only (`drawing_mode="rect"`).
  - ROI setup / redraw / clear controls.
  - Inference uses ROI if set (`roi_source="live"`).
- Upload ROI:
  - Rectangle ROI only.
  - Must apply ROI before enabling upload inference start button.
  - Inference requires ROI (`roi_source="upload"`).

## 7) Add Product (`pages/1_Add_Product.py`)
- Admin page for product embedding registration.
- Upload 1-3 product images with product name.
- Creates raw embeddings and appends DB arrays.
- Updates FAISS index incrementally or rebuilds when mismatched.
- Note: inference logic is still partly duplicated here (not fully core-integrated).

## 8) Validate Bill (`pages/3_Validate_Bill.py`)
- Uses `st.session_state.billing_items`.
- Supports quantity +/- and item delete.
- Confirm action resets billing items.
- Adapts navigation defaults based on `navigation_mode` (`desktop`/`mobile`).

## 9) Session State Highlights
- Shared:
  - `billing_items`, `item_scores`, `last_seen_at`
  - `last_label`, `last_score`, `last_status`, `last_fps`, `last_frame_time`
  - `labels`, `weighted_db`, `faiss_index`, `db_mtime`, `index_mtime`
- Desktop ROI:
  - `roi_poly_norm`, `roi_frame`, `roi_setup`, `roi_status`
  - `roi_occupied`, `roi_empty_frames`
- Mobile:
  - `mobile_camera_index`, `mobile_active_camera_index`
  - `mobile_detect_every_n_frames`, `mobile_count_cooldown_seconds`
  - `mobile_upload_roi_*`, `mobile_live_roi_*`

## 10) Runtime and Secrets
- Launch:
  - Desktop: `./run.sh` or `streamlit run app.py`
  - Mobile: `./run_mobile.sh` or `streamlit run mobile_app.py`
- `.env` keys:
  - `HF_TOKEN`
  - `HUGGINGFACE_HUB_TOKEN`
- Generated artifacts (ignored):
  - `data/embeddings.npy`
  - `data/labels.npy`
  - `data/faiss_index.bin`
  - `data/adapter_model.safetensors`

## 11) Current Constraints / Debt
- Upload inference is synchronous (no progress/cancel controls).
- Desktop upload currently does not apply ROI.
- Add Product has duplicated model/embedding logic vs `checkout_core`.
- No formal automated test suite currently.
- `requirements.txt` has unpinned versions (reproducibility risk).

## 12) Roboflow Status
- Roboflow cart ROI auto-detection is not integrated in current code.
- Existing ROI flow is user-drawn ROI.

