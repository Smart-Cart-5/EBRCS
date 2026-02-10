import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import html
import sys
import time

import cv2
import faiss
import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image
from streamlit.elements.lib import image_utils
from streamlit.elements.lib.layout_utils import LayoutConfig
from streamlit_drawable_canvas import st_canvas

from checkout_core.counting import ensure_last_seen_at_state
from checkout_core.frame_processor import create_bg_subtractor, process_checkout_frame
from checkout_core.inference import (
    build_or_load_index as core_build_or_load_index,
    load_db as core_load_db,
    load_models as core_load_models,
)
from checkout_core.video_input import persist_uploaded_video
from mobile_nav import MOBILE_NAV_ITEMS, MOBILE_NAV_TO_PAGE
from ui_theme import apply_theme

DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ADAPTER_DIR = DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

MIN_AREA = 2500
DETECT_EVERY_N_FRAMES = 5
MATCH_THRESHOLD = 0.62
COUNT_COOLDOWN_SECONDS = 1.0
STREAM_TARGET_WIDTH = 960
CAMERA_SCAN_MAX_INDEX = 8
VIDEO_FILE_TYPES = ["mp4", "mov", "avi", "mkv"]
ROI_DISPLAY_MAX_WIDTH = 920
ROI_DISPLAY_MAX_HEIGHT = 560

video_placeholder = None
status_placeholder = None
billing_placeholder = None

# Compatibility patch for streamlit>=1.54 where image_to_url moved.
if not hasattr(st_image, "image_to_url"):
    def _image_to_url(image, width, clamp, channels, output_format, image_id):
        layout = LayoutConfig(width=width)
        return image_utils.image_to_url(
            image, layout, clamp, channels, output_format, image_id
        )

    st_image.image_to_url = _image_to_url


@st.cache_resource(show_spinner=False)
def load_models():
    return core_load_models(ADAPTER_DIR)


@st.cache_data(show_spinner=False)
def load_db(dino_dim, clip_dim, embeddings_mtime, labels_mtime):
    return core_load_db(
        dino_dim,
        clip_dim,
        EMBEDDINGS_PATH,
        LABELS_PATH,
        embeddings_mtime,
        labels_mtime,
    )


def build_or_load_index(weighted_db: np.ndarray) -> faiss.IndexFlatIP:
    return core_build_or_load_index(weighted_db, FAISS_INDEX_PATH)


def resize_to_stream_size(frame):
    h, w = frame.shape[:2]
    stream_size = st.session_state.get("mobile_stream_size")
    if stream_size is None:
        if w == 0:
            return frame
        scale = STREAM_TARGET_WIDTH / float(w)
        target_w = STREAM_TARGET_WIDTH
        target_h = max(1, int(h * scale))
        stream_size = (target_w, target_h)
        st.session_state.mobile_stream_size = stream_size

    target_w, target_h = stream_size
    if (w, h) == (target_w, target_h):
        return frame

    return cv2.resize(frame, (target_w, target_h))


def _video_signature(uploaded_video) -> str | None:
    if uploaded_video is None:
        return None
    name = str(getattr(uploaded_video, "name", ""))
    size = str(getattr(uploaded_video, "size", ""))
    return f"{name}:{size}"


def _extract_rect_from_obj(obj):
    if not isinstance(obj, dict) or obj.get("type") != "rect":
        return None

    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
    height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
    if width <= 1 or height <= 1:
        return None

    return left, top, left + width, top + height


def clear_mobile_upload_roi(clear_frame: bool = False) -> None:
    st.session_state.mobile_upload_roi_canvas_seed += 1
    st.session_state.mobile_upload_roi_norm = None
    st.session_state.mobile_upload_roi_ready = False
    if clear_frame:
        st.session_state.mobile_upload_roi_frame = None


def clear_mobile_live_roi(clear_frame: bool = False) -> None:
    st.session_state.mobile_live_roi_canvas_seed += 1
    st.session_state.mobile_live_roi_norm = None
    st.session_state.mobile_live_roi_ready = False
    if clear_frame:
        st.session_state.mobile_live_roi_frame = None


def capture_mobile_live_roi_frame(camera_index: int) -> None:
    cap = open_camera(int(camera_index))
    if cap is None:
        st.session_state.mobile_live_roi_status = {
            "level": "error",
            "text": f"카메라 인덱스 {camera_index}를 열 수 없어 ROI 프레임을 캡처하지 못했습니다.",
        }
        return

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        st.session_state.mobile_live_roi_status = {
            "level": "error",
            "text": "카메라 프레임 캡처에 실패했습니다.",
        }
        return

    st.session_state.mobile_stream_size = None
    frame = resize_to_stream_size(frame)
    st.session_state.mobile_live_roi_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.mobile_live_roi_status = {
        "level": "success",
        "text": "라이브 ROI 기준 프레임을 캡처했습니다. 사각형 영역을 그린 뒤 [ROI 적용]을 누르세요.",
    }


def capture_mobile_upload_roi_frame(uploaded_video) -> None:
    if uploaded_video is None:
        st.session_state.mobile_upload_roi_status = {
            "level": "info",
            "text": "ROI 설정을 위해 업로드 영상을 먼저 선택하세요.",
        }
        return

    temp_video_path = persist_uploaded_video(uploaded_video, prefix="mobile_roi_preview_")
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.session_state.mobile_upload_roi_status = {
            "level": "error",
            "text": "ROI 프레임 추출을 위해 업로드 영상을 열 수 없습니다.",
        }
        return

    ret, frame = cap.read()
    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    if not ret or frame is None:
        st.session_state.mobile_upload_roi_status = {
            "level": "error",
            "text": "업로드 영상에서 ROI 기준 프레임을 읽지 못했습니다.",
        }
        return

    st.session_state.mobile_stream_size = None
    frame = resize_to_stream_size(frame)
    st.session_state.mobile_upload_roi_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.mobile_upload_roi_status = {
        "level": "success",
        "text": "ROI 기준 프레임을 불러왔습니다. 사각형 영역을 그린 뒤 [ROI 적용]을 누르세요.",
    }


def get_mobile_upload_roi_polygon(frame_shape):
    roi_norm = st.session_state.get("mobile_upload_roi_norm")
    if not roi_norm:
        return None

    h, w = frame_shape[:2]
    pts = []
    for x_norm, y_norm in roi_norm:
        x = int(max(0, min(1, x_norm)) * w)
        y = int(max(0, min(1, y_norm)) * h)
        pts.append([x, y])

    if len(pts) < 4:
        return None

    return np.array(pts, dtype=np.int32)


def get_mobile_live_roi_polygon(frame_shape):
    roi_norm = st.session_state.get("mobile_live_roi_norm")
    if not roi_norm:
        return None

    h, w = frame_shape[:2]
    pts = []
    for x_norm, y_norm in roi_norm:
        x = int(max(0, min(1, x_norm)) * w)
        y = int(max(0, min(1, y_norm)) * h)
        pts.append([x, y])

    if len(pts) < 4:
        return None

    return np.array(pts, dtype=np.int32)


def render_mobile_upload_roi_editor(uploaded_video) -> bool:
    if uploaded_video is None:
        st.session_state.mobile_upload_roi_ready = False
        st.info("추론 전에 ROI를 설정하려면 영상을 먼저 업로드하세요.")
        return False

    signature = _video_signature(uploaded_video)
    if st.session_state.get("mobile_upload_roi_video_signature") != signature:
        st.session_state.mobile_upload_roi_video_signature = signature
        clear_mobile_upload_roi(clear_frame=True)
        capture_mobile_upload_roi_frame(uploaded_video)

    roi_status = st.session_state.get("mobile_upload_roi_status")
    if isinstance(roi_status, dict) and roi_status.get("text"):
        level = roi_status.get("level", "info")
        if level == "success":
            st.success(roi_status["text"])
        elif level == "warning":
            st.warning(roi_status["text"])
        elif level == "error":
            st.error(roi_status["text"])
        else:
            st.info(roi_status["text"])

    row_col1, row_col2 = st.columns([1, 1], gap="small")
    with row_col1:
        if st.button(
            "ROI 프레임 다시 불러오기",
            key="mobile_upload_roi_recapture",
            use_container_width=True,
        ):
            clear_mobile_upload_roi(clear_frame=True)
            capture_mobile_upload_roi_frame(uploaded_video)
            st.rerun()
    with row_col2:
        if st.button(
            "ROI 해제",
            key="mobile_upload_roi_clear",
            use_container_width=True,
        ):
            clear_mobile_upload_roi(clear_frame=False)
            st.session_state.mobile_upload_roi_status = {
                "level": "info",
                "text": "업로드 ROI가 해제되었습니다.",
            }
            st.rerun()

    roi_frame = st.session_state.get("mobile_upload_roi_frame")
    if roi_frame is None:
        st.session_state.mobile_upload_roi_ready = False
        st.info("ROI 프레임을 불러오지 못했습니다. [ROI 프레임 다시 불러오기]를 눌러주세요.")
        return False

    height, width = roi_frame.shape[:2]
    scale = min(
        ROI_DISPLAY_MAX_WIDTH / float(width),
        ROI_DISPLAY_MAX_HEIGHT / float(height),
        1.0,
    )
    disp_w = max(1, int(width * scale))
    disp_h = max(1, int(height * scale))
    resized = cv2.resize(roi_frame, (disp_w, disp_h))

    canvas_key = f"mobile_upload_roi_canvas_{st.session_state.mobile_upload_roi_canvas_seed}"
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.08)",
        stroke_width=2,
        stroke_color="#10B981",
        background_image=Image.fromarray(resized),
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key=canvas_key,
    )

    objects = (canvas_result.json_data or {}).get("objects", [])
    rect = None
    if objects:
        for obj in reversed(objects):
            rect = _extract_rect_from_obj(obj)
            if rect is not None:
                break

    if rect is not None:
        x1, y1, x2, y2 = rect
        x1 = float(np.clip(x1, 0, disp_w - 1))
        y1 = float(np.clip(y1, 0, disp_h - 1))
        x2 = float(np.clip(x2, 0, disp_w - 1))
        y2 = float(np.clip(y2, 0, disp_h - 1))
        if x2 > x1 + 1 and y2 > y1 + 1:
            roi_norm = [
                (x1 / disp_w, y1 / disp_h),
                (x2 / disp_w, y1 / disp_h),
                (x2 / disp_w, y2 / disp_h),
                (x1 / disp_w, y2 / disp_h),
            ]
            if st.button("ROI 적용", key="mobile_upload_roi_apply", type="primary", use_container_width=True):
                st.session_state.mobile_upload_roi_norm = roi_norm
                st.session_state.mobile_upload_roi_ready = True
                st.session_state.mobile_upload_roi_status = {
                    "level": "success",
                    "text": "업로드 ROI가 저장되었습니다. 이제 해당 영역에서만 추론됩니다.",
                }
                st.rerun()
        else:
            st.caption("너무 작은 ROI는 사용할 수 없습니다. 사각형을 다시 그려주세요.")
    else:
        st.caption("사각형 ROI를 그리면 [ROI 적용] 버튼이 활성화됩니다.")

    ready = bool(st.session_state.get("mobile_upload_roi_norm"))
    st.session_state.mobile_upload_roi_ready = ready
    return ready


def render_mobile_live_roi_editor() -> bool:
    roi_status = st.session_state.get("mobile_live_roi_status")
    if isinstance(roi_status, dict) and roi_status.get("text"):
        level = roi_status.get("level", "info")
        if level == "success":
            st.success(roi_status["text"])
        elif level == "warning":
            st.warning(roi_status["text"])
        elif level == "error":
            st.error(roi_status["text"])
        else:
            st.info(roi_status["text"])

    row_col1, row_col2 = st.columns([1, 1], gap="small")
    with row_col1:
        if st.button(
            "라이브 화면으로 돌아가기",
            key="mobile_live_roi_back",
            use_container_width=True,
        ):
            st.session_state.mobile_live_roi_setup = False
            st.rerun()
    with row_col2:
        if st.button(
            "프레임 다시 캡처",
            key="mobile_live_roi_recapture",
            use_container_width=True,
        ):
            clear_mobile_live_roi(clear_frame=True)
            capture_mobile_live_roi_frame(int(st.session_state.get("mobile_camera_index", 0)))
            st.rerun()

    roi_frame = st.session_state.get("mobile_live_roi_frame")
    if roi_frame is None:
        st.session_state.mobile_live_roi_ready = False
        st.info("ROI 설정을 위해 카메라 프레임이 필요합니다. [프레임 다시 캡처]를 눌러주세요.")
        return False

    height, width = roi_frame.shape[:2]
    scale = min(
        ROI_DISPLAY_MAX_WIDTH / float(width),
        ROI_DISPLAY_MAX_HEIGHT / float(height),
        1.0,
    )
    disp_w = max(1, int(width * scale))
    disp_h = max(1, int(height * scale))
    resized = cv2.resize(roi_frame, (disp_w, disp_h))

    canvas_key = f"mobile_live_roi_canvas_{st.session_state.mobile_live_roi_canvas_seed}"
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.08)",
        stroke_width=2,
        stroke_color="#F59E0B",
        background_image=Image.fromarray(resized),
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key=canvas_key,
    )

    objects = (canvas_result.json_data or {}).get("objects", [])
    rect = None
    if objects:
        for obj in reversed(objects):
            rect = _extract_rect_from_obj(obj)
            if rect is not None:
                break

    if rect is not None:
        x1, y1, x2, y2 = rect
        x1 = float(np.clip(x1, 0, disp_w - 1))
        y1 = float(np.clip(y1, 0, disp_h - 1))
        x2 = float(np.clip(x2, 0, disp_w - 1))
        y2 = float(np.clip(y2, 0, disp_h - 1))
        if x2 > x1 + 1 and y2 > y1 + 1:
            roi_norm = [
                (x1 / disp_w, y1 / disp_h),
                (x2 / disp_w, y1 / disp_h),
                (x2 / disp_w, y2 / disp_h),
                (x1 / disp_w, y2 / disp_h),
            ]
            if st.button("ROI 적용", key="mobile_live_roi_apply", type="primary", use_container_width=True):
                st.session_state.mobile_live_roi_norm = roi_norm
                st.session_state.mobile_live_roi_ready = True
                st.session_state.mobile_live_roi_setup = False
                st.session_state.mobile_live_roi_status = {
                    "level": "success",
                    "text": "라이브 ROI가 저장되었습니다. 라이브 스트리밍에서 해당 영역만 추론됩니다.",
                }
                st.rerun()
        else:
            st.caption("너무 작은 ROI는 사용할 수 없습니다. 사각형을 다시 그려주세요.")
    else:
        st.caption("사각형 ROI를 그리면 [ROI 적용] 버튼이 활성화됩니다.")

    ready = bool(st.session_state.get("mobile_live_roi_norm"))
    st.session_state.mobile_live_roi_ready = ready
    return ready


def open_camera(index: int):
    backends = []
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(None)

    for backend in backends:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)

        if cap.isOpened():
            return cap
        cap.release()

    return None


def probe_available_cameras(max_index: int):
    cameras = []
    for idx in range(max_index + 1):
        cap = open_camera(idx)
        if cap is None:
            continue

        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            cameras.append({"index": idx, "width": int(w), "height": int(h)})
        cap.release()

    return cameras


def update_status_ui() -> None:
    if status_placeholder is None:
        return

    with status_placeholder.container():
        last_label = html.escape(str(st.session_state.get("last_label", "-")))
        last_score = float(st.session_state.get("last_score", 0.0))
        last_status = html.escape(str(st.session_state.get("last_status", "대기")))
        fps = float(st.session_state.get("last_fps", 0.0))
        source_mode = st.session_state.get("mobile_input_source", "라이브 카메라")
        if source_mode == "업로드 영상":
            source_value = "업로드 영상"
        else:
            source_value = str(
                int(
                    st.session_state.get(
                        "mobile_active_camera_index",
                        st.session_state.get("mobile_camera_index", 0),
                    )
                )
            )

        score_percent = int(max(0.0, min(1.0, last_score)) * 100)

        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric-card card-hover">
                <div style="display:flex; align-items:center; gap:10px;">
                  <span class="warn-orange" style="font-size:18px;">🕐</span>
                  <span class="metric-label">최근 인식</span>
                </div>
                <div class="metric-value">{last_label}</div>
              </div>
              <div class="metric-card card-hover">
                <div style="display:flex; align-items:center; gap:10px;">
                  <span class="info-blue" style="font-size:18px;">📈</span>
                  <span class="metric-label">유사도</span>
                </div>
                <div class="metric-value">{score_percent}%</div>
              </div>
              <div class="metric-card card-hover">
                <div style="display:flex; align-items:center; gap:10px;">
                  <span class="status-good" style="font-size:18px;">✅</span>
                  <span class="metric-label">상태</span>
                </div>
                <div class="metric-value">{last_status}</div>
              </div>
              <div class="metric-card card-hover">
                <div style="display:flex; align-items:center; gap:10px;">
                  <span style="font-size:18px; color:#8B5CF6;">📷</span>
                  <span class="metric-label">입력 소스</span>
                </div>
                <div class="metric-value">{source_value}</div>
              </div>
              <div class="metric-card card-hover">
                <div style="display:flex; align-items:center; gap:10px;">
                  <span style="font-size:18px; color:#8B5CF6;">⚡</span>
                  <span class="metric-label">FPS</span>
                </div>
                <div class="metric-value">{fps:.1f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def update_billing_ui() -> None:
    if billing_placeholder is None:
        return

    with billing_placeholder.container():
        items = st.session_state.billing_items
        total_count = int(sum(items.values()))

        st.markdown(
            f"""
            <div class="soft-card" style="padding:0; overflow:hidden;">
              <div style="padding:18px 18px 14px 18px; border-bottom:1px solid rgba(0,0,0,0.08); display:flex; justify-content:space-between; align-items:center;">
                <h3 class="card-title" style="font-size:20px; margin:0;">📦 인식된 상품</h3>
                <span class="pill-badge" style="background:#FFB74D; color:#fff;">{total_count}개</span>
              </div>
              <div style="padding:14px 16px;">
            """,
            unsafe_allow_html=True,
        )

        if not items:
            st.markdown(
                """
                <div class="soft-card" style="background:#FBFBFB; text-align:center; padding:18px;">
                  <div style="font-size:36px; margin-bottom:6px;">🧺</div>
                  <div class="card-subtitle">아직 인식된 상품이 없습니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="product-list-wrap">', unsafe_allow_html=True)
            for name, qty in items.items():
                safe_name = html.escape(str(name))
                score = float(st.session_state.item_scores.get(name, 0.0))
                conf_pct = int(max(0.0, min(1.0, score)) * 100)
                st.markdown(
                    f"""
                    <div class="product-item">
                      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                        <div>
                          <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                            <span style="font-size:16px; font-weight:600; color:#030213;">{safe_name}</span>
                            <span class="pill-badge confidence-badge">{conf_pct}%</span>
                          </div>
                          <div class="card-subtitle">신뢰도: {conf_pct}%</div>
                        </div>
                        <div class="count-chip">{int(qty)}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
              <div style="height:6px;"></div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin:8px 2px 14px 2px;">
                <span style="font-size:16px; font-weight:600; color:#030213;">총 상품 수</span>
                <span style="font-size:30px; font-weight:700; color:#FF8A65;">{total_count}개</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def refresh_camera_options() -> None:
    max_index = int(st.session_state.get("mobile_scan_max_index", CAMERA_SCAN_MAX_INDEX))
    st.session_state.mobile_camera_options = probe_available_cameras(max_index)

    available_indices = [opt["index"] for opt in st.session_state.mobile_camera_options]
    current_index = int(st.session_state.get("mobile_camera_index", 0))
    if available_indices and current_index not in available_indices:
        st.session_state.mobile_camera_index = available_indices[0]


def init_app_state(bundle) -> None:
    if "billing_items" not in st.session_state:
        st.session_state.billing_items = {}
    if "item_scores" not in st.session_state:
        st.session_state.item_scores = {}
    ensure_last_seen_at_state(st.session_state)
    if "last_label" not in st.session_state:
        st.session_state.last_label = "-"
    if "last_score" not in st.session_state:
        st.session_state.last_score = 0.0
    if "last_status" not in st.session_state:
        st.session_state.last_status = "대기"
    if "last_fps" not in st.session_state:
        st.session_state.last_fps = 0.0
    if "last_frame_time" not in st.session_state:
        st.session_state.last_frame_time = time.perf_counter()

    if "mobile_camera_index" not in st.session_state:
        st.session_state.mobile_camera_index = 0
    if "mobile_active_camera_index" not in st.session_state:
        st.session_state.mobile_active_camera_index = 0
    if "mobile_detect_every_n_frames" not in st.session_state:
        st.session_state.mobile_detect_every_n_frames = DETECT_EVERY_N_FRAMES
    if "mobile_count_cooldown_seconds" not in st.session_state:
        st.session_state.mobile_count_cooldown_seconds = COUNT_COOLDOWN_SECONDS
    if "mobile_stream_size" not in st.session_state:
        st.session_state.mobile_stream_size = None
    if "mobile_scan_max_index" not in st.session_state:
        st.session_state.mobile_scan_max_index = CAMERA_SCAN_MAX_INDEX
    if "mobile_camera_options" not in st.session_state:
        st.session_state.mobile_camera_options = []
    if "mobile_upload_roi_norm" not in st.session_state:
        st.session_state.mobile_upload_roi_norm = None
    if "mobile_upload_roi_frame" not in st.session_state:
        st.session_state.mobile_upload_roi_frame = None
    if "mobile_upload_roi_canvas_seed" not in st.session_state:
        st.session_state.mobile_upload_roi_canvas_seed = 0
    if "mobile_upload_roi_status" not in st.session_state:
        st.session_state.mobile_upload_roi_status = None
    if "mobile_upload_roi_video_signature" not in st.session_state:
        st.session_state.mobile_upload_roi_video_signature = None
    if "mobile_upload_roi_ready" not in st.session_state:
        st.session_state.mobile_upload_roi_ready = False
    if "mobile_live_roi_norm" not in st.session_state:
        st.session_state.mobile_live_roi_norm = None
    if "mobile_live_roi_frame" not in st.session_state:
        st.session_state.mobile_live_roi_frame = None
    if "mobile_live_roi_canvas_seed" not in st.session_state:
        st.session_state.mobile_live_roi_canvas_seed = 0
    if "mobile_live_roi_status" not in st.session_state:
        st.session_state.mobile_live_roi_status = None
    if "mobile_live_roi_setup" not in st.session_state:
        st.session_state.mobile_live_roi_setup = False
    if "mobile_live_roi_ready" not in st.session_state:
        st.session_state.mobile_live_roi_ready = False

    if not st.session_state.mobile_camera_options:
        refresh_camera_options()

    embeddings_mtime = os.path.getmtime(EMBEDDINGS_PATH) if os.path.exists(EMBEDDINGS_PATH) else 0
    labels_mtime = os.path.getmtime(LABELS_PATH) if os.path.exists(LABELS_PATH) else 0
    db_mtime = (embeddings_mtime, labels_mtime)

    if st.session_state.get("db_mtime") != db_mtime:
        weighted_db, labels = load_db(
            bundle["dino_dim"],
            bundle["clip_dim"],
            embeddings_mtime,
            labels_mtime,
        )
        st.session_state.labels = labels
        st.session_state.weighted_db = weighted_db
        st.session_state.db_mtime = db_mtime

    if "faiss_index" not in st.session_state or st.session_state.get("index_mtime") != db_mtime:
        st.session_state.faiss_index = build_or_load_index(st.session_state.weighted_db)
        st.session_state.index_mtime = db_mtime


def render_input_config_ui(source_mode: str):
    uploaded_video = None

    if source_mode == "라이브 카메라":
        st.markdown(
            """
            <div class="soft-card">
              <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:8px;">
                <div class="icon-square" style="background:linear-gradient(135deg,#3B82F6,#2563EB);">📷</div>
                <div>
                  <div class="card-title">카메라 설정</div>
                  <div class="card-subtitle">iPhone Iriun 앱이 실행 중이면 목록 새로고침 후 카메라를 선택하세요.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        scan_col, refresh_col = st.columns([2, 1], gap="small")
        with scan_col:
            st.session_state.mobile_scan_max_index = st.slider(
                "탐색 최대 인덱스",
                min_value=1,
                max_value=20,
                value=int(st.session_state.mobile_scan_max_index),
                step=1,
                key="mobile_scan_max_index_slider",
            )
        with refresh_col:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            if st.button("목록 새로고침", key="mobile_refresh_camera_list", use_container_width=True):
                refresh_camera_options()
                st.rerun()

        options = st.session_state.mobile_camera_options
        if options:
            option_map = {
                f"{opt['index']}번 카메라 ({opt['width']}x{opt['height']})": opt["index"] for opt in options
            }
            labels = list(option_map.keys())
            current_index = int(st.session_state.mobile_camera_index)

            default_idx = 0
            for idx, label in enumerate(labels):
                if option_map[label] == current_index:
                    default_idx = idx
                    break

            selected_label = st.selectbox(
                "사용할 카메라",
                options=labels,
                index=default_idx,
                key="mobile_camera_selector",
            )
            st.session_state.mobile_camera_index = int(option_map[selected_label])
        else:
            st.warning("감지된 카메라가 없습니다. Iriun 앱을 실행한 뒤 목록을 새로고침하세요.")
            st.session_state.mobile_camera_index = int(
                st.number_input(
                    "수동 카메라 인덱스",
                    min_value=0,
                    max_value=20,
                    value=int(st.session_state.mobile_camera_index),
                    step=1,
                    key="mobile_camera_manual_index",
                )
            )
    else:
        st.markdown(
            """
            <div class="soft-card">
              <div style="display:flex; align-items:flex-start; gap:12px;">
                <div class="icon-square" style="background:linear-gradient(135deg,#10B981,#059669);">🎬</div>
                <div>
                  <div class="card-title">영상 업로드 추론</div>
                  <div class="card-subtitle">업로드 영상을 재생 속도에 맞춰 실시간처럼 추론합니다.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_video = st.file_uploader(
            "추론할 영상 업로드",
            type=VIDEO_FILE_TYPES,
            key="mobile_checkout_uploaded_video",
        )
        st.markdown(
            """
            <div class="soft-card" style="margin-top:10px;">
              <div style="display:flex; align-items:flex-start; gap:12px;">
                <div class="icon-square" style="background:linear-gradient(135deg,#F59E0B,#D97706);">▭</div>
                <div>
                  <div class="card-title">업로드 ROI 설정 (사각형)</div>
                  <div class="card-subtitle">추론 시작 전에 사각형 ROI를 적용해야 하며, 해당 영역에서만 추론/카운트됩니다.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_mobile_upload_roi_editor(uploaded_video)

    st.session_state.mobile_detect_every_n_frames = st.slider(
        "추론 간격 (프레임)",
        min_value=1,
        max_value=15,
        value=int(st.session_state.mobile_detect_every_n_frames),
        step=1,
        key="mobile_detect_every_slider",
        help="값이 클수록 추론 빈도가 줄어 CPU 사용량이 감소합니다.",
    )

    st.session_state.mobile_count_cooldown_seconds = st.slider(
        "중복 카운트 방지 쿨다운 (초)",
        min_value=0.2,
        max_value=10.0,
        value=float(st.session_state.mobile_count_cooldown_seconds),
        step=0.1,
        format="%.1f초",
        key="mobile_cooldown_slider",
        help="Iriun FPS 변동이 크면 값을 늘려 중복 카운트를 줄이세요.",
    )

    return uploaded_video


def run_checkout_capture_loop(
    model_bundle,
    cap,
    *,
    use_roi: bool,
    detect_every_n_frames: int,
    cooldown_seconds: float,
    playback_fps: float | None = None,
    roi_source: str = "upload",
    stop_when_live_roi_setup: bool = False,
) -> None:
    bg_subtractor = create_bg_subtractor()
    frame_count = 0
    frame_interval = 1.0 / playback_fps if playback_fps and playback_fps > 0 else None
    read_failures = 0

    try:
        while True:
            loop_started = time.perf_counter()
            if stop_when_live_roi_setup and st.session_state.get("mobile_live_roi_setup"):
                break
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures >= 30:
                    break
                continue

            read_failures = 0
            frame = resize_to_stream_size(frame)
            frame_count += 1

            now = time.perf_counter()
            delta = now - st.session_state.last_frame_time
            if delta > 0:
                st.session_state.last_fps = 1.0 / delta
            st.session_state.last_frame_time = now

            roi_poly = None
            if use_roi:
                if roi_source == "live":
                    roi_poly = get_mobile_live_roi_polygon(frame.shape)
                else:
                    roi_poly = get_mobile_upload_roi_polygon(frame.shape)
            display_frame = process_checkout_frame(
                frame=frame,
                frame_count=frame_count,
                bg_subtractor=bg_subtractor,
                model_bundle=model_bundle,
                faiss_index=st.session_state.faiss_index,
                labels=st.session_state.labels,
                state=st.session_state,
                min_area=MIN_AREA,
                detect_every_n_frames=detect_every_n_frames,
                match_threshold=MATCH_THRESHOLD,
                cooldown_seconds=cooldown_seconds,
                roi_poly=roi_poly,
                roi_entry_mode=use_roi,
            )

            if video_placeholder is not None:
                video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
            update_status_ui()
            update_billing_ui()

            if frame_interval is not None:
                elapsed = time.perf_counter() - loop_started
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
    finally:
        cap.release()


def run_checkout_pipeline(
    model_bundle,
    camera_index: int,
    detect_every_n_frames: int,
    cooldown_seconds: float,
) -> None:
    cap = open_camera(camera_index)
    if cap is None:
        st.error(f"카메라 인덱스 {camera_index}를 열 수 없습니다. Iriun 실행 상태를 확인하세요.")
        st.info("Iriun Desktop(iPhone) 앱이 모두 실행 중인지, 같은 네트워크인지 확인하세요.")
        return

    st.session_state.mobile_active_camera_index = camera_index
    st.session_state.mobile_stream_size = None

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    run_checkout_capture_loop(
        model_bundle,
        cap,
        use_roi=True,
        detect_every_n_frames=detect_every_n_frames,
        cooldown_seconds=cooldown_seconds,
        roi_source="live",
        stop_when_live_roi_setup=True,
    )


def run_uploaded_video_pipeline(
    model_bundle,
    uploaded_video,
    *,
    detect_every_n_frames: int,
    cooldown_seconds: float,
) -> None:
    if st.session_state.get("mobile_upload_roi_norm") is None:
        st.error("업로드 추론 전에 사각형 ROI를 먼저 적용하세요.")
        return

    temp_video_path = persist_uploaded_video(uploaded_video, prefix="mobile_checkout_")
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        os.remove(temp_video_path)
        st.error("업로드한 영상을 열 수 없습니다.")
        return

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 240:
            fps = 15.0

        st.session_state.mobile_stream_size = None
        st.session_state.last_status = "영상 추론 중"

        run_checkout_capture_loop(
            model_bundle,
            cap,
            use_roi=True,
            detect_every_n_frames=detect_every_n_frames,
            cooldown_seconds=cooldown_seconds,
            playback_fps=fps,
            roi_source="upload",
        )
        st.success("업로드 영상 추론이 완료되었습니다.")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def render_mobile_checkout() -> None:
    global video_placeholder
    global status_placeholder
    global billing_placeholder

    apply_theme(
        page_title="모바일 체크아웃",
        page_icon="📷",
        current_nav="📷 모바일 체크아웃",
        nav_items=MOBILE_NAV_ITEMS,
        nav_to_page=MOBILE_NAV_TO_PAGE,
        nav_key_prefix="mobile",
    )

    st.session_state.navigation_mode = "mobile"
    st.session_state.home_page_path = "mobile_app.py"
    st.session_state.checkout_page_path = "pages/4_Checkout_Mobile.py"

    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:16px;">
          <div>
            <h1 class="page-title">모바일 체크아웃</h1>
            <p class="subtitle-text">Iriun Webcam(iPhone) 입력과 업로드 영상에서 사각형 ROI 기반 상품 인식을 수행합니다.</p>
          </div>
          <span class="pill-badge" style="background:#DBEAFE; color:#1E40AF;">Iriun Webcam</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            """
            <div class="soft-card loading-card" style="max-width:620px; margin:0 auto 14px auto;">
              <div class="loading-spinner"></div>
              <div class="card-title" style="font-size:36px; margin-bottom:4px;">임베딩 모델 로딩 중</div>
              <div class="card-subtitle">
                잠시만 기다려 주세요
                <span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>
              </div>
              <div class="loading-progress"></div>
              <div class="loading-caption">모델 다운로드/초기화 시간에 따라 로딩 시간이 달라질 수 있습니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    model_bundle = load_models()
    loading_placeholder.empty()

    if model_bundle["lora_loaded"]:
        st.success("모델 준비 완료: DINO LoRA + CLIP")
    else:
        warning = "모델 준비 완료: DINO 베이스 + CLIP"
        if model_bundle["lora_error"]:
            warning += f" · LoRA 로드 실패({model_bundle['lora_error']})"
        st.warning(warning)

    try:
        init_app_state(model_bundle)
    except Exception as exc:
        st.error(f"Embedding load error: {exc}")
        st.info("모바일 체크아웃 전에 Add Product에서 상품 임베딩을 등록하세요.")
        st.stop()

    col_camera, col_panel = st.columns([2, 1], gap="large")
    source_mode = st.radio(
        "입력 소스",
        options=["라이브 카메라", "업로드 영상"],
        horizontal=True,
        key="mobile_input_source",
    )
    is_live_source = source_mode == "라이브 카메라"
    if not is_live_source and st.session_state.get("mobile_live_roi_setup"):
        st.session_state.mobile_live_roi_setup = False
    uploaded_video = None
    start_video_inference = False

    if is_live_source and st.session_state.get("mobile_live_roi_setup"):
        with col_camera:
            st.markdown(
                """
                <div class="soft-card">
                  <h2 class="section-title" style="margin-bottom:8px;">▭ 라이브 ROI 설정 (사각형)</h2>
                  <p class="card-subtitle">사각형 ROI를 적용하면 라이브 스트리밍에서 해당 영역만 추론합니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_mobile_live_roi_editor()

        with col_panel:
            status_placeholder = st.empty()
            billing_placeholder = st.empty()
            update_status_ui()
            update_billing_ui()
            if st.button("체크아웃 완료", type="primary", use_container_width=True, key="mobile_checkout_done"):
                st.switch_page("pages/3_Validate_Bill.py")
    else:
        with col_camera:
            st.markdown('<div class="camera-shell">', unsafe_allow_html=True)
            st.markdown(
                (
                    '<div class="live-badge"><span class="live-dot"></span>Live</div>'
                    if is_live_source
                    else '<div class="live-badge"><span class="live-dot"></span>Video</div>'
                ),
                unsafe_allow_html=True,
            )
            video_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            uploaded_video = render_input_config_ui(source_mode)
            if is_live_source:
                st.markdown(
                    """
                    <div class="soft-card" style="margin-top:10px;">
                      <div style="display:flex; align-items:flex-start; gap:12px;">
                        <div class="icon-square" style="background:linear-gradient(135deg,#F59E0B,#D97706);">▭</div>
                        <div>
                          <div class="card-title">라이브 ROI 설정 (사각형)</div>
                          <div class="card-subtitle">사각형 ROI를 설정하면 해당 영역에서만 라이브 추론/카운트가 수행됩니다.</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                live_roi_status = st.session_state.get("mobile_live_roi_status")
                if isinstance(live_roi_status, dict) and live_roi_status.get("text"):
                    level = live_roi_status.get("level", "info")
                    if level == "success":
                        st.success(live_roi_status["text"])
                    elif level == "warning":
                        st.warning(live_roi_status["text"])
                    elif level == "error":
                        st.error(live_roi_status["text"])
                    else:
                        st.info(live_roi_status["text"])
                roi_col1, roi_col2, roi_col3 = st.columns([2, 1, 1], gap="small")
                with roi_col1:
                    if st.button("ROI 설정", key="mobile_live_roi_open", type="primary", use_container_width=True):
                        clear_mobile_live_roi(clear_frame=True)
                        capture_mobile_live_roi_frame(int(st.session_state.get("mobile_camera_index", 0)))
                        st.session_state.mobile_live_roi_setup = True
                        st.rerun()
                with roi_col2:
                    if st.button("ROI 다시 그리기", key="mobile_live_roi_redraw", use_container_width=True):
                        clear_mobile_live_roi(clear_frame=False)
                        st.session_state.mobile_live_roi_setup = True
                        st.rerun()
                with roi_col3:
                    if st.button("ROI 해제", key="mobile_live_roi_clear_btn", use_container_width=True):
                        clear_mobile_live_roi(clear_frame=False)
                        st.session_state.mobile_live_roi_status = {
                            "level": "info",
                            "text": "라이브 ROI가 해제되었습니다.",
                        }
                        st.rerun()
                if st.session_state.get("mobile_live_roi_norm") is not None:
                    st.caption("라이브 ROI 적용됨: 현재 스트리밍에서 ROI 내부만 추론됩니다.")
            else:
                roi_ready = bool(st.session_state.get("mobile_upload_roi_ready"))
                start_video_inference = st.button(
                    "업로드 영상 추론 시작",
                    type="primary",
                    use_container_width=True,
                    key="mobile_run_uploaded_video",
                    disabled=(uploaded_video is None) or (not roi_ready),
                )
                if uploaded_video is not None and not roi_ready:
                    st.caption("사각형 ROI를 적용하면 추론 시작 버튼이 활성화됩니다.")

        with col_panel:
            status_placeholder = st.empty()
            billing_placeholder = st.empty()
            if st.button("체크아웃 완료", type="primary", use_container_width=True, key="mobile_checkout_done"):
                st.switch_page("pages/3_Validate_Bill.py")

        if is_live_source:
            run_checkout_pipeline(
                model_bundle=model_bundle,
                camera_index=int(st.session_state.mobile_camera_index),
                detect_every_n_frames=int(st.session_state.mobile_detect_every_n_frames),
                cooldown_seconds=float(st.session_state.mobile_count_cooldown_seconds),
            )
        elif start_video_inference and uploaded_video is not None:
            run_uploaded_video_pipeline(
                model_bundle=model_bundle,
                uploaded_video=uploaded_video,
                detect_every_n_frames=int(st.session_state.mobile_detect_every_n_frames),
                cooldown_seconds=float(st.session_state.mobile_count_cooldown_seconds),
            )
        else:
            update_status_ui()
            update_billing_ui()
