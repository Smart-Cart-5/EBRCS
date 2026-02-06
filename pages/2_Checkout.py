import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import faiss
import html
import numpy as np
import pandas as pd
import streamlit as st
import time
import torch
from streamlit_drawable_canvas import st_canvas
import streamlit.elements.image as st_image
from streamlit.elements.lib import image_utils
from streamlit.elements.lib.layout_utils import LayoutConfig
from peft import PeftModel
from PIL import Image
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ADAPTER_DIR = DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_PROCESSOR_ID = "facebook/dinov2-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

DINO_WEIGHT = 0.7
CLIP_WEIGHT = 0.3

MIN_AREA = 2500
DETECT_EVERY_N_FRAMES = 5
MATCH_THRESHOLD = 0.7
COUNT_COOLDOWN_FRAMES = 20
ROI_CLEAR_FRAMES = 8
STREAM_TARGET_WIDTH = 960
ROI_DISPLAY_MAX_WIDTH = 900
ROI_DISPLAY_MAX_HEIGHT = 520

st.set_page_config(layout="wide", page_title="실시간 결제 대시보드")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'Noto Sans KR', sans-serif;
}

.stApp {
  background: radial-gradient(1200px 500px at 10% 0%, #f3f6ff 0%, #f7fafc 45%, #ffffff 100%);
}

.dash-title {
  font-size: 34px;
  font-weight: 700;
  margin-bottom: 6px;
}

.dash-subtitle {
  color: #4b5563;
  margin-bottom: 18px;
}

.card {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px 18px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
}

.card-title {
  font-size: 16px;
  font-weight: 700;
  color: #111827;
  margin-bottom: 8px;
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.metric-card {
  background: #0f172a;
  color: #e2e8f0;
  padding: 14px 16px;
  border-radius: 14px;
}

.metric-label {
  font-size: 12px;
  color: #94a3b8;
  margin-bottom: 6px;
}

.metric-value {
  font-size: 18px;
  font-weight: 700;
  color: #f8fafc;
}

.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #e0f2fe;
  color: #0369a1;
  font-size: 12px;
  font-weight: 600;
}

.item-row {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid #eef2f7;
}

.item-row:last-child {
  border-bottom: none;
}

.item-name {
  font-weight: 600;
  color: #111827;
}

.item-qty {
  font-weight: 600;
  color: #0f172a;
}
</style>
""",
    unsafe_allow_html=True,
)

# Compatibility patch for streamlit>=1.54 where image_to_url moved.
if not hasattr(st_image, "image_to_url"):
    def _image_to_url(image, width, clamp, channels, output_format, image_id):
        layout = LayoutConfig(width=width)
        return image_utils.image_to_url(
            image, layout, clamp, channels, output_format, image_id
        )

    st_image.image_to_url = _image_to_url


def ensure_roi_state():
    if "roi_poly_norm" not in st.session_state:
        st.session_state.roi_poly_norm = None
    if "roi_frame" not in st.session_state:
        st.session_state.roi_frame = None
    if "roi_occupied" not in st.session_state:
        st.session_state.roi_occupied = False
    if "roi_empty_frames" not in st.session_state:
        st.session_state.roi_empty_frames = 0
    if "roi_canvas_seed" not in st.session_state:
        st.session_state.roi_canvas_seed = 0
    if "stream_size" not in st.session_state:
        st.session_state.stream_size = None
    if "roi_setup" not in st.session_state:
        st.session_state.roi_setup = False
    if "roi_status" not in st.session_state:
        st.session_state.roi_status = None


def reset_roi_canvas(clear_saved=False):
    st.session_state.roi_canvas_seed += 1
    if clear_saved:
        st.session_state.roi_poly_norm = None
    st.session_state.roi_occupied = False
    st.session_state.roi_empty_frames = 0


def save_roi(roi_norm):
    st.session_state.roi_poly_norm = roi_norm
    st.session_state.roi_occupied = False
    st.session_state.roi_empty_frames = 0
    st.session_state.roi_setup = False
    st.session_state.roi_status = {"level": "success", "text": "ROI가 저장되었습니다."}


def exit_roi_setup():
    st.session_state.roi_setup = False
    st.session_state.roi_status = {"level": "info", "text": "ROI 설정 모드를 종료했습니다."}


def capture_roi_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("카메라를 열 수 없습니다.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("프레임을 캡처할 수 없습니다.")
        return
    frame = resize_to_stream_size(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.roi_frame = frame_rgb


def get_roi_polygon(frame_shape):
    roi_norm = st.session_state.get("roi_poly_norm")
    if not roi_norm:
        return None
    h, w = frame_shape[:2]
    pts = []
    for x_norm, y_norm in roi_norm:
        x = int(max(0, min(1, x_norm)) * w)
        y = int(max(0, min(1, y_norm)) * h)
        pts.append([x, y])
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32)


def resize_to_stream_size(frame):
    h, w = frame.shape[:2]
    stream_size = st.session_state.get("stream_size")
    if stream_size is None:
        if w == 0:
            return frame
        scale = STREAM_TARGET_WIDTH / float(w)
        target_w = STREAM_TARGET_WIDTH
        target_h = max(1, int(h * scale))
        stream_size = (target_w, target_h)
        st.session_state.stream_size = stream_size
    target_w, target_h = stream_size
    if (w, h) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h))


def _extract_polygon_points(obj):
    points = obj.get("points") or []
    if len(points) < 3 and obj.get("path"):
        path_points = []
        for seg in obj.get("path") or []:
            if not isinstance(seg, (list, tuple)) or len(seg) < 3:
                continue
            cmd = seg[0]
            if cmd in ("M", "L"):
                path_points.append({"x": seg[1], "y": seg[2]})
        points = path_points
    if len(points) < 3:
        return None
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    path_offset = obj.get("pathOffset") or {}
    offset_x = float(path_offset.get("x", 0))
    offset_y = float(path_offset.get("y", 0))

    abs_points = []
    for p in points:
        x = left + (float(p.get("x", 0)) - offset_x) * scale_x
        y = top + (float(p.get("y", 0)) - offset_y) * scale_y
        abs_points.append((x, y))
    return abs_points


def get_hf_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        if hasattr(st, "secrets"):
            return st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_HUB_TOKEN")
    except Exception:
        return None
    return None


def safe_cls_from_output(output):
    if output is None:
        return None

    if isinstance(output, dict):
        if output.get("last_hidden_state") is not None:
            return output["last_hidden_state"][:, 0]
        if output.get("hidden_states") is not None:
            return output["hidden_states"][-1][:, 0]

    last_hidden = getattr(output, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden[:, 0]

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1][:, 0]

    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if torch.is_tensor(first) and first.dim() >= 3:
            return first[:, 0]

    return None


@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = get_hf_token()

    dino_processor = AutoImageProcessor.from_pretrained(
        DINO_PROCESSOR_ID,
        use_fast=True,
        token=token,
    )
    dino_model = AutoModel.from_pretrained(
        DINO_MODEL_NAME,
        trust_remote_code=True,
        token=token,
    )

    lora_loaded = False
    lora_error = None
    adapter_path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        try:
            dino_model = PeftModel.from_pretrained(dino_model, ADAPTER_DIR)
            lora_loaded = True
        except Exception as exc:
            lora_error = str(exc)

    dino_model.to(device).eval()

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, token=token)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, token=token)
    clip_model.to(device).eval()

    dino_dim = getattr(dino_model.config, "hidden_size", None)
    clip_dim = getattr(clip_model.config, "projection_dim", None)

    if dino_dim is None or clip_dim is None:
        raise RuntimeError("Model embedding dimensions are missing.")

    return {
        "device": device,
        "dino_processor": dino_processor,
        "dino_model": dino_model,
        "clip_processor": clip_processor,
        "clip_model": clip_model,
        "dino_dim": int(dino_dim),
        "clip_dim": int(clip_dim),
        "lora_loaded": lora_loaded,
        "lora_error": lora_error,
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@st.cache_data(show_spinner=False)
def load_db(dino_dim, clip_dim, embeddings_mtime, labels_mtime):
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("embeddings.npy not found.")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("labels.npy not found.")

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    labels = np.load(LABELS_PATH)

    expected_dim = dino_dim + clip_dim
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"embeddings.npy must be {expected_dim} dims (DINO {dino_dim} + CLIP {clip_dim})."
        )
    if embeddings.shape[0] != len(labels):
        raise ValueError("embeddings.npy and labels.npy rows do not match.")

    dino = embeddings[:, :dino_dim]
    clip = embeddings[:, dino_dim : dino_dim + clip_dim]

    dino = _normalize_rows(dino)
    clip = _normalize_rows(clip)

    weighted = np.concatenate(
        [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=1
    ).astype(np.float32)
    weighted = _normalize_rows(weighted)

    return weighted, labels


def build_or_load_index(weighted_db: np.ndarray) -> faiss.IndexFlatIP:
    dim = weighted_db.shape[1]
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        if index.d == dim and index.ntotal == weighted_db.shape[0]:
            return index
    index = faiss.IndexFlatIP(dim)
    if weighted_db.shape[0] > 0:
        index.add(weighted_db)
    faiss.write_index(index, FAISS_INDEX_PATH)
    return index


def extract_dino_embedding(image_bgr, model, processor, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    cls_tensor = safe_cls_from_output(output)
    if cls_tensor is None:
        raise RuntimeError("Could not extract CLS token from DINO output.")
    cls_tensor = F.normalize(cls_tensor, p=2, dim=-1)
    return cls_tensor[0].detach().cpu().numpy().astype(np.float32)


def extract_clip_embedding(image_bgr, model, processor, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        projected = model.visual_projection(pooled)
    projected = F.normalize(projected, p=2, dim=-1)
    return projected[0].detach().cpu().numpy().astype(np.float32)


def build_query_embedding(image_bgr, bundle):
    dino = extract_dino_embedding(
        image_bgr,
        bundle["dino_model"],
        bundle["dino_processor"],
        bundle["device"],
    )
    clip = extract_clip_embedding(
        image_bgr,
        bundle["clip_model"],
        bundle["clip_processor"],
        bundle["device"],
    )

    combined = np.concatenate(
        [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=0
    ).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def update_billing_ui():
    with billing_placeholder.container():
        st.markdown('<div class="card-title">인식된 상품 목록</div>', unsafe_allow_html=True)
        items = st.session_state.billing_items

        if not items:
            st.markdown(
                '<div class="card">아직 인식된 상품이 없습니다.</div>',
                unsafe_allow_html=True,
            )
        else:
            rows = []
            for name, qty in items.items():
                safe_name = html.escape(str(name))
                rows.append(
                    f'<div class="item-row"><span class="item-name">{safe_name}</span>'
                    f'<span class="item-qty">x{qty}</span></div>'
                )
            st.markdown(
                f'<div class="card">{"".join(rows)}</div>',
                unsafe_allow_html=True,
            )


def update_status_ui():
    with status_placeholder.container():
        last_label = st.session_state.get("last_label", "-")
        last_score = st.session_state.get("last_score", 0.0)
        last_status = st.session_state.get("last_status", "대기")
        fps = st.session_state.get("last_fps", 0.0)

        st.markdown(
            f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="metric-label">최근 인식</div>
                <div class="metric-value">{html.escape(str(last_label))}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">유사도</div>
                <div class="metric-value">{last_score:.3f}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">상태</div>
                <div class="metric-value">{html.escape(str(last_status))}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">FPS</div>
                <div class="metric-value">{fps:.1f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def init_app_state(bundle):
    if "billing_items" not in st.session_state:
        st.session_state.billing_items = {}
    if "last_seen" not in st.session_state:
        st.session_state.last_seen = {}
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
    ensure_roi_state()

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


st.markdown('<div class="dash-title">실시간 결제 대시보드</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-subtitle">DINOv3 + CLIP 앙상블 기반 상품 인식 · 가중치 0.7 / 0.3</div>',
    unsafe_allow_html=True,
)

model_bundle = load_models()

roi_status = st.session_state.get("roi_status")
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

with st.sidebar:
    st.header("모델 상태")
    if model_bundle["lora_loaded"]:
        st.success("DINO LoRA 로드 완료")
    else:
        st.warning("DINO LoRA 로드 실패 (베이스 모델 사용)")
        if model_bundle["lora_error"]:
            st.caption(model_bundle["lora_error"])

    st.markdown("---")
    st.header("ROI 설정")
    roi_setup = st.toggle("ROI 설정 모드", key="roi_setup")
    if st.button("현재 프레임 캡처"):
        st.session_state.stream_size = None
        capture_roi_frame()
        reset_roi_canvas(clear_saved=False)
    if st.session_state.get("roi_poly_norm"):
        st.success("ROI 적용됨")
        if st.button("ROI 해제"):
            reset_roi_canvas(clear_saved=True)
            st.session_state.roi_status = {"level": "info", "text": "ROI가 해제되었습니다."}
    if st.button("ROI 다시 그리기"):
        reset_roi_canvas(clear_saved=False)

try:
    init_app_state(model_bundle)
except Exception as exc:
    st.error(f"Embedding load error: {exc}")
    st.stop()

page_container = st.empty()


def run_checkout_pipeline():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

    bg_subtractor = cv2.createBackgroundSubtractorKNN(
        history=300, dist2Threshold=500, detectShadows=False
    )

    frame_count = 0

    try:
        while True:
            if st.session_state.get("roi_setup"):
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_to_stream_size(frame)

            frame_count += 1
            display_frame = frame.copy()

            now = time.perf_counter()
            delta = now - st.session_state.last_frame_time
            if delta > 0:
                st.session_state.last_fps = 1.0 / delta
            st.session_state.last_frame_time = now

            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.erode(fg_mask, None, iterations=2)
            fg_mask = cv2.dilate(fg_mask, None, iterations=4)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            roi_poly = get_roi_polygon(frame.shape)
            if roi_poly is not None:
                roi_mask = np.zeros_like(fg_mask)
                cv2.fillPoly(roi_mask, [roi_poly], 255)
                fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

            if candidates and st.session_state.faiss_index is not None \
               and st.session_state.faiss_index.ntotal > 0:
                st.session_state.last_status = "탐지됨"
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

                    inside = False
                    entry_event = False
                    if roi_poly is not None:
                        cx = x + (w / 2)
                        cy = y + (h / 2)
                        inside = cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0
                        if inside:
                            st.session_state.roi_empty_frames = 0
                            entry_event = not st.session_state.roi_occupied
                            st.session_state.roi_occupied = True
                        else:
                            st.session_state.roi_empty_frames += 1

                    if roi_poly is not None and st.session_state.roi_empty_frames >= ROI_CLEAR_FRAMES:
                        st.session_state.roi_occupied = False

                    crop = frame[y:y + h, x:x + w]

                    allow_inference = False
                    if roi_poly is not None:
                        allow_inference = entry_event
                        if entry_event:
                            st.session_state.last_status = "ROI 진입"
                        else:
                            st.session_state.last_status = "ROI 내부"
                    else:
                        allow_inference = frame_count % DETECT_EVERY_N_FRAMES == 0

                    if allow_inference:
                        emb = build_query_embedding(crop, model_bundle)
                        query = np.expand_dims(emb, axis=0)

                        D, I = st.session_state.faiss_index.search(query, 1)
                        best_idx = int(I[0][0])
                        best_score = float(D[0][0])

                        if best_score > MATCH_THRESHOLD and best_idx < len(st.session_state.labels):
                            name = str(st.session_state.labels[best_idx])
                            label = f"{name} ({best_score:.3f})"
                            st.session_state.last_label = name
                            st.session_state.last_score = best_score
                            st.session_state.last_status = "매칭됨"
                            cv2.putText(
                                display_frame,
                                label,
                                (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )

                            last_seen = st.session_state.last_seen.get(name, -COUNT_COOLDOWN_FRAMES)
                            if frame_count - last_seen >= COUNT_COOLDOWN_FRAMES:
                                st.session_state.billing_items[name] = (
                                    st.session_state.billing_items.get(name, 0) + 1
                                )
                                st.session_state.last_seen[name] = frame_count
                        else:
                            st.session_state.last_label = "미매칭"
                            st.session_state.last_score = best_score
                            st.session_state.last_status = "매칭 실패"
            else:
                if st.session_state.roi_occupied:
                    st.session_state.roi_empty_frames += 1
                    if st.session_state.roi_empty_frames >= ROI_CLEAR_FRAMES:
                        st.session_state.roi_occupied = False
                st.session_state.last_label = "-"
                st.session_state.last_score = 0.0
                st.session_state.last_status = "미탐지"

            if roi_poly is not None:
                cv2.polylines(display_frame, [roi_poly], True, (0, 181, 255), 2)

            video_placeholder.image(display_frame, channels="BGR", use_container_width=True)
            update_status_ui()
            update_billing_ui()
    finally:
        cap.release()

with page_container.container():
    if st.session_state.get("roi_setup"):
        st.markdown("### ROI 설정")
        st.button("라이브 화면으로 돌아가기", on_click=exit_roi_setup)
        roi_frame = st.session_state.get("roi_frame")
        if roi_frame is None:
            st.info("좌측 버튼으로 현재 프레임을 캡처하세요.")
        else:
            height, width = roi_frame.shape[:2]
            scale = min(
                ROI_DISPLAY_MAX_WIDTH / float(width),
                ROI_DISPLAY_MAX_HEIGHT / float(height),
                1.0,
            )
            disp_w = max(1, int(width * scale))
            disp_h = max(1, int(height * scale))
            resized = cv2.resize(roi_frame, (disp_w, disp_h))
            if st.session_state.get("stream_size") is None:
                st.session_state.stream_size = (width, height)
            canvas_key = f"roi_canvas_{st.session_state.roi_canvas_seed}"
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=2,
                stroke_color="#00B5FF",
                background_image=Image.fromarray(resized),
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="polygon",
                key=canvas_key,
            )
            objects = (canvas_result.json_data or {}).get("objects", [])
            polygon_points = None
            if objects:
                for obj in reversed(objects):
                    if obj.get("type") in ("polygon", "path", "polyline"):
                        polygon_points = _extract_polygon_points(obj)
                        if polygon_points:
                            break

            if polygon_points and len(polygon_points) >= 3:
                roi_norm = [
                    (min(max(x / disp_w, 0.0), 1.0), min(max(y / disp_h, 0.0), 1.0))
                    for x, y in polygon_points
                ]
                st.button("ROI 저장", on_click=save_roi, args=(roi_norm,))
            else:
                st.caption("클릭으로 꼭지점을 찍고, 더블클릭으로 완료하세요.")
    else:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.header("라이브 카메라")
            video_placeholder = st.empty()

        with col2:
            st.header("인식 상태")
            status_placeholder = st.empty()
            st.markdown(" ")
            st.header("상품 리스트")
            billing_placeholder = st.empty()

        run_checkout_pipeline()
