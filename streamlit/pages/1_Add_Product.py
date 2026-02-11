import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from peft import PeftModel
from PIL import Image
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from ui_theme import apply_theme

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

apply_theme(page_title="상품 등록", page_icon="🧩", current_nav="🏠 홈")

st.session_state.navigation_mode = "desktop"
st.session_state.home_page_path = "pages/0_Desktop_Home.py"
st.session_state.checkout_page_path = "pages/2_Checkout.py"


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


def extract_dino_embedding(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    cls_tensor = safe_cls_from_output(output)
    if cls_tensor is None:
        raise RuntimeError("Could not extract CLS token from DINO output.")
    cls_tensor = F.normalize(cls_tensor, p=2, dim=-1)
    return cls_tensor[0].detach().cpu().numpy().astype(np.float32)


def extract_clip_embedding(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        projected = model.visual_projection(pooled)
    projected = F.normalize(projected, p=2, dim=-1)
    return projected[0].detach().cpu().numpy().astype(np.float32)


def build_raw_embedding(image, bundle):
    dino = extract_dino_embedding(
        image,
        bundle["dino_model"],
        bundle["dino_processor"],
        bundle["device"],
    )
    clip = extract_clip_embedding(
        image,
        bundle["clip_model"],
        bundle["clip_processor"],
        bundle["device"],
    )
    return np.concatenate([dino, clip], axis=0).astype(np.float32)


def build_weighted_embedding(raw_embedding, dino_dim, clip_dim):
    dino = raw_embedding[:dino_dim]
    clip = raw_embedding[dino_dim:dino_dim + clip_dim]
    combined = np.concatenate(
        [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=0
    ).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


@st.cache_data(show_spinner=False)
def load_raw_db(dino_dim, clip_dim, embeddings_mtime, labels_mtime):
    if os.path.exists(EMBEDDINGS_PATH):
        embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    else:
        embeddings = np.empty((0, dino_dim + clip_dim), dtype=np.float32)

    if os.path.exists(LABELS_PATH):
        labels = np.load(LABELS_PATH)
    else:
        labels = np.array([], dtype=str)

    if embeddings.shape[1] != dino_dim + clip_dim and embeddings.shape[0] > 0:
        raise ValueError("embeddings.npy dimension mismatch.")
    if embeddings.shape[0] != len(labels):
        raise ValueError("embeddings.npy and labels.npy rows do not match.")

    return embeddings, labels


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


def get_weighted_db(embeddings, dino_dim, clip_dim):
    if embeddings.shape[0] == 0:
        return np.empty((0, dino_dim + clip_dim), dtype=np.float32)

    dino = embeddings[:, :dino_dim]
    clip = embeddings[:, dino_dim:dino_dim + clip_dim]
    dino = _normalize_rows(dino)
    clip = _normalize_rows(clip)

    weighted = np.concatenate(
        [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=1
    ).astype(np.float32)
    weighted = _normalize_rows(weighted)
    return weighted


st.markdown(
    """
    <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px;">
      <div>
        <h1 class="page-title">관리자 상품 등록</h1>
        <p class="subtitle-text">상품명과 이미지(1~3장)를 업로드하면 임베딩 DB와 인덱스가 자동 갱신됩니다.</p>
      </div>
      <span class="pill-badge" style="background:#FFF1E7; color:#EA580C;">Admin</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("홈으로 돌아가기", key="add_back_home"):
    st.switch_page(st.session_state.get("home_page_path", "pages/0_Desktop_Home.py"))

loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown(
        """
        <div class="soft-card loading-card" style="max-width:620px; margin:0 auto 14px auto;">
          <div class="loading-spinner"></div>
          <div class="card-title" style="font-size:24px; margin-bottom:4px;">상품 등록 모델 로딩 중</div>
          <div class="card-subtitle">
            잠시만 기다려 주세요
            <span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>
          </div>
          <div class="loading-progress"></div>
          <div class="loading-caption">최초 실행 시 모델 다운로드로 인해 시간이 더 걸릴 수 있습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_bundle = load_models()
loading_placeholder.empty()

status_col1, status_col2 = st.columns(2, gap="large")
with status_col1:
    if model_bundle["lora_loaded"]:
        st.success("모델 상태: DINO LoRA 로드 완료")
    else:
        warning_message = "모델 상태: DINO 베이스 모델 사용"
        if model_bundle["lora_error"]:
            warning_message += f" · LoRA 로드 실패({model_bundle['lora_error']})"
        st.warning(warning_message)

embeddings_mtime = os.path.getmtime(EMBEDDINGS_PATH) if os.path.exists(EMBEDDINGS_PATH) else 0
labels_mtime = os.path.getmtime(LABELS_PATH) if os.path.exists(LABELS_PATH) else 0

try:
    raw_embeddings, labels = load_raw_db(
        model_bundle["dino_dim"],
        model_bundle["clip_dim"],
        embeddings_mtime,
        labels_mtime,
    )
except Exception as exc:
    st.error(f"Embedding load error: {exc}")
    st.stop()

unique_labels, counts = np.unique(labels, return_counts=True) if len(labels) else ([], [])
with status_col2:
    st.info(f"DB 상태: {len(unique_labels)}개 상품 · {len(labels)}개 이미지")

existing_df = pd.DataFrame({"상품명": unique_labels, "등록 이미지 수": counts})


def add_product(name, images):
    if not name or not images:
        st.warning("상품명과 이미지를 입력해주세요.")
        return

    if len(images) < 1 or len(images) > 3:
        st.warning("이미지는 1~3장만 업로드해주세요.")
        return

    new_raw = []
    for upload in images:
        try:
            image = Image.open(upload).convert("RGB")
            raw_emb = build_raw_embedding(image, model_bundle)
            new_raw.append(raw_emb)
        except Exception as exc:
            st.error(f"이미지 처리 중 오류: {upload.name} ({exc})")
            return

    new_raw = np.stack(new_raw, axis=0).astype(np.float32)
    new_labels = np.array([name] * new_raw.shape[0])

    if raw_embeddings.shape[0] == 0:
        updated_embeddings = new_raw
    else:
        updated_embeddings = np.vstack([raw_embeddings, new_raw])

    if labels.shape[0] == 0:
        updated_labels = new_labels
    else:
        updated_labels = np.concatenate([labels, new_labels])

    np.save(EMBEDDINGS_PATH, updated_embeddings)
    np.save(LABELS_PATH, updated_labels)

    weighted_new = np.stack(
        [
            build_weighted_embedding(raw, model_bundle["dino_dim"], model_bundle["clip_dim"])
            for raw in new_raw
        ],
        axis=0,
    ).astype(np.float32)

    weighted_db = get_weighted_db(
        updated_embeddings,
        model_bundle["dino_dim"],
        model_bundle["clip_dim"],
    )

    index = None
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)

    dim = model_bundle["dino_dim"] + model_bundle["clip_dim"]
    old_count = raw_embeddings.shape[0]
    total_count = updated_embeddings.shape[0]

    if index is None or index.d != dim:
        index = build_or_load_index(weighted_db)
    elif index.ntotal == old_count:
        if weighted_new.shape[0] > 0:
            index.add(weighted_new)
            faiss.write_index(index, FAISS_INDEX_PATH)
    elif index.ntotal != total_count:
        index = build_or_load_index(weighted_db)

    st.success(f"'{name}' 상품이 등록되었습니다!")
    st.cache_data.clear()
    st.rerun()


col_left, col_right = st.columns([1, 1.45], gap="large")

with col_left:
    st.markdown(
        """
        <div class="soft-card card-hover">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
            <div class="icon-square" style="background:linear-gradient(135deg,#FFB74D,#FF8A65);">📦</div>
            <h3 class="card-title">상품 등록</h3>
          </div>
          <p class="card-subtitle" style="margin-bottom:12px;">권장: 동일 상품 2~3장 업로드</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("add_product_form"):
        new_name = st.text_input("상품명", placeholder="예: 사과")
        new_images = st.file_uploader(
            "상품 이미지 업로드 (1~3장)",
            accept_multiple_files=True,
            type=["jpg", "png", "jpeg"],
        )
        submitted = st.form_submit_button("상품 등록", type="primary")

with col_right:
    st.markdown(
        f"""
        <div class="soft-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h3 class="card-title">등록된 상품</h3>
            <span class="pill-badge" style="background:#FFF1E7; color:#EA580C;">총 {len(unique_labels)}개</span>
          </div>
          <p class="card-subtitle" style="margin-bottom:10px;">현재까지 등록된 총 이미지 수: {len(labels)}개</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(existing_df, use_container_width=True, height=460)

if submitted:
    add_product(new_name, new_images)
