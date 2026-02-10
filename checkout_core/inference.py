from __future__ import annotations

import os

import cv2
import faiss
import numpy as np
import streamlit as st
import torch
from peft import PeftModel
from PIL import Image
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_PROCESSOR_ID = "facebook/dinov2-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

DINO_WEIGHT = 0.7
CLIP_WEIGHT = 0.3


def get_hf_token() -> str | None:
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
def load_models(adapter_dir: str = "data"):
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
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        try:
            dino_model = PeftModel.from_pretrained(dino_model, adapter_dir)
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
def load_db(
    dino_dim: int,
    clip_dim: int,
    embeddings_path: str,
    labels_path: str,
    embeddings_mtime: float,
    labels_mtime: float,
):
    del embeddings_mtime
    del labels_mtime

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("embeddings.npy not found.")
    if not os.path.exists(labels_path):
        raise FileNotFoundError("labels.npy not found.")

    embeddings = np.load(embeddings_path).astype(np.float32)
    labels = np.load(labels_path)

    expected_dim = dino_dim + clip_dim
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"embeddings.npy must be {expected_dim} dims (DINO {dino_dim} + CLIP {clip_dim})."
        )
    if embeddings.shape[0] != len(labels):
        raise ValueError("embeddings.npy and labels.npy rows do not match.")

    dino = embeddings[:, :dino_dim]
    clip = embeddings[:, dino_dim:dino_dim + clip_dim]

    dino = _normalize_rows(dino)
    clip = _normalize_rows(clip)

    weighted = np.concatenate(
        [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=1
    ).astype(np.float32)
    weighted = _normalize_rows(weighted)

    return weighted, labels


def build_or_load_index(weighted_db: np.ndarray, index_path: str) -> faiss.IndexFlatIP:
    dim = weighted_db.shape[1]
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        if index.d == dim and index.ntotal == weighted_db.shape[0]:
            return index

    index = faiss.IndexFlatIP(dim)
    if weighted_db.shape[0] > 0:
        index.add(weighted_db)
    faiss.write_index(index, index_path)

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

    combined = np.concatenate([dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=0).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    return combined
