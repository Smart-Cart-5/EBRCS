from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
import time

import cv2
import faiss
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
)

logger = logging.getLogger(__name__)

# --- Ensemble mode constants (original data/) ---
DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_PROCESSOR_ID = "facebook/dinov2-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

DINO_WEIGHT = 0.7
CLIP_WEIGHT = 0.3


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device() -> torch.device:
    requested = (os.getenv("EMBEDDING_DEVICE") or "").strip().lower()
    cuda_ok = torch.cuda.is_available()
    if requested in {"cuda", "gpu"}:
        if cuda_ok:
            return torch.device("cuda")
        logger.warning("EMBEDDING_DEVICE=cuda requested but CUDA unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_ok else "cpu")


def _amp_context(bundle: dict):
    if not bundle.get("use_amp", False):
        return nullcontext()
    if str(bundle.get("device", "cpu")) != "cuda":
        return nullcontext()
    amp_dtype = bundle.get("amp_dtype", torch.float16)
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _model_param_device(model) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _debug_device_check(bundle: dict, *, tag: str, model, input_tensor: torch.Tensor | None) -> None:
    if not _env_flag("EMBEDDING_DEBUG_DEVICE_CHECK", False):
        return
    interval_ms = int(os.getenv("EMBEDDING_DEBUG_DEVICE_CHECK_INTERVAL_MS", "5000"))
    now_ms = int(time.time() * 1000)
    last_ms = int(bundle.get("_device_check_last_log_ms", 0))
    if now_ms - last_ms < max(0, interval_ms):
        return
    bundle["_device_check_last_log_ms"] = now_ms

    resolved = bundle.get("device")
    model_dev = _model_param_device(model)
    input_dev = input_tensor.device if torch.is_tensor(input_tensor) else None
    match = model_dev is not None and input_dev is not None and str(model_dev) == str(input_dev)
    logger.info(
        "[EMBED_DEVICE] tag=%s resolved=%s model_device=%s input_device=%s match=%s",
        tag,
        resolved,
        str(model_dev),
        str(input_dev),
        match,
    )


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


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def _detect_data_mode(adapter_dir: str) -> str:
    """Detect data mode from adapter_config.json.

    Returns:
        "fusion" - UltimateFusionModel (DINOv3-Base timm + CLIP + bottleneck)
        "ensemble" - Original DINOv3-Large + LoRA + CLIP weighted ensemble
    """
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if "num_products" in cfg and "num_majors" in cfg:
            return "fusion"
    return "ensemble"


# ---------------------------------------------------------------------------
# UltimateFusionModel (matches training code exactly)
# ---------------------------------------------------------------------------


class UltimateFusionModel(nn.Module):
    """DINOv3-Base (timm) + CLIP-Base/32 -> Bottleneck -> 512d.

    Architecture matches training code exactly:
    - DINO: vit_base_patch16_dinov3.lvd1689m (timm, 768d)
    - CLIP: CLIPVisionModel openai/clip-vit-base-patch32 (768d pooler)
    - clip_resizer: Upsample to 224x224
    - Bottleneck: Linear(1536,512) + BN(512) + ReLU + Dropout(0.2)
    - product_head, major_head (not used for embedding inference)
    """

    def __init__(self, num_products: int, num_majors: int):
        super().__init__()
        self.dino = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=False,
            num_classes=0,
        )
        self.clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_resizer = nn.Upsample(
            size=(224, 224), mode="bilinear", align_corners=False
        )

        feature_dim = 768 + 768  # DINO CLS + CLIP pooler
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.product_head = nn.Linear(512, num_products)
        self.major_head = nn.Linear(512, num_majors)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-d L2-normalized embedding for FAISS search."""
        with torch.inference_mode():
            dino_feat = self.dino(x)  # [B, 768]
            clip_feat = self.clip(self.clip_resizer(x)).pooler_output  # [B, 768]
        fusion_feat = torch.cat([dino_feat, clip_feat], dim=-1)  # [B, 1536]
        embedding = self.bottleneck(fusion_feat)  # [B, 512]
        return F.normalize(embedding, p=2, dim=1)


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_models(adapter_dir: str = "data"):
    device = _resolve_device()
    token = get_hf_token()
    use_amp = _env_flag("EMBEDDING_USE_AMP", True) and device.type == "cuda"
    amp_dtype_name = (os.getenv("EMBEDDING_AMP_DTYPE") or "float16").strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    if device.type == "cpu":
        cpu_threads = int(os.getenv("EMBEDDING_CPU_THREADS", "0"))
        if cpu_threads > 0:
            try:
                torch.set_num_threads(cpu_threads)
                logger.info("Set torch CPU threads=%d", cpu_threads)
            except Exception as exc:
                logger.warning("Failed to set torch CPU threads: %s", exc)

    mode = _detect_data_mode(adapter_dir)

    if mode == "fusion":
        logger.info("Loading UltimateFusionModel (DINOv3-Base timm + CLIP-Base)")

        # Read metadata
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        num_products = cfg["num_products"]
        num_majors = cfg["num_majors"]
        input_size = cfg.get("input_size", 512)
        norm_mean = cfg.get("norm_mean", [0.485, 0.456, 0.406])
        norm_std = cfg.get("norm_std", [0.229, 0.224, 0.225])

        # Build model (pretrained=False, we load custom weights)
        fusion_model = UltimateFusionModel(num_products, num_majors)

        # Load trained weights from safetensors
        adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        weights_loaded = False
        weights_error = None

        if os.path.exists(adapter_path):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(adapter_path)

                # Direct load: safetensors keys match model keys exactly
                # (dino.*, clip.*, bottleneck.*, product_head.*, major_head.*)
                result = fusion_model.load_state_dict(state_dict, strict=False)
                weights_loaded = True

                missing = len(result.missing_keys)
                unexpected = len(result.unexpected_keys)
                loaded = len(state_dict) - unexpected
                logger.info(
                    "Loaded weights: %d tensors, missing=%d, unexpected=%d",
                    loaded,
                    missing,
                    unexpected,
                )
                if result.missing_keys:
                    logger.warning("Missing keys: %s", result.missing_keys[:5])
                if result.unexpected_keys:
                    logger.warning("Unexpected keys: %s", result.unexpected_keys[:5])
            except Exception as exc:
                weights_error = str(exc)
                logger.warning("Failed to load fusion weights: %s", exc)

        fusion_model.to(device).eval()

        # Build transform matching training code
        fusion_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

        return {
            "device": device,
            "mode": "fusion",
            "fusion_model": fusion_model,
            "fusion_transform": fusion_transform,
            "use_amp": use_amp,
            "amp_dtype": amp_dtype,
            "dino_dim": 0,  # Signal to load_db: embeddings are 512d, not split
            "clip_dim": 512,
            "lora_loaded": weights_loaded,
            "lora_error": weights_error,
            "product_map": cfg.get("product_id_to_name", {}),
        }

    else:
        # Original ensemble mode: DINOv3-Large + LoRA + CLIP
        logger.info("Loading ensemble model (DINOv3-Large + LoRA + CLIP)")
        from peft import PeftModel

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
            "mode": "ensemble",
            "dino_processor": dino_processor,
            "dino_model": dino_model,
            "clip_processor": clip_processor,
            "clip_model": clip_model,
            "use_amp": use_amp,
            "amp_dtype": amp_dtype,
            "dino_dim": int(dino_dim),
            "clip_dim": int(clip_dim),
            "lora_loaded": lora_loaded,
            "lora_error": lora_error,
        }


# ---------------------------------------------------------------------------
# load_db
# ---------------------------------------------------------------------------


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


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

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings.npy must be 2D array (got shape {embeddings.shape}).")
    if embeddings.shape[0] != len(labels):
        raise ValueError("embeddings.npy and labels.npy rows do not match.")

    actual_dim = int(embeddings.shape[1])

    # ----------------------------
    # Fusion mode: embeddings are 512-d (bottleneck output)
    # ----------------------------
    if dino_dim == 0:
        if actual_dim != int(clip_dim):
            raise ValueError(
                f"[fusion] embeddings.npy must be {clip_dim} dims (got {actual_dim})."
            )
        db_mode = "fusion"
        weighted = _normalize_rows(embeddings)

    # ----------------------------
    # Ensemble pipeline:
    # allow 1536 (DINO+CLIP), 1024 (DINO-only), 512 (CLIP-only)
    # ----------------------------
    else:
        expected_dim = int(dino_dim + clip_dim)

        if actual_dim == expected_dim:
            # DINO+CLIP concat DB
            db_mode = "ensemble"
            dino = embeddings[:, :dino_dim]
            clip = embeddings[:, dino_dim : dino_dim + clip_dim]
            dino = _normalize_rows(dino)
            clip = _normalize_rows(clip)
            weighted = np.concatenate(
                [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=1
            ).astype(np.float32)
            weighted = _normalize_rows(weighted)

        elif actual_dim == int(dino_dim):
            # DINO-only legacy DB
            db_mode = "dino"
            weighted = _normalize_rows(embeddings)

        elif actual_dim == int(clip_dim):
            # CLIP-only legacy DB  <-- your current embeddings.npy (309,512)
            db_mode = "clip"
            weighted = _normalize_rows(embeddings)

        else:
            raise ValueError(
                f"embeddings.npy dims mismatch. "
                f"Expected {expected_dim} (DINO {dino_dim} + CLIP {clip_dim}) "
                f"or {dino_dim} (DINO-only) or {clip_dim} (CLIP-only), got {actual_dim}."
            )

    logger.info("DB loaded: mode=%s dim=%d rows=%d", db_mode, actual_dim, embeddings.shape[0])
    db_norms = np.linalg.norm(weighted, axis=1)
    logger.info(
        "DB norm stats after normalize: mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(np.mean(db_norms)),
        float(np.std(db_norms)),
        float(np.min(db_norms)),
        float(np.max(db_norms)),
    )
    return weighted, labels, db_mode, actual_dim


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------


def build_or_load_index(weighted_db: np.ndarray, index_path: str) -> faiss.IndexFlatIP:
    dim = weighted_db.shape[1]
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        metric_ok = getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT
        if index.d == dim and index.ntotal == weighted_db.shape[0] and metric_ok:
            return index
        logger.warning(
            "Rebuilding FAISS index due to mismatch: d=%s ntotal=%s metric_type=%s (expect d=%d ntotal=%d metric=IP)",
            getattr(index, "d", None),
            getattr(index, "ntotal", None),
            getattr(index, "metric_type", None),
            dim,
            weighted_db.shape[0],
        )

    index = faiss.IndexFlatIP(dim)
    if weighted_db.shape[0] > 0:
        index.add(weighted_db)
    faiss.write_index(index, index_path)

    return index


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_dino_embedding(image_bgr, model, processor, device, bundle: dict | None = None):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    amp_ctx = nullcontext()
    if str(device) == "cuda" and _env_flag("EMBEDDING_USE_AMP", True):
        amp_dtype_name = (os.getenv("EMBEDDING_AMP_DTYPE") or "float16").strip().lower()
        amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    if bundle is not None:
        first_input = next(iter(inputs.values())) if inputs else None
        _debug_device_check(bundle, tag="dino", model=model, input_tensor=first_input)
    with torch.inference_mode():
        with amp_ctx:
            output = model(**inputs)

    cls_tensor = safe_cls_from_output(output)
    if cls_tensor is None:
        raise RuntimeError("Could not extract CLS token from DINO output.")
    cls_tensor = F.normalize(cls_tensor, p=2, dim=-1)

    return cls_tensor[0].detach().cpu().numpy().astype(np.float32)


def extract_clip_embedding(image_bgr, model, processor, device, bundle: dict | None = None):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    amp_ctx = nullcontext()
    if str(device) == "cuda" and _env_flag("EMBEDDING_USE_AMP", True):
        amp_dtype_name = (os.getenv("EMBEDDING_AMP_DTYPE") or "float16").strip().lower()
        amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    if bundle is not None:
        _debug_device_check(bundle, tag="clip", model=model, input_tensor=pixel_values)
    with torch.inference_mode():
        with amp_ctx:
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            projected = model.visual_projection(pooled)

    projected = F.normalize(projected, p=2, dim=-1)
    return projected[0].detach().cpu().numpy().astype(np.float32)


def build_query_embedding(image_bgr, bundle):
    mode = bundle.get("mode", "ensemble")

    if mode == "fusion":
        # UltimateFusionModel: single forward pass -> 512d
        device = bundle["device"]
        fusion_model = bundle["fusion_model"]
        fusion_transform = bundle["fusion_transform"]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        pixel_values = fusion_transform(pil_image).unsqueeze(0).to(device)

        _debug_device_check(bundle, tag="fusion", model=fusion_model, input_tensor=pixel_values)
        with torch.inference_mode():
            with _amp_context(bundle):
                embedding = fusion_model.forward_embedding(pixel_values)

        return _normalize_vector(embedding[0].cpu().numpy().astype(np.float32))

    else:
        # Ensemble: DINO + CLIP weighted combination
        dino = extract_dino_embedding(
            image_bgr,
            bundle["dino_model"],
            bundle["dino_processor"],
            bundle["device"],
            bundle=bundle,
        )
        clip = extract_clip_embedding(
            image_bgr,
            bundle["clip_model"],
            bundle["clip_processor"],
            bundle["device"],
            bundle=bundle,
        )

        combined = np.concatenate(
            [dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=0
        ).astype(np.float32)
        return _normalize_vector(combined)
