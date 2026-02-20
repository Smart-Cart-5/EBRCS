"""FastAPI application entry point.

Loads AI models once at startup via lifespan, registers routers,
and configures CORS for the React frontend.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import numpy as np
import torch

# Configure logging for backend and checkout_core
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logging.getLogger("backend").setLevel(logging.INFO)
logging.getLogger("checkout_core").setLevel(logging.INFO)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Streamlit shim MUST be imported before checkout_core ---
import backend.st_shim  # noqa: F401

from backend import config
from backend.dependencies import app_state
from backend.routers import billing, checkout, products, sessions

logger = logging.getLogger("backend")


def _model_param_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


def _cuda_device_name() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _find_self_test_image() -> Path | None:
    configured = str(getattr(config, "EMBEDDING_SELF_TEST_IMAGE_PATH", "") or "").strip()
    if configured:
        p = Path(configured)
        if p.is_file():
            return p
        logger.warning("Self-test image path not found: %s", configured)
    data_dir = Path(getattr(config, "DATA_DIR", "."))
    if not data_dir.exists():
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            return p
    return None


def _run_embedding_self_test(
    *,
    model_bundle: dict,
    faiss_index,
    labels: np.ndarray,
) -> None:
    if not bool(getattr(config, "EMBEDDING_SELF_TEST_ENABLE", True)):
        return
    if faiss_index is None or int(getattr(faiss_index, "ntotal", 0)) <= 0:
        logger.warning("[SELF_TEST] skipped: empty FAISS index")
        return
    image_path = _find_self_test_image()
    if image_path is None:
        logger.warning("[SELF_TEST] skipped: no image found under DATA_DIR=%s", getattr(config, "DATA_DIR", ""))
        return
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("[SELF_TEST] failed: cannot read image %s", image_path)
        return
    try:
        from checkout_core.inference import build_query_embedding

        query = np.expand_dims(build_query_embedding(image, model_bundle), axis=0).astype(np.float32)
        search_k = max(1, min(max(5, int(getattr(config, "FAISS_TOP_K", 3))), int(faiss_index.ntotal)))
        distances, indices = faiss_index.search(query, search_k)
        top1_raw = float(distances[0][0]) if distances.size > 0 else 0.0
        top1_idx = int(indices[0][0]) if indices.size > 0 else -1
        top1_label = str(labels[top1_idx]) if 0 <= top1_idx < len(labels) else "UNKNOWN"
        threshold = float(getattr(config, "EMBEDDING_SELF_TEST_MIN_RAW", 0.75))
        logger.info(
            "[SELF_TEST] image=%s top1_label=%s top1_raw=%.4f threshold=%.4f",
            str(image_path),
            top1_label,
            top1_raw,
            threshold,
        )
        if top1_raw < threshold:
            logger.warning(
                "[SELF_TEST] low similarity (%.4f < %.4f). Possible preprocessing mismatch (BGR/RGB, normalize, resize, crop).",
                top1_raw,
                threshold,
            )
    except Exception:
        logger.exception("[SELF_TEST] failed during embedding/search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and FAISS index once at startup."""
    from checkout_core.inference import (
        build_or_load_index,
        load_db,
        load_models,
    )

    logger.info("Loading AI models from %s ...", config.ADAPTER_DIR)
    bundle = load_models(
        adapter_dir=config.ADAPTER_DIR,
        adapter_model_path=config.ADAPTER_MODEL_PATH,
        use_adapter=config.USE_ADAPTER,
    )
    logger.info(
        "Embedding runtime: requested_device=%s resolved_device=%s torch.cuda.is_available=%s torch.version.cuda=%s cuda_device=%s use_amp=%s amp_dtype=%s torch_num_threads=%s",
        os.getenv("EMBEDDING_DEVICE", "auto"),
        bundle.get("device"),
        torch.cuda.is_available(),
        str(torch.version.cuda),
        _cuda_device_name(),
        bundle.get("use_amp", False),
        str(bundle.get("amp_dtype", "")),
        torch.get_num_threads(),
    )
    if "fusion_model" in bundle:
        logger.info("Fusion model device: %s", _model_param_device(bundle["fusion_model"]))
    if "dino_model" in bundle:
        logger.info("DINO model device: %s", _model_param_device(bundle["dino_model"]))
    if "clip_model" in bundle:
        logger.info("CLIP model device: %s", _model_param_device(bundle["clip_model"]))
    logger.info(
        "Models loaded on %s (mode: %s, weights: %s)",
        bundle["device"],
        bundle.get("mode", "unknown"),
        bundle.get("lora_loaded", False),
    )
    logger.info(
        "Adapter startup: enabled=%s path=%s found=%s loaded=%s applied=%s",
        bundle.get("adapter_enabled", False),
        bundle.get("adapter_path", ""),
        bundle.get("adapter_exists", False),
        bundle.get("adapter_loaded", False),
        bundle.get("adapter_applied", False),
    )
    if bundle.get("adapter_enabled", False) and not bundle.get("adapter_exists", False):
        logger.warning(
            "Adapter enabled but adapter file is missing at %s. Runtime will continue with base model.",
            bundle.get("adapter_path", ""),
        )
    if bundle.get("adapter_enabled", False) and bundle.get("adapter_exists", False) and not bundle.get("adapter_loaded", False):
        logger.warning(
            "Adapter file found but not loaded from %s. Runtime will continue with base model. error=%s",
            bundle.get("adapter_path", ""),
            bundle.get("adapter_error", ""),
        )

    emb_mtime = os.path.getmtime(config.EMBEDDINGS_PATH)
    lbl_mtime = os.path.getmtime(config.LABELS_PATH)

    weighted_db, labels, db_mode, db_dim = load_db(
        bundle["dino_dim"],
        bundle["clip_dim"],
        config.EMBEDDINGS_PATH,
        config.LABELS_PATH,
        emb_mtime,
        lbl_mtime,
    )

    bundle["db_mode"] = db_mode
    bundle["db_dim"] = db_dim
    runtime_mode = str(bundle.get("mode", "unknown"))
    mismatch_likely = (
        (runtime_mode == "fusion" and db_mode != "fusion")
        or (runtime_mode == "ensemble" and db_mode == "fusion")
    )
    if mismatch_likely:
        logger.warning(
            "Embedding runtime/DB mismatch likely: runtime_mode=%s db_mode=%s db_dim=%s. "
            "Current files may have been generated with different embedding settings. "
            "Regenerate %s and %s (and keep labels.npy aligned) for consistent search results.",
            runtime_mode,
            db_mode,
            db_dim,
            config.EMBEDDINGS_PATH,
            config.FAISS_INDEX_PATH,
        )

    logger.info("Embedding DB loaded: %d entries", len(labels))

    faiss_index = build_or_load_index(weighted_db, config.FAISS_INDEX_PATH)
    logger.info("FAISS index ready: %d vectors", faiss_index.ntotal)
    _run_embedding_self_test(model_bundle=bundle, faiss_index=faiss_index, labels=labels)

    # Load YOLO detector (optional)
    yolo_detector = None
    if config.USE_YOLO:
        if os.path.exists(config.YOLO_MODEL_PATH):
            try:
                from checkout_core.yolo_detector import YOLODetector

                device_str = str(bundle["device"])
                yolo_detector = YOLODetector(
                    model_path=config.YOLO_MODEL_PATH,
                    conf_threshold=config.YOLO_CONF_THRESHOLD,
                    device=device_str,
                )
                logger.info("YOLO detector loaded: %s", config.YOLO_MODEL_PATH)
            except Exception as e:
                logger.warning(
                    "Failed to load YOLO detector: %s (falling back to background subtraction)",
                    e,
                )
        else:
            logger.warning(
                "YOLO model not found at %s (falling back to background subtraction)",
                config.YOLO_MODEL_PATH,
            )

    # Load cart ROI segmenter (optional, Roboflow semantic segmentation)
    cart_roi_segmenter = None
    cart_roi_available = False
    cart_roi_unavailable_reason = None
    guard_enabled = bool(getattr(config, "CART_ROI_ENABLED", True))
    api_key = str(getattr(config, "ROBOFLOW_API_KEY", "")).strip()
    if not guard_enabled:
        cart_roi_unavailable_reason = "server_disabled"
        logger.warning("Cart ROI external API calls disabled by CART_ROI_ENABLED=0")
    elif not api_key:
        cart_roi_unavailable_reason = "missing_api_key"
        logger.warning("ROBOFLOW_API_KEY missing; auto ROI will fallback to no_roi")
    else:
        try:
            from backend.roi import RoboflowCartSegmenter

            cart_roi_segmenter = RoboflowCartSegmenter(
                api_key=config.ROBOFLOW_API_KEY,
                endpoint="smartcart-fd4z1",
                version=4,
                every_n_frames=config.CART_ROI_EVERY_N_FRAMES,
                debug=config.CART_ROI_DEBUG,
                debug_dir=config.CART_ROI_DEBUG_DIR,
                target_class_name=config.CART_ROI_CLASS_NAME,
                class_aliases=config.CART_ROI_CLASS_ALIASES,
            )
            cart_roi_available = True
            logger.info(
                "Cart ROI segmenter available: endpoint=%s version=%s every_n_frames=%d debug=%s target_class=%s",
                "smartcart-fd4z1",
                4,
                int(config.CART_ROI_EVERY_N_FRAMES),
                bool(config.CART_ROI_DEBUG),
                str(config.CART_ROI_CLASS_NAME),
            )
        except Exception as exc:
            cart_roi_unavailable_reason = f"init_error:{type(exc).__name__}"
            logger.warning("Failed to initialize cart ROI segmenter; auto ROI will fallback: %s", exc)

    # Populate shared state
    app_state.model_bundle = bundle
    app_state.weighted_db = weighted_db
    app_state.labels = labels
    app_state.faiss_index = faiss_index
    app_state.yolo_detector = yolo_detector
    app_state.cart_roi_segmenter = cart_roi_segmenter
    app_state.cart_roi_available = bool(cart_roi_available)
    app_state.cart_roi_unavailable_reason = cart_roi_unavailable_reason
    app_state.session_manager._ttl = config.SESSION_TTL_SECONDS
    app_state.session_manager._max_sessions = config.MAX_SESSIONS
    logger.info(
        "Runtime perf knobs: OCR_ENABLED=%s OCR_COOLDOWN_SEC=%.1f SEARCH_COOLDOWN_SEC=%.2f",
        bool(getattr(config, "OCR_ENABLED", False)),
        float(getattr(config, "OCR_COOLDOWN_SEC", 10.0)),
        float(getattr(config, "SEARCH_COOLDOWN_SEC", float(getattr(config, "SEARCH_COOLDOWN_MS", 1000)) / 1000.0)),
    )

    yield

    logger.info("Shutting down ...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EBRCS Checkout API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sessions.router, prefix="/api")
    app.include_router(billing.router, prefix="/api")
    app.include_router(products.router, prefix="/api")
    app.include_router(checkout.router, prefix="/api")

    @app.get("/api/health")
    async def health():
        active_phase = None
        active_confirmed = False
        try:
            sessions = list(getattr(app_state.session_manager, "_sessions", {}).values())
            if sessions:
                s0 = sessions[0]
                active_phase = str(s0.state.get("phase", ""))
                active_confirmed = bool(s0.state.get("_cart_roi_confirmed", False))
        except Exception:
            active_phase = None
            active_confirmed = False
        return {
            "status": "ok",
            "device": str(app_state.model_bundle.get("device", "unknown")),
            "lora_loaded": app_state.model_bundle.get("lora_loaded", False),
            "index_vectors": (
                app_state.faiss_index.ntotal if app_state.faiss_index else 0
            ),
            "active_sessions": app_state.session_manager.active_count,
            "yolo_loaded": app_state.yolo_detector is not None,
            "cart_roi_enabled": app_state.cart_roi_segmenter is not None,
            "cart_roi_available": bool(app_state.cart_roi_available),
            "cart_roi_unavailable_reason": app_state.cart_roi_unavailable_reason,
            "phase": active_phase,
            "cart_roi_confirmed": active_confirmed,
        }

    # Serve frontend static files in production (Docker)
    static_dir = Path(os.getenv("STATIC_DIR", "frontend/dist"))
    if static_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=str(static_dir), html=True), name="frontend"
        )

    return app


app = create_app()
