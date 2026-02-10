"""FastAPI application entry point.

Loads AI models once at startup via lifespan, registers routers,
and configures CORS for the React frontend.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Streamlit shim MUST be imported before checkout_core ---
import backend.st_shim  # noqa: F401

from backend import config
from backend.dependencies import app_state
from backend.routers import billing, checkout, products, sessions

logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and FAISS index once at startup."""
    from checkout_core.inference import (
        build_or_load_index,
        load_db,
        load_models,
    )

    logger.info("Loading AI models (DINOv3 + CLIP) ...")
    bundle = load_models(adapter_dir=config.ADAPTER_DIR)
    logger.info(
        "Models loaded on %s (LoRA: %s)",
        bundle["device"],
        bundle.get("lora_loaded", False),
    )

    emb_mtime = os.path.getmtime(config.EMBEDDINGS_PATH)
    lbl_mtime = os.path.getmtime(config.LABELS_PATH)

    weighted_db, labels = load_db(
        bundle["dino_dim"],
        bundle["clip_dim"],
        config.EMBEDDINGS_PATH,
        config.LABELS_PATH,
        emb_mtime,
        lbl_mtime,
    )
    logger.info("Embedding DB loaded: %d entries", len(labels))

    faiss_index = build_or_load_index(weighted_db, config.FAISS_INDEX_PATH)
    logger.info("FAISS index ready: %d vectors", faiss_index.ntotal)

    # Populate shared state
    app_state.model_bundle = bundle
    app_state.weighted_db = weighted_db
    app_state.labels = labels
    app_state.faiss_index = faiss_index
    app_state.session_manager._ttl = config.SESSION_TTL_SECONDS
    app_state.session_manager._max_sessions = config.MAX_SESSIONS

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
        return {
            "status": "ok",
            "device": str(app_state.model_bundle.get("device", "unknown")),
            "lora_loaded": app_state.model_bundle.get("lora_loaded", False),
            "index_vectors": app_state.faiss_index.ntotal if app_state.faiss_index else 0,
            "active_sessions": app_state.session_manager.active_count,
        }

    # Serve frontend static files in production (Docker)
    static_dir = Path(os.getenv("STATIC_DIR", "frontend/dist"))
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app


app = create_app()
