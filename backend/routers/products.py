"""Product registration endpoints.

Reuses checkout_core embedding functions to add new products
to the FAISS index, mirroring pages/1_Add_Product.py logic.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Annotated

import cv2
import faiss
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from backend import config
from backend.dependencies import app_state

logger = logging.getLogger("backend.products")

router = APIRouter(tags=["products"])

DINO_WEIGHT = 0.7
CLIP_WEIGHT = 0.3


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR ndarray."""
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _build_raw_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """Build raw concatenated [DINO, CLIP] embedding for a single image."""
    from checkout_core.inference import extract_clip_embedding, extract_dino_embedding

    bundle = app_state.model_bundle
    dino_emb = extract_dino_embedding(
        image_bgr, bundle["dino_model"], bundle["dino_processor"], bundle["device"]
    )
    clip_emb = extract_clip_embedding(
        image_bgr, bundle["clip_model"], bundle["clip_processor"], bundle["device"]
    )
    return np.concatenate([dino_emb, clip_emb], axis=0)


def _build_weighted(raw: np.ndarray, dino_dim: int, clip_dim: int) -> np.ndarray:
    """Apply DINO/CLIP weighting and L2-normalize."""
    weighted = raw.copy().astype(np.float32)
    weighted[:dino_dim] *= DINO_WEIGHT
    weighted[dino_dim:] *= CLIP_WEIGHT
    norm = np.linalg.norm(weighted)
    if norm > 0:
        weighted /= norm
    return weighted


@router.post("/products")
async def add_product(
    name: Annotated[str, Form()],
    images: list[UploadFile] = File(...),
):
    """Register a new product with 1-3 images.

    Generates embeddings, appends to DB files, and updates the FAISS index.
    """
    if not name.strip():
        raise HTTPException(status_code=422, detail="Product name is required")
    if len(images) < 1 or len(images) > 3:
        raise HTTPException(status_code=422, detail="Provide 1-3 images")

    bundle = app_state.model_bundle
    dino_dim = bundle["dino_dim"]
    clip_dim = bundle["clip_dim"]

    # Generate raw embeddings for each image
    new_raw_list: list[np.ndarray] = []
    for upload in images:
        data = await upload.read()
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=422, detail=f"Invalid image: {upload.filename}")
        bgr = _pil_to_bgr(img)
        raw_emb = _build_raw_embedding(bgr)
        new_raw_list.append(raw_emb)

    new_raw = np.stack(new_raw_list, axis=0).astype(np.float32)
    new_labels = np.array([name.strip()] * len(new_raw_list), dtype=object)

    # Build weighted embeddings for new images BEFORE acquiring lock
    weighted_new = np.stack(
        [_build_weighted(r, dino_dim, clip_dim) for r in new_raw_list],
        axis=0,
    ).astype(np.float32)

    # Acquire writer lock: blocks all inference requests until update completes
    async with app_state.index_rwlock.writer_lock:
        # Load existing DB
        if os.path.exists(config.EMBEDDINGS_PATH) and os.path.exists(config.LABELS_PATH):
            old_emb = np.load(config.EMBEDDINGS_PATH, allow_pickle=False)
            old_lbl = np.load(config.LABELS_PATH, allow_pickle=True)
            updated_emb = np.vstack([old_emb, new_raw])
            updated_lbl = np.concatenate([old_lbl, new_labels])
        else:
            updated_emb = new_raw
            updated_lbl = new_labels

        # Save updated DB files
        np.save(config.EMBEDDINGS_PATH, updated_emb)
        np.save(config.LABELS_PATH, updated_lbl)

        # --- INCREMENTAL UPDATE (핵심 개선!) ---
        # Before: Rebuilt entire index O(n) - slow for large databases
        # After: Add only new vectors O(k) where k = number of new products
        if app_state.faiss_index is None or app_state.faiss_index.ntotal == 0:
            # First product registration: create new index
            dim = weighted_new.shape[1]
            app_state.faiss_index = faiss.IndexFlatIP(dim)

        # Incremental add: only adds new weighted vectors (fast!)
        app_state.faiss_index.add(weighted_new)

        # Persist updated index to disk
        faiss.write_index(app_state.faiss_index, config.FAISS_INDEX_PATH)

        # Update in-memory weighted_db by appending new vectors
        if app_state.weighted_db is None or len(app_state.weighted_db) == 0:
            app_state.weighted_db = weighted_new
        else:
            app_state.weighted_db = np.vstack([app_state.weighted_db, weighted_new])

        # Swap labels atomically
        app_state.labels = updated_lbl

    logger.info("Product '%s' added (%d images, total DB: %d)", name, len(images), len(updated_lbl))

    return {
        "status": "added",
        "product_name": name.strip(),
        "images_count": len(images),
        "total_products": len(set(updated_lbl)),
        "total_embeddings": len(updated_lbl),
    }


@router.get("/products")
async def list_products():
    """List all registered products with their embedding counts."""
    labels = app_state.labels
    if labels is None or len(labels) == 0:
        return {"products": [], "total_embeddings": 0}

    product_counts: dict[str, int] = {}
    for lbl in labels:
        name = str(lbl)
        product_counts[name] = product_counts.get(name, 0) + 1

    products = [
        {"name": name, "embedding_count": count}
        for name, count in sorted(product_counts.items())
    ]
    return {
        "products": products,
        "total_embeddings": int(len(labels)),
    }
