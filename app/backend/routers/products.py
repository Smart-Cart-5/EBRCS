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
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import config
from backend.database import get_db
from backend.dependencies import app_state

logger = logging.getLogger("backend.products")

router = APIRouter(tags=["products"])

DINO_WEIGHT = 0.7
CLIP_WEIGHT = 0.3
MIN_IMAGES = 3
MAX_IMAGES = 5


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


def _recompute_weighted_db(embeddings: np.ndarray, dino_dim: int, clip_dim: int) -> np.ndarray:
    if embeddings.size == 0:
        return np.empty((0, dino_dim + clip_dim), dtype=np.float32)

    dino = embeddings[:, :dino_dim].astype(np.float32)
    clip = embeddings[:, dino_dim : dino_dim + clip_dim].astype(np.float32)

    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    dino = _normalize_rows(dino)
    clip = _normalize_rows(clip)
    weighted = np.concatenate([dino * DINO_WEIGHT, clip * CLIP_WEIGHT], axis=1).astype(
        np.float32
    )
    weighted = _normalize_rows(weighted)
    return weighted


@router.post("/products")
async def add_product(
    item_no: Annotated[str, Form()],
    name: Annotated[str, Form()],
    price: Annotated[int, Form()],
    barcd: Annotated[str | None, Form()] = None,
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Register a new product with 3-5 images.

    Generates embeddings, appends to DB files, and updates the FAISS index.
    """
    if not item_no.strip():
        raise HTTPException(status_code=422, detail="Item number is required")
    if not name.strip():
        raise HTTPException(status_code=422, detail="Product name is required")
    if price <= 0:
        raise HTTPException(status_code=422, detail="Price must be positive")
    if len(images) < MIN_IMAGES or len(images) > MAX_IMAGES:
        raise HTTPException(
            status_code=422, detail=f"Provide {MIN_IMAGES}-{MAX_IMAGES} images"
        )

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
    clean_name = name.strip()
    clean_item_no = item_no.strip()
    clean_barcd = barcd.strip() if barcd and barcd.strip() else None
    label = f"{clean_item_no}_{clean_name}"
    new_labels = np.array([label] * len(new_raw_list), dtype=object)

    # Build weighted embeddings for new images BEFORE acquiring lock
    weighted_new = np.stack(
        [_build_weighted(r, dino_dim, clip_dim) for r in new_raw_list],
        axis=0,
    ).astype(np.float32)

    product_id: int | None = None
    try:
        existing = db.execute(
            text(
                """
                SELECT id FROM products
                WHERE item_no = :item_no
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"item_no": clean_item_no},
        ).mappings().first()
        if existing:
            product_id = int(existing["id"])
            db.execute(
                text(
                    """
                    UPDATE products
                    SET product_name = :name, barcd = :barcd, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id
                    """
                ),
                {"name": clean_name, "barcd": clean_barcd, "id": product_id},
            )
        else:
            result = db.execute(
                text(
                    """
                    INSERT INTO products (item_no, barcd, product_name, created_at, updated_at)
                    VALUES (:item_no, :barcd, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """
                ),
                {"item_no": clean_item_no, "barcd": clean_barcd, "name": clean_name},
            )
            product_id = int(result.lastrowid or 0)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc

    if not product_id:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create product record")

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

    try:
        db.execute(
            text(
                """
                INSERT INTO product_prices (product_id, price, currency, source, checked_at, created_at)
                VALUES (:product_id, :price, 'KRW', 'admin_upload', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            ),
            {"product_id": product_id, "price": int(price)},
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc

    logger.info(
        "Product '%s' (%s) added (%d images, total DB: %d)",
        clean_name,
        clean_item_no,
        len(images),
        len(updated_lbl),
    )

    return {
        "status": "added",
        "item_no": clean_item_no,
        "product_name": clean_name,
        "label": label,
        "price": int(price),
        "images_count": len(images),
        "total_products": len(set(updated_lbl)),
        "total_embeddings": len(updated_lbl),
    }


@router.get("/products")
async def list_products(db: Session = Depends(get_db)):
    """List all registered products with their embedding counts."""
    labels = app_state.labels
    embedding_total = int(len(labels)) if labels is not None else 0
    label_counts: dict[str, int] = {}
    embed_products: dict[str, dict[str, object]] = {}
    if labels is not None:
        for lbl in labels:
            label_str = str(lbl)
            label_counts[label_str] = label_counts.get(label_str, 0) + 1

        for label_str, count in label_counts.items():
            item_no = None
            name = label_str
            if "_" in label_str:
                prefix, suffix = label_str.split("_", 1)
                if prefix.isdigit():
                    item_no = prefix
                    name = suffix or label_str

            key = item_no if item_no else f"__name__:{name}"
            if key in embed_products:
                embed_products[key]["embedding_count"] = int(
                    embed_products[key]["embedding_count"]
                ) + int(count)
            else:
                embed_products[key] = {
                    "item_no": item_no,
                    "name": name,
                    "embedding_count": int(count),
                }

    try:
        rows = db.execute(
            text(
                """
                SELECT id, item_no, barcd, product_name
                FROM products
                ORDER BY id DESC
                """
            )
        ).mappings().all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc

    products = []
    seen_item_nos: set[str] = set()
    for row in rows:
        item_no = str(row["item_no"])
        if item_no in seen_item_nos:
            continue
        seen_item_nos.add(item_no)

        name = str(row["product_name"])
        embedding_key = item_no
        embedding_count = 0
        if embedding_key in embed_products:
            embedding_count = int(embed_products[embedding_key]["embedding_count"])
            embed_products.pop(embedding_key, None)
        else:
            label = f"{item_no}_{name}"
            embedding_count = label_counts.get(label, 0)
            if embedding_count == 0:
                embedding_count = label_counts.get(name, 0)

        price_row = db.execute(
            text(
                """
                SELECT price
                FROM product_prices
                WHERE product_id = :pid
                ORDER BY checked_at DESC, id DESC
                LIMIT 1
                """
            ),
            {"pid": row["id"]},
        ).mappings().first()
        price = int(price_row["price"]) if price_row else None

        products.append(
            {
                "id": int(row["id"]),
                "item_no": item_no,
                "name": name,
                "price": price,
                "barcd": row.get("barcd"),
                "label": f"{item_no}_{name}",
                "embedding_count": int(embedding_count),
            }
        )

    # Add embedding-only products not present in DB
    for key in sorted(embed_products.keys()):
        entry = embed_products[key]
        item_no = entry.get("item_no")
        name = entry.get("name")
        products.append(
            {
                "id": None,
                "item_no": item_no or "",
                "name": str(name),
                "price": None,
                "barcd": None,
                "label": f"{item_no}_{name}" if item_no else str(name),
                "embedding_count": int(entry.get("embedding_count") or 0),
            }
        )

    return {
        "products": products,
        "total_embeddings": embedding_total,
    }


@router.delete("/products/{item_no}")
async def delete_product(item_no: str, db: Session = Depends(get_db)):
    """Delete a product from catalog and embeddings by item_no."""
    clean_item_no = (item_no or "").strip()
    if not clean_item_no:
        raise HTTPException(status_code=422, detail="Item number is required")

    try:
        rows = db.execute(
            text(
                """
                SELECT id, product_name
                FROM products
                WHERE item_no = :item_no
                ORDER BY id DESC
                """
            ),
            {"item_no": clean_item_no},
        ).mappings().all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc

    if not rows:
        raise HTTPException(status_code=404, detail="Product not found")

    product_ids = [int(r["id"]) for r in rows]
    product_names = {str(r["product_name"]) for r in rows if r.get("product_name")}

    # Update embeddings + FAISS index
    removed_embeddings = 0
    async with app_state.index_rwlock.writer_lock:
        if os.path.exists(config.EMBEDDINGS_PATH) and os.path.exists(config.LABELS_PATH):
            embeddings = np.load(config.EMBEDDINGS_PATH).astype(np.float32)
            labels = np.load(config.LABELS_PATH, allow_pickle=True)

            if embeddings.shape[0] != len(labels):
                raise HTTPException(
                    status_code=500, detail="embeddings.npy and labels.npy mismatch"
                )

            keep_mask = []
            for lbl in labels:
                label_str = str(lbl)
                should_remove = label_str.startswith(f"{clean_item_no}_")
                if not should_remove and label_str in product_names:
                    should_remove = True
                keep_mask.append(not should_remove)

            keep_mask = np.array(keep_mask, dtype=bool)
            removed_embeddings = int((~keep_mask).sum())

            updated_emb = embeddings[keep_mask]
            updated_lbl = labels[keep_mask]

            np.save(config.EMBEDDINGS_PATH, updated_emb)
            np.save(config.LABELS_PATH, updated_lbl)

            bundle = app_state.model_bundle
            weighted_db = _recompute_weighted_db(
                updated_emb, bundle["dino_dim"], bundle["clip_dim"]
            )
            app_state.weighted_db = weighted_db
            app_state.labels = updated_lbl

            from checkout_core.inference import build_or_load_index

            app_state.faiss_index = build_or_load_index(weighted_db, config.FAISS_INDEX_PATH)

    try:
        for pid in product_ids:
            db.execute(
                text("DELETE FROM product_prices WHERE product_id = :pid"),
                {"pid": pid},
            )
        db.execute(
            text("DELETE FROM products WHERE item_no = :item_no"),
            {"item_no": clean_item_no},
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {exc}") from exc

    return {
        "status": "deleted",
        "item_no": clean_item_no,
        "removed_embeddings": removed_embeddings,
        "removed_products": len(product_ids),
    }
