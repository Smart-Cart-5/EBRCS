"""Backend configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory (data2 = UltimateFusion model, data = original ensemble model)
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT.parent / "data2")))

# File paths
EMBEDDINGS_PATH = str(DATA_DIR / "embeddings.npy")
LABELS_PATH = str(DATA_DIR / "labels.npy")
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
ADAPTER_DIR = str(DATA_DIR)

# YOLO model (optional: if not found, falls back to background subtraction)
_DEFAULT_YOLO_PATH = str(PROJECT_ROOT.parent / "smartcart_hand_yolo11_best_arg_best.pt")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", _DEFAULT_YOLO_PATH)
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
USE_YOLO = os.getenv("USE_YOLO", "true").lower() == "true"

# Server
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# Session
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# Inference constants (mirrors checkout_core / pages/2_Checkout.py)
MIN_AREA = 2500
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "6"))
MIN_PRODUCT_BBOX_AREA = int(os.getenv("MIN_PRODUCT_BBOX_AREA", str(MIN_AREA)))
MAX_PRODUCTS_PER_FRAME = int(os.getenv("MAX_PRODUCTS_PER_FRAME", "3"))
MATCH_THRESHOLD = 0.62
COUNT_COOLDOWN_SECONDS = 3.0  # 중복 방지: 동일 상품 3초 내 재카운트 방지
ROI_CLEAR_FRAMES = 8
STREAM_TARGET_WIDTH = 960  # Restored for better quality
STREAM_SEND_IMAGES = os.getenv("STREAM_SEND_IMAGES", "false").lower() == "true"  # Send images in WebSocket responses (default: false, JSON only)
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "3"))
VOTE_WINDOW_SIZE = int(os.getenv("VOTE_WINDOW_SIZE", "5"))
VOTE_MIN_SAMPLES = int(os.getenv("VOTE_MIN_SAMPLES", "3"))
ASSOCIATION_IOU_WEIGHT = float(os.getenv("ASSOCIATION_IOU_WEIGHT", "0.5"))
ASSOCIATION_DIST_WEIGHT = float(os.getenv("ASSOCIATION_DIST_WEIGHT", "0.5"))
ASSOCIATION_MAX_CENTER_DIST = float(os.getenv("ASSOCIATION_MAX_CENTER_DIST", "0.35"))
ASSOCIATION_MIN_SCORE = float(os.getenv("ASSOCIATION_MIN_SCORE", "0.1"))
EVENT_MODE = os.getenv("EVENT_MODE", "false").lower() == "true"
T_GRASP_MIN_FRAMES = int(os.getenv("T_GRASP_MIN_FRAMES", "4"))
T_PLACE_STABLE_FRAMES = int(os.getenv("T_PLACE_STABLE_FRAMES", "12"))
SNAPSHOT_MAX_FRAMES = int(os.getenv("SNAPSHOT_MAX_FRAMES", "8"))
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
PROFILE_EVERY_N_FRAMES = int(os.getenv("PROFILE_EVERY_N_FRAMES", "30"))
