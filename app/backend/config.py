"""Backend configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# Resolve project root robustly for both:
# - local layout: <repo>/app/backend/config.py -> <repo>
# - docker layout: /app/backend/config.py -> /app
_BACKEND_DIR = Path(__file__).resolve().parent
_APP_DIR = _BACKEND_DIR.parent
_REPO_CANDIDATE = _APP_DIR.parent
PROJECT_ROOT = _REPO_CANDIDATE if (_REPO_CANDIDATE / "checkout_core").is_dir() else _APP_DIR

# Data directory
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

# File paths
EMBEDDINGS_PATH = str(DATA_DIR / "embeddings.npy")
LABELS_PATH = str(DATA_DIR / "labels.npy")
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
ADAPTER_DIR = str(DATA_DIR)

# Server
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# Session
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# Inference constants (mirrors checkout_core / pages/2_Checkout.py)
MIN_AREA = 2500
DETECT_EVERY_N_FRAMES = 3  # Smooth display: inference every 3 frames, display all frames
MATCH_THRESHOLD = 0.62
COUNT_COOLDOWN_SECONDS = 3.0  # 중복 방지: 동일 상품 3초 내 재카운트 방지
ROI_CLEAR_FRAMES = 8
STREAM_TARGET_WIDTH = 960  # Restored for better quality
STREAM_SEND_IMAGES = os.getenv("STREAM_SEND_IMAGES", "false").lower() == "true"  # Send images in WebSocket responses (default: false, JSON only)
