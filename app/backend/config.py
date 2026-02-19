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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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

# Inference
MIN_AREA = int(os.getenv("MIN_AREA", "2500"))
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "3"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.62"))
COUNT_COOLDOWN_SECONDS = float(os.getenv("COUNT_COOLDOWN_SECONDS", "3.0"))
ROI_CLEAR_FRAMES = int(os.getenv("ROI_CLEAR_FRAMES", "8"))

# Tracking + direction event controls
USE_DEEPSORT = _env_bool("USE_DEEPSORT", True)
TRACK_N_INIT = int(os.getenv("TRACK_N_INIT", "3"))
TRACK_MAX_AGE = int(os.getenv("TRACK_MAX_AGE", "12"))
TRACK_STALE_FRAMES = int(os.getenv("TRACK_STALE_FRAMES", "45"))

# DeepSORT runtime options
DEEPSORT_EMBEDDER_MODE = (os.getenv("DEEPSORT_EMBEDDER_MODE", "simple") or "simple").strip().lower()
DEEPSORT_MAX_IOU_DISTANCE = float(
    os.getenv("DEEPSORT_MAX_IOU_DISTANCE", os.getenv("TRACK_MAX_IOU_DISTANCE", "0.90"))
)
DEEPSORT_MAX_COSINE_DISTANCE = float(os.getenv("DEEPSORT_MAX_COSINE_DISTANCE", "0.45"))
DEEPSORT_GATING_ONLY_POSITION = _env_bool("DEEPSORT_GATING_ONLY_POSITION", True)
DEEPSORT_BBOX_PAD_RATIO = float(os.getenv("DEEPSORT_BBOX_PAD_RATIO", "0.08"))
DEEPSORT_TSU_TOLERANCE = int(os.getenv("DEEPSORT_TSU_TOLERANCE", "1"))
DEEPSORT_SIMPLE_HS_BINS = int(os.getenv("DEEPSORT_SIMPLE_HS_BINS", "16"))

# Backward compatibility alias
TRACK_MAX_IOU_DISTANCE = DEEPSORT_MAX_IOU_DISTANCE

DIRECTION_GATE_Y_NORM = float(os.getenv("DIRECTION_GATE_Y_NORM", "0.55"))
DIRECTION_MIN_DELTA_PX = float(os.getenv("DIRECTION_MIN_DELTA_PX", "18.0"))
DIRECTION_EVENT_COOLDOWN_SECONDS = float(
    os.getenv("DIRECTION_EVENT_COOLDOWN_SECONDS", str(COUNT_COOLDOWN_SECONDS))
)
DIRECTION_EVENT_BAND_PX = int(os.getenv("DIRECTION_EVENT_BAND_PX", "36"))

RECLASSIFY_EVERY_N_FRAMES = int(os.getenv("RECLASSIFY_EVERY_N_FRAMES", "6"))
RECLASSIFY_AREA_GAIN = float(os.getenv("RECLASSIFY_AREA_GAIN", "1.15"))

# Debugging
PIPELINE_DEBUG = _env_bool("PIPELINE_DEBUG", False)
PIPELINE_LOG_EVERY_N_INFER = int(os.getenv("PIPELINE_LOG_EVERY_N_INFER", "30"))

# Streaming
STREAM_TARGET_WIDTH = int(os.getenv("STREAM_TARGET_WIDTH", "960"))
STREAM_SEND_IMAGES = _env_bool("STREAM_SEND_IMAGES", False)
