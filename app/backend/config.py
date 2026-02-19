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
SEARCH_EVERY_N_FRAMES = int(os.getenv("SEARCH_EVERY_N_FRAMES", "10"))
MIN_BOX_AREA_RATIO = float(os.getenv("MIN_BOX_AREA_RATIO", "0.05"))
STABLE_FRAMES_FOR_SEARCH = int(os.getenv("STABLE_FRAMES_FOR_SEARCH", "5"))
SEARCH_COOLDOWN_MS = int(os.getenv("SEARCH_COOLDOWN_MS", "600"))
SEARCH_COOLDOWN_MS_UNSTABLE = int(os.getenv("SEARCH_COOLDOWN_MS_UNSTABLE", "250"))
SEARCH_COOLDOWN_MS_PRECONFIRM = int(os.getenv("SEARCH_COOLDOWN_MS_PRECONFIRM", "50"))
SEARCH_POST_CONFIRM_WINDOW_MS = int(os.getenv("SEARCH_POST_CONFIRM_WINDOW_MS", "2500"))
ROI_BOX_MIN_OVERLAP = float(os.getenv("ROI_BOX_MIN_OVERLAP", "0.2"))
PRODUCT_CONF_MIN = float(os.getenv("PRODUCT_CONF_MIN", "0.65"))
PRODUCT_MIN_AREA_RATIO = float(os.getenv("PRODUCT_MIN_AREA_RATIO", "0.06"))
PRODUCT_ASPECT_RATIO_MIN = float(os.getenv("PRODUCT_ASPECT_RATIO_MIN", "0.35"))
PRODUCT_ASPECT_RATIO_MAX = float(os.getenv("PRODUCT_ASPECT_RATIO_MAX", "3.0"))
PRODUCT_MAX_HEIGHT_RATIO = float(os.getenv("PRODUCT_MAX_HEIGHT_RATIO", "0.95"))
PRODUCT_MAX_WIDTH_RATIO = float(os.getenv("PRODUCT_MAX_WIDTH_RATIO", "0.95"))
PRODUCT_EDGE_TOUCH_EPS = float(os.getenv("PRODUCT_EDGE_TOUCH_EPS", "0.01"))
MIN_SEARCH_AREA_RATIO = float(os.getenv("MIN_SEARCH_AREA_RATIO", "0.12"))
MIN_CROP_SIZE = int(os.getenv("MIN_CROP_SIZE", "224"))
SEARCH_CROP_MIN_SIDE = int(os.getenv("SEARCH_CROP_MIN_SIDE", "320"))
SEARCH_CROP_PAD_RATIO = float(os.getenv("SEARCH_CROP_PAD_RATIO", "0.20"))
SEARCH_CROP_MIN_SIDE_AFTER_EXPAND = int(os.getenv("SEARCH_CROP_MIN_SIDE_AFTER_EXPAND", "240"))
SEARCH_CROP_ASPECT_MIN = float(os.getenv("SEARCH_CROP_ASPECT_MIN", "0.5"))
SEARCH_CROP_ASPECT_MAX = float(os.getenv("SEARCH_CROP_ASPECT_MAX", "2.0"))
SEARCH_CROP_EDGE_PAD_RATIO = float(os.getenv("SEARCH_CROP_EDGE_PAD_RATIO", "0.02"))
SEARCH_SELECT_W_CONF = float(os.getenv("SEARCH_SELECT_W_CONF", "0.4"))
SEARCH_SELECT_W_AREA = float(os.getenv("SEARCH_SELECT_W_AREA", "0.4"))
SEARCH_SELECT_W_ROI = float(os.getenv("SEARCH_SELECT_W_ROI", "0.2"))
ROI_IOU_MIN = float(os.getenv("ROI_IOU_MIN", "0.15"))
ROI_CENTER_PASS = os.getenv("ROI_CENTER_PASS", "true").lower() == "true"
HAND_CONF_MIN = float(os.getenv("HAND_CONF_MIN", "0.5"))
HAND_OVERLAP_IOU = float(os.getenv("HAND_OVERLAP_IOU", "0.25"))
SEARCH_MASK_HAND = os.getenv("SEARCH_MASK_HAND", "false").lower() == "true"
MIN_PRODUCT_BBOX_AREA = int(os.getenv("MIN_PRODUCT_BBOX_AREA", str(MIN_AREA)))
MAX_PRODUCTS_PER_FRAME = int(os.getenv("MAX_PRODUCTS_PER_FRAME", "3"))
MATCH_THRESHOLD = 0.62
COUNT_COOLDOWN_SECONDS = 3.0  # 중복 방지: 동일 상품 3초 내 재카운트 방지
ROI_CLEAR_FRAMES = 8
STREAM_TARGET_WIDTH = 960  # Restored for better quality
STREAM_SEND_IMAGES = os.getenv("STREAM_SEND_IMAGES", "false").lower() == "true"  # Send images in WebSocket responses (default: false, JSON only)
CHECKOUT_QUEUE_MAXSIZE = int(os.getenv("CHECKOUT_QUEUE_MAXSIZE", "1"))
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
T_REMOVE_CONFIRM_FRAMES = int(os.getenv("T_REMOVE_CONFIRM_FRAMES", "45"))
ROI_HYSTERESIS_INSET_RATIO = float(os.getenv("ROI_HYSTERESIS_INSET_RATIO", "0.05"))
ROI_HYSTERESIS_OUTSET_RATIO = float(os.getenv("ROI_HYSTERESIS_OUTSET_RATIO", "0.05"))
WARP_MODE = os.getenv("WARP_MODE", "false").lower() == "true"
WARP_WIDTH = int(os.getenv("WARP_WIDTH", "640"))
WARP_HEIGHT = int(os.getenv("WARP_HEIGHT", "480"))
WARP_BLACK_MAX_THRESHOLD = int(os.getenv("WARP_BLACK_MAX_THRESHOLD", "5"))
WARP_BLACK_MEAN_THRESHOLD = float(os.getenv("WARP_BLACK_MEAN_THRESHOLD", "2.0"))
WARP_DEBUG_LOG = os.getenv("WARP_DEBUG_LOG", "true").lower() == "true"
WARP_DEBUG_LOG_INTERVAL_MS = int(os.getenv("WARP_DEBUG_LOG_INTERVAL_MS", "1000"))
WARP_DEBUG_SAVE = os.getenv("WARP_DEBUG_SAVE", "false").lower() == "true"
WARP_DEBUG_DIR = os.getenv("WARP_DEBUG_DIR", str(PROJECT_ROOT / "warp_debug"))
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
PROFILE_EVERY_N_FRAMES = int(os.getenv("PROFILE_EVERY_N_FRAMES", "30"))
SEARCH_DEBUG_LOG = os.getenv("SEARCH_DEBUG_LOG", "false").lower() == "true"
SEARCH_DEBUG_SAVE_CROP = os.getenv("SEARCH_DEBUG_SAVE_CROP", "false").lower() == "true"
SEARCH_DEBUG_CROP_DIR = os.getenv("SEARCH_DEBUG_CROP_DIR", str(PROJECT_ROOT / "debug_crops"))
SEARCH_DEBUG_LOG_INTERVAL_MS = int(os.getenv("SEARCH_DEBUG_LOG_INTERVAL_MS", "5000"))
SEARCH_DEBUG_SAVE_INTERVAL_MS = int(os.getenv("SEARCH_DEBUG_SAVE_INTERVAL_MS", "1000"))
SEARCH_DEBUG_DB_NORM_SAMPLE = int(os.getenv("SEARCH_DEBUG_DB_NORM_SAMPLE", "2048"))
SEARCH_DEBUG_SAVE_SEARCH_CROP = os.getenv("SEARCH_DEBUG_SAVE_SEARCH_CROP", "false").lower() == "true"
SEARCH_DEBUG_SAVE_LOW_SCORE_THRESHOLD = float(os.getenv("SEARCH_DEBUG_SAVE_LOW_SCORE_THRESHOLD", "0.35"))
SEARCH_DEBUG_SAVE_FRAME_IDS = os.getenv("SEARCH_DEBUG_SAVE_FRAME_IDS", "")
UNKNOWN_SCORE_THRESHOLD = float(os.getenv("UNKNOWN_SCORE_THRESHOLD", "0.45"))
UNKNOWN_GAP_THRESHOLD = float(os.getenv("UNKNOWN_GAP_THRESHOLD", "0.02"))
STABLE_RESULT_FRAMES = int(os.getenv("STABLE_RESULT_FRAMES", "3"))
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.65"))
HIGH_CONFIDENCE_THRESHOLD_OCCLUDED = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD_OCCLUDED", "0.78"))
OCR_AMBIGUOUS_GAP_THRESHOLD = float(os.getenv("OCR_AMBIGUOUS_GAP_THRESHOLD", "0.02"))
OCR_AMBIGUOUS_SCORE_THRESHOLD = float(os.getenv("OCR_AMBIGUOUS_SCORE_THRESHOLD", "0.50"))
OCR_RERANK_LAMBDA = float(os.getenv("OCR_RERANK_LAMBDA", "0.10"))
OCR_RERANK_LAMBDA_OCCLUDED = float(os.getenv("OCR_RERANK_LAMBDA_OCCLUDED", "0.05"))
OCR_LABEL_TOP_RATIO = float(os.getenv("OCR_LABEL_TOP_RATIO", "0.45"))
OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "2"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "40.0"))
OCR_SKIP_HAND_IOU_THRESHOLD = float(os.getenv("OCR_SKIP_HAND_IOU_THRESHOLD", "0.6"))
OCR_USE_ADAPTIVE_THRESHOLD = os.getenv("OCR_USE_ADAPTIVE_THRESHOLD", "false").lower() == "true"
OCR_DEBUG_LOG = os.getenv("OCR_DEBUG_LOG", "true").lower() == "true"
EMBEDDING_SELF_TEST_ENABLE = os.getenv("EMBEDDING_SELF_TEST_ENABLE", "true").lower() == "true"
EMBEDDING_SELF_TEST_IMAGE_PATH = os.getenv("EMBEDDING_SELF_TEST_IMAGE_PATH", "")
EMBEDDING_SELF_TEST_MIN_RAW = float(os.getenv("EMBEDDING_SELF_TEST_MIN_RAW", "0.75"))
