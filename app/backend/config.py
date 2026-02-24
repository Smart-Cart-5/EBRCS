"""Backend configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory (data = UltimateFusion model, data = original ensemble model)
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT.parent / "data")))

# File paths
EMBEDDINGS_PATH = str(DATA_DIR / "embeddings.npy")
LABELS_PATH = str(DATA_DIR / "labels.npy")
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
# Adapter (optional). If present, it should be loaded by the embedding runtime.
ADAPTER_DIR = str(DATA_DIR)
ADAPTER_MODEL_PATH = os.getenv(
    "ADAPTER_MODEL_PATH", str(DATA_DIR / "adapter_model.safetensors")
)
USE_ADAPTER = os.getenv("USE_ADAPTER", "true").lower() == "true"

# YOLO model (optional: if not found, falls back to background subtraction)
_DEFAULT_YOLO_PATH = str(PROJECT_ROOT.parent / "smartcart_hand_yolo11_best_arg_best.pt")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", _DEFAULT_YOLO_PATH)
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
USE_YOLO = os.getenv("USE_YOLO", "true").lower() == "true"
YOLO_HAND_CLASS_ALIASES = os.getenv(
    "YOLO_HAND_CLASS_ALIASES",
    "hand,hands,palm",
).strip()
YOLO_PRODUCT_CLASS_ALIASES = os.getenv(
    "YOLO_PRODUCT_CLASS_ALIASES",
    "object,product,item,goods",
).strip()

# Cart ROI segmentation (Roboflow semantic segmentation)
# Guardrail flag for external API calls. User start mode remains primary.
CART_ROI_ENABLED = os.getenv("CART_ROI_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
CART_ROI_EVERY_N_FRAMES = int(os.getenv("CART_ROI_EVERY_N_FRAMES", "10"))
CART_ROI_DEBUG = os.getenv("CART_ROI_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
CART_ROI_DEBUG_DIR = os.getenv("CART_ROI_DEBUG_DIR", str(PROJECT_ROOT / "debug_cart_roi"))
CART_ROI_CLASS_NAME = os.getenv("CART_ROI_CLASS_NAME", "cartline").strip()
CART_ROI_CLASS_ALIASES = os.getenv(
    "CART_ROI_CLASS_ALIASES",
    "cartline,cart,shopping_cart,trolley",
).strip()

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
SEARCH_COOLDOWN_SEC = float(os.getenv("SEARCH_COOLDOWN_SEC", "1.0"))
SEARCH_COOLDOWN_MS = int(
    float(os.getenv("SEARCH_COOLDOWN_MS", str(int(SEARCH_COOLDOWN_SEC * 1000.0))))
)
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
SEARCH_CROP_MIN_SIDE_AFTER_EXPAND = int(
    os.getenv("SEARCH_CROP_MIN_SIDE_AFTER_EXPAND", "240")
)
SEARCH_CROP_ASPECT_MIN = float(os.getenv("SEARCH_CROP_ASPECT_MIN", "0.5"))
SEARCH_CROP_ASPECT_MAX = float(os.getenv("SEARCH_CROP_ASPECT_MAX", "2.0"))
SEARCH_CROP_EDGE_PAD_RATIO = float(os.getenv("SEARCH_CROP_EDGE_PAD_RATIO", "0.02"))
SEARCH_SELECT_W_CONF = float(os.getenv("SEARCH_SELECT_W_CONF", "0.4"))
SEARCH_SELECT_W_AREA = float(os.getenv("SEARCH_SELECT_W_AREA", "0.4"))
SEARCH_SELECT_W_ROI = float(os.getenv("SEARCH_SELECT_W_ROI", "0.2"))
ROI_IOU_MIN = float(os.getenv("ROI_IOU_MIN", "0.15"))
ROI_CENTER_PASS = os.getenv("ROI_CENTER_PASS", "true").lower() == "true"
# Allow HAND_CONF_TH alias for quick runtime tuning in webcam tests.
HAND_CONF_MIN = float(os.getenv("HAND_CONF_TH", os.getenv("HAND_CONF_MIN", "0.35")))
HAND_APPLY_CART_ROI_FILTER = os.getenv("HAND_APPLY_CART_ROI_FILTER", "false").lower() == "true"
HAND_OVERLAP_IOU = float(os.getenv("HAND_OVERLAP_IOU", "0.25"))
SEARCH_MASK_HAND = os.getenv("SEARCH_MASK_HAND", "false").lower() == "true"
MIN_PRODUCT_BBOX_AREA = int(os.getenv("MIN_PRODUCT_BBOX_AREA", str(MIN_AREA)))
MAX_PRODUCTS_PER_FRAME = int(os.getenv("MAX_PRODUCTS_PER_FRAME", "3"))
MATCH_THRESHOLD = 0.62
COUNT_COOLDOWN_SECONDS = 3.0  # 중복 방지: 동일 상품 3초 내 재카운트 방지
ROI_CLEAR_FRAMES = 8
STREAM_TARGET_WIDTH = 960  # Restored for better quality
VIRTUAL_DISTANCE_SCALE = float(os.getenv("VIRTUAL_DISTANCE_SCALE", "1.0"))
STREAM_SEND_IMAGES = (
    os.getenv("STREAM_SEND_IMAGES", "false").lower() == "true"
)  # Send images in WebSocket responses (default: false, JSON only)
CHECKOUT_QUEUE_MAXSIZE = int(os.getenv("CHECKOUT_QUEUE_MAXSIZE", "1"))
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "3"))
VOTE_WINDOW_SIZE = int(os.getenv("VOTE_WINDOW_SIZE", "5"))
VOTE_MIN_SAMPLES = int(os.getenv("VOTE_MIN_SAMPLES", "3"))
ASSOCIATION_IOU_WEIGHT = float(os.getenv("ASSOCIATION_IOU_WEIGHT", "0.5"))
ASSOCIATION_DIST_WEIGHT = float(os.getenv("ASSOCIATION_DIST_WEIGHT", "0.5"))
ASSOCIATION_MAX_CENTER_DIST = float(os.getenv("ASSOCIATION_MAX_CENTER_DIST", "0.35"))
ASSOCIATION_MIN_SCORE = float(os.getenv("ASSOCIATION_MIN_SCORE", "0.1"))
EVENT_MODE = os.getenv("EVENT_MODE", "true").lower() == "true"
HAND_EVENT_ASSOC_TOP_K = int(os.getenv("HAND_EVENT_ASSOC_TOP_K", "3"))
HAND_EVENT_ASSOC_STABLE_FRAMES = int(os.getenv("HAND_EVENT_ASSOC_STABLE_FRAMES", "1"))
HAND_EVENT_MIN_ASSOC_SCORE = float(os.getenv("HAND_EVENT_MIN_ASSOC_SCORE", "0.03"))
HAND_EVENT_MIN_ASSOC_IOU = float(os.getenv("HAND_EVENT_MIN_ASSOC_IOU", "0.0"))
HAND_EVENT_ALLOW_DET_FALLBACK = os.getenv("HAND_EVENT_ALLOW_DET_FALLBACK", "true").lower() == "true"
HAND_EVENT_DET_FALLBACK_MIN_SCORE = float(os.getenv("HAND_EVENT_DET_FALLBACK_MIN_SCORE", "0.01"))
HAND_EVENT_ADD_EVIDENCE_FRAMES = int(os.getenv("HAND_EVENT_ADD_EVIDENCE_FRAMES", "2"))
HAND_EVENT_REMOVE_EVIDENCE_FRAMES = int(os.getenv("HAND_EVENT_REMOVE_EVIDENCE_FRAMES", "2"))
HAND_EVENT_REMOVE_MISSING_GRACE_FRAMES = int(os.getenv("HAND_EVENT_REMOVE_MISSING_GRACE_FRAMES", "3"))
HAND_EVENT_CANDIDATE_TIMEOUT_S = float(os.getenv("HAND_EVENT_CANDIDATE_TIMEOUT_S", "1.2"))
HAND_EVENT_COOLDOWN_S = float(os.getenv("HAND_EVENT_COOLDOWN_S", "0.7"))
HAND_EVENT_CANDIDATE_SWITCH_MIN_DELTA = float(
    os.getenv("HAND_EVENT_CANDIDATE_SWITCH_MIN_DELTA", "0.08")
)
HAND_EVENT_TRACK_IOU_MATCH_THRESHOLD = float(
    os.getenv("HAND_EVENT_TRACK_IOU_MATCH_THRESHOLD", "0.05")
)
HAND_EVENT_TRACK_MAX_MISSED_FRAMES = int(os.getenv("HAND_EVENT_TRACK_MAX_MISSED_FRAMES", "8"))
HAND_EVENT_CANDIDATE_ROI_RELAX = os.getenv("HAND_EVENT_CANDIDATE_ROI_RELAX", "true").lower() == "true"
HAND_EVENT_CANDIDATE_HAND_NEAR_DIST = float(
    os.getenv("HAND_EVENT_CANDIDATE_HAND_NEAR_DIST", "0.22")
)
HAND_EVENT_HAND_REPRESENTATIVE_POINT = os.getenv("HAND_EVENT_HAND_REPRESENTATIVE_POINT", "center")
HAND_EVENT_OBJECT_REPRESENTATIVE_POINT = os.getenv(
    "HAND_EVENT_OBJECT_REPRESENTATIVE_POINT",
    "bottom_center",
)
HAND_EVENT_ROI_MARGIN_PX = float(os.getenv("HAND_EVENT_ROI_MARGIN_PX", "6.0"))
HAND_EVENT_ROI_MARGIN_RATIO = float(os.getenv("HAND_EVENT_ROI_MARGIN_RATIO", "0.0"))
HAND_EVENT_DEBUG_LOG = os.getenv("HAND_EVENT_DEBUG_LOG", "true").lower() == "true"
HAND_EVENT_DEBUG_OVERLAY = os.getenv("HAND_EVENT_DEBUG_OVERLAY", "false").lower() == "true"
HAND_EVENT_WS_DEBUG = os.getenv("HAND_EVENT_WS_DEBUG", "true").lower() == "true"
CHECKOUT_DEBUG_TICK_LOG = os.getenv("CHECKOUT_DEBUG_TICK_LOG", "true").lower() == "true"
CHECKOUT_AUTO_CONFIRM_ROI = os.getenv("CHECKOUT_AUTO_CONFIRM_ROI", "false").lower() == "true"
CHECKOUT_AUTO_CONFIRM_ROI_MIN_RATIO = float(os.getenv("CHECKOUT_AUTO_CONFIRM_ROI_MIN_RATIO", "0.005"))
CHECKOUT_AUTO_CONFIRM_ROI_MIN_POLY_POINTS = int(os.getenv("CHECKOUT_AUTO_CONFIRM_ROI_MIN_POLY_POINTS", "3"))
EVENT_ROI_TOO_LARGE_RATIO = float(os.getenv("EVENT_ROI_TOO_LARGE_RATIO", "0.90"))
EVENT_ROI_EDGE_WARN_MARGIN = float(os.getenv("EVENT_ROI_EDGE_WARN_MARGIN", "0.05"))
EVENT_ROI_BLOCK_WHEN_TOO_LARGE = os.getenv("EVENT_ROI_BLOCK_WHEN_TOO_LARGE", "false").lower() == "true"
SNAPSHOT_MAX_FRAMES = int(os.getenv("SNAPSHOT_MAX_FRAMES", "8"))
WARP_MODE = os.getenv("WARP_MODE", "false").lower() == "true"
WARP_AUTO_ENABLE_WITH_CALIB4 = os.getenv("WARP_AUTO_ENABLE_WITH_CALIB4", "false").lower() == "true"
DETECT_OUTSIDE_EVENT_ROI = os.getenv("DETECT_OUTSIDE_EVENT_ROI", "true").lower() == "true"
DEBUG_DISABLE_OBJECT_ROI_GATE = os.getenv("DEBUG_DISABLE_OBJECT_ROI_GATE", "false").lower() == "true"
DEBUG_EVENT_ROI_ONLY = os.getenv("DEBUG_EVENT_ROI_ONLY", "false").lower() == "true"
DEBUG_BYPASS_OBJECT_FILTER = os.getenv("DEBUG_BYPASS_OBJECT_FILTER", "false").lower() == "true"
OBJECT_KEEP_IF_HAND_OVERLAP = os.getenv("OBJECT_KEEP_IF_HAND_OVERLAP", "true").lower() == "true"
OBJECT_KEEP_HAND_OVERLAP_IOU = float(os.getenv("OBJECT_KEEP_HAND_OVERLAP_IOU", "0.1"))
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
SEARCH_DEBUG_CROP_DIR = os.getenv(
    "SEARCH_DEBUG_CROP_DIR", str(PROJECT_ROOT / "debug_crops")
)
SEARCH_DEBUG_LOG_INTERVAL_MS = int(os.getenv("SEARCH_DEBUG_LOG_INTERVAL_MS", "5000"))
SEARCH_DEBUG_SAVE_INTERVAL_MS = int(os.getenv("SEARCH_DEBUG_SAVE_INTERVAL_MS", "1000"))
SEARCH_DEBUG_DB_NORM_SAMPLE = int(os.getenv("SEARCH_DEBUG_DB_NORM_SAMPLE", "2048"))
SEARCH_DEBUG_SAVE_SEARCH_CROP = (
    os.getenv("SEARCH_DEBUG_SAVE_SEARCH_CROP", "false").lower() == "true"
)
SEARCH_DEBUG_SAVE_LOW_SCORE_THRESHOLD = float(
    os.getenv("SEARCH_DEBUG_SAVE_LOW_SCORE_THRESHOLD", "0.35")
)
SEARCH_DEBUG_SAVE_FRAME_IDS = os.getenv("SEARCH_DEBUG_SAVE_FRAME_IDS", "")
UNKNOWN_SCORE_THRESHOLD = float(os.getenv("UNKNOWN_SCORE_THRESHOLD", "0.45"))
UNKNOWN_GAP_THRESHOLD = float(os.getenv("UNKNOWN_GAP_THRESHOLD", "0.02"))
STABLE_RESULT_FRAMES = int(os.getenv("STABLE_RESULT_FRAMES", "3"))
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.65"))
HIGH_CONFIDENCE_THRESHOLD_OCCLUDED = float(
    os.getenv("HIGH_CONFIDENCE_THRESHOLD_OCCLUDED", "0.78")
)
OCR_AMBIGUOUS_GAP_THRESHOLD = float(os.getenv("OCR_AMBIGUOUS_GAP_THRESHOLD", "0.02"))
OCR_AMBIGUOUS_SCORE_THRESHOLD = float(
    os.getenv("OCR_AMBIGUOUS_SCORE_THRESHOLD", "0.50")
)
OCR_RERANK_LAMBDA = float(os.getenv("OCR_RERANK_LAMBDA", "0.10"))
OCR_RERANK_LAMBDA_OCCLUDED = float(os.getenv("OCR_RERANK_LAMBDA_OCCLUDED", "0.05"))
OCR_LABEL_TOP_RATIO = float(os.getenv("OCR_LABEL_TOP_RATIO", "0.45"))
OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "2"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "40.0"))
OCR_SKIP_HAND_IOU_THRESHOLD = float(os.getenv("OCR_SKIP_HAND_IOU_THRESHOLD", "0.6"))
OCR_USE_ADAPTIVE_THRESHOLD = (
    os.getenv("OCR_USE_ADAPTIVE_THRESHOLD", "false").lower() == "true"
)
OCR_DEBUG_LOG = os.getenv("OCR_DEBUG_LOG", "true").lower() == "true"
OCR_ENABLED = os.getenv("OCR_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
OCR_COOLDOWN_SEC = float(os.getenv("OCR_COOLDOWN_SEC", "10"))
EMBEDDING_SELF_TEST_ENABLE = (
    os.getenv("EMBEDDING_SELF_TEST_ENABLE", "true").lower() == "true"
)
EMBEDDING_SELF_TEST_IMAGE_PATH = os.getenv("EMBEDDING_SELF_TEST_IMAGE_PATH", "")
EMBEDDING_SELF_TEST_MIN_RAW = float(os.getenv("EMBEDDING_SELF_TEST_MIN_RAW", "0.75"))
