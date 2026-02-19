"""Backend configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
ROI_CLEAR_FRAMES = 36  # 3초 @ 12FPS (기존 8=0.64초는 너무 짧아 오검출 발생)
STREAM_TARGET_WIDTH = 960  # Restored for better quality
STREAM_SEND_IMAGES = os.getenv("STREAM_SEND_IMAGES", "false").lower() == "true"  # Send images in WebSocket responses (default: false, JSON only)
BG_WARMUP_FRAMES = int(os.getenv("BG_WARMUP_FRAMES", "24"))  # 초반 KNN 잔상 안정화용 워밍업 프레임

# DeepSORT Object Tracking (Phase 3)
# 역할: 기존 인식 로직은 그대로, DeepSORT는 Track ID 기반 중복 방지만 담당
USE_DEEPSORT = os.getenv("USE_DEEPSORT", "true").lower() == "true"
DEEPSORT_MAX_AGE = 30       # 물체 소실 후 Track 유지 프레임 수 (30 ≈ 0.5초 @ 60FPS)
DEEPSORT_N_INIT = 1         # 즉시 Track 확정 (기존 인식 속도 유지)
DEEPSORT_MAX_IOU_DISTANCE = 0.9  # IOU 매칭 임계값 (높을수록 이동 중 Track 유지 유리)
DEEPSORT_EMBEDDER = None    # None = IOU만 사용 (mobilenet 연산 제거 → 속도 유지)

# 방향 기반 장바구니 추가/제거 (카메라 좌표계: y=0 상단, y 증가=아래)
# - ADD_DIRECTION="down": 위→아래 이동(dy > 0) 시 담기
# - ADD_DIRECTION="up":   아래→위 이동(dy < 0) 시 담기
ADD_DIRECTION = os.getenv("ADD_DIRECTION", "down")
DIRECTION_MIN_MOVEMENT = float(os.getenv("DIRECTION_MIN_MOVEMENT", "0.05"))
DIRECTION_HISTORY_FRAMES = int(os.getenv("DIRECTION_HISTORY_FRAMES", "20"))
DIRECTION_MIN_HISTORY_POINTS = int(os.getenv("DIRECTION_MIN_HISTORY_POINTS", "4"))
INLINE_DIRECTION_MIN_MOVEMENT = float(os.getenv("INLINE_DIRECTION_MIN_MOVEMENT", "0.08"))
INLINE_DIRECTION_MIN_HISTORY_POINTS = int(os.getenv("INLINE_DIRECTION_MIN_HISTORY_POINTS", "8"))
DIRECTION_SIGN_CONSISTENCY = float(os.getenv("DIRECTION_SIGN_CONSISTENCY", "0.7"))
DIRECTION_SIGN_EPSILON = float(os.getenv("DIRECTION_SIGN_EPSILON", "0.003"))
FAST_DECISION_MIN_MOVEMENT = float(os.getenv("FAST_DECISION_MIN_MOVEMENT", "0.12"))
FAST_DECISION_MIN_HISTORY_POINTS = int(os.getenv("FAST_DECISION_MIN_HISTORY_POINTS", "4"))
FAST_DECISION_SIGN_CONSISTENCY = float(os.getenv("FAST_DECISION_SIGN_CONSISTENCY", "0.85"))
INFER_BURST_TRACK_AGE = int(os.getenv("INFER_BURST_TRACK_AGE", "6"))
TRACK_MIN_AGE_FRAMES = int(os.getenv("TRACK_MIN_AGE_FRAMES", "6"))
TRACK_LABEL_WINDOW = int(os.getenv("TRACK_LABEL_WINDOW", "6"))
TRACK_LABEL_MIN_VOTES = int(os.getenv("TRACK_LABEL_MIN_VOTES", "2"))
TRACK_STATE_TTL_FRAMES = int(os.getenv("TRACK_STATE_TTL_FRAMES", "18"))
SOFT_REENTRY_FRAMES = int(os.getenv("SOFT_REENTRY_FRAMES", "5"))
OPPOSITE_ACTION_COOLDOWN_SECONDS = float(
    os.getenv("OPPOSITE_ACTION_COOLDOWN_SECONDS", "1.5")
)
LABEL_STALE_FRAMES = int(os.getenv("LABEL_STALE_FRAMES", "4"))

# 후처리: 무시할 상품 라벨 (FAISS 매칭 결과에서 제외)
IGNORE_LABELS: set[str] = {
    "10168_CJ스팸340G",
}
