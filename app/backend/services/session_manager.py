"""Per-user checkout session management.

Replaces Streamlit's st.session_state with an in-memory dict keyed by
session UUID, each holding its own billing state, bg_subtractor, ROI, etc.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

PHASE_IDLE = "IDLE"
PHASE_ROI_CALIBRATING = "ROI_CALIBRATING"
PHASE_CHECKOUT_RUNNING = "CHECKOUT_RUNNING"


@dataclass
class CheckoutSession:
    """State for a single checkout session (one user/tab)."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Mutable state dict -- passed directly to process_checkout_frame(state=...)
    state: dict[str, Any] = field(default_factory=lambda: {
        "billing_items": {},
        "item_scores": {},
        "last_seen_at": {},
        "candidate_votes": {},
        "candidate_history": [],
        "topk_candidates": [],
        "confidence": 0.0,
        "best_pair": None,
        "event_state": "IDLE",
        "occluded_by_hand": False,
        "overlap_hand_iou_max": 0.0,
        "search_crop_min_side_before": None,
        "search_crop_min_side_after": None,
        "search_crop_padding_ratio": 0.0,
        "ocr_used": False,
        "ocr_attempted": False,
        "ocr_error": None,
        "ocr_skip_reason": None,
        "ocr_text": "",
        "ocr_matched_keywords": {},
        "ocr_text_score": 0.0,
        "ocr_ambiguous": False,
        "ocr_chosen_slice": None,
        "ocr_chosen_psm": None,
        "ocr_chosen_thresh": None,
        "ocr_chosen_conf_cut": None,
        "ocr_chosen_lang": None,
        "ocr_token_count": 0,
        "ocr_korean_char_count": 0,
        "ocr_avg_conf": 0.0,
        "ocr_reranked_topk": [],
        "did_search": False,
        "skip_reason": "init",
        "last_result_name": None,
        "last_result_score": None,
        "last_result_topk": [],
        "last_result_confidence": None,
        "last_result_at_ms": None,
        "result_label": "-",
        "is_unknown": True,
        "match_score_raw": None,
        "match_top2_raw": None,
        "match_score_percent": None,
        "match_gap": None,
        "match_gap_reason": None,
        "unknown_reason": "init",
        "in_cart_sequence": [],
        "last_label": "-",
        "last_score": 0.0,
        "last_status": "대기",
        "roi_occupied": False,
        "roi_empty_frames": 0,
        "phase": PHASE_IDLE,
        "_checkout_started": False,
        "_cart_roi_confirmed": False,
        "_cart_roi_mask": None,
        "_cart_roi_last_update_frame": -1,
        "_cart_roi_mask_pending": None,
        "_cart_roi_pending_last_update_frame": -1,
        "_cart_roi_pending_polygon": None,
        "_cart_roi_pending_ratio": 0.0,
        "_cart_roi_preview_last_sent_ms": 0,
        "_cart_roi_auto_enabled": None,
        "_checkout_start_mode": None,
        "_checkout_user_message": None,
        "_cart_roi_last_segmenter": None,
        "_cart_roi_last_error": None,
        "_cart_roi_invalid_reason": None,
        "_cart_roi_calib_log_last_ms": 0,
    })

    # OpenCV background subtractor -- per-session, not serializable
    bg_subtractor: Any = field(default=None)

    # Frame counter for DETECT_EVERY_N_FRAMES gating
    frame_count: int = 0

    # Normalized ROI polygon [[x, y], ...] in [0, 1] range, or None
    roi_poly_norm: list[list[float]] | None = None
    warp_enabled: bool = False
    warp_points_norm: list[list[float]] | None = None
    warp_size: tuple[int, int] = (640, 480)

    # Video upload task tracking
    video_task_id: str | None = None
    video_progress: dict[str, Any] = field(default_factory=lambda: {
        "done": False,
        "progress": 0.0,
        "total_frames": 0,
        "current_frame": 0,
    })

    def __post_init__(self) -> None:
        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=300)

    def touch(self) -> None:
        self.last_active = time.time()

    def reset_billing(self) -> None:
        self.state["billing_items"] = {}
        self.state["item_scores"] = {}
        self.state["last_seen_at"] = {}
        self.state["candidate_votes"] = {}
        self.state["candidate_history"] = []
        self.state["topk_candidates"] = []
        self.state["confidence"] = 0.0
        self.state["best_pair"] = None
        self.state["event_state"] = "IDLE"
        self.state["occluded_by_hand"] = False
        self.state["overlap_hand_iou_max"] = 0.0
        self.state["search_crop_min_side_before"] = None
        self.state["search_crop_min_side_after"] = None
        self.state["search_crop_padding_ratio"] = 0.0
        self.state["ocr_used"] = False
        self.state["ocr_attempted"] = False
        self.state["ocr_error"] = None
        self.state["ocr_skip_reason"] = None
        self.state["ocr_text"] = ""
        self.state["ocr_matched_keywords"] = {}
        self.state["ocr_text_score"] = 0.0
        self.state["ocr_ambiguous"] = False
        self.state["ocr_chosen_slice"] = None
        self.state["ocr_chosen_psm"] = None
        self.state["ocr_chosen_thresh"] = None
        self.state["ocr_chosen_conf_cut"] = None
        self.state["ocr_chosen_lang"] = None
        self.state["ocr_token_count"] = 0
        self.state["ocr_korean_char_count"] = 0
        self.state["ocr_avg_conf"] = 0.0
        self.state["ocr_reranked_topk"] = []
        self.state["did_search"] = False
        self.state["skip_reason"] = "init"
        self.state["last_result_name"] = None
        self.state["last_result_score"] = None
        self.state["last_result_topk"] = []
        self.state["last_result_confidence"] = None
        self.state["last_result_at_ms"] = None
        self.state["result_label"] = "-"
        self.state["is_unknown"] = True
        self.state["match_score_raw"] = None
        self.state["match_top2_raw"] = None
        self.state["match_score_percent"] = None
        self.state["match_gap"] = None
        self.state["match_gap_reason"] = None
        self.state["unknown_reason"] = "init"
        self.state["in_cart_sequence"] = []
        self.state["last_label"] = "-"
        self.state["last_score"] = 0.0
        self.state["last_status"] = "대기"
        self.state["roi_occupied"] = False
        self.state["roi_empty_frames"] = 0
        self.state["phase"] = PHASE_IDLE
        self.state["_checkout_started"] = False
        self.state["_cart_roi_confirmed"] = False
        self.state["_cart_roi_mask"] = None
        self.state["_cart_roi_last_update_frame"] = -1
        self.state["_cart_roi_mask_pending"] = None
        self.state["_cart_roi_pending_last_update_frame"] = -1
        self.state["_cart_roi_pending_polygon"] = None
        self.state["_cart_roi_pending_ratio"] = 0.0
        self.state["_cart_roi_preview_last_sent_ms"] = 0
        self.state["_cart_roi_auto_enabled"] = None
        self.state["_checkout_start_mode"] = None
        self.state["_checkout_user_message"] = None
        self.state["_cart_roi_last_segmenter"] = None
        self.state["_cart_roi_last_error"] = None
        self.state["_cart_roi_invalid_reason"] = None
        self.state["_cart_roi_calib_log_last_ms"] = 0
        self.state.pop("_cart_roi_last_debug_save_ms", None)
        self.state.pop("_event_engine", None)
        self.state.pop("_snapshot_buffer", None)
        self.frame_count = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=300)

    def start_checkout(self, *, require_roi_calibration: bool) -> str:
        if not bool(self.state.get("_checkout_started", False)):
            self.state["_checkout_started"] = True
            self.state["_cart_roi_confirmed"] = False
            self.state["_cart_roi_mask_pending"] = None
            self.state["_cart_roi_pending_last_update_frame"] = -1
            self.state["_cart_roi_pending_polygon"] = None
            self.state["_cart_roi_pending_ratio"] = 0.0
            self.state["_cart_roi_preview_last_sent_ms"] = 0
            self.state["phase"] = (
                PHASE_ROI_CALIBRATING if require_roi_calibration else PHASE_CHECKOUT_RUNNING
            )
        return str(self.state.get("phase", PHASE_IDLE))

    def confirm_pending_cart_roi(self) -> bool:
        pending = self.state.get("_cart_roi_mask_pending")
        if not isinstance(pending, np.ndarray):
            return False
        self.state["_cart_roi_mask"] = pending
        self.state["_cart_roi_confirmed"] = True
        self.state["phase"] = PHASE_CHECKOUT_RUNNING
        self.state["_cart_roi_mask_pending"] = None
        self.state["_cart_roi_pending_last_update_frame"] = -1
        pending_polygon = self.state.get("_cart_roi_pending_polygon")
        if isinstance(pending_polygon, list) and len(pending_polygon) >= 3:
            self.roi_poly_norm = pending_polygon
        self.state["_cart_roi_pending_polygon"] = None
        self.state["_cart_roi_pending_ratio"] = 0.0
        self.state["_cart_roi_preview_last_sent_ms"] = 0
        return True

    def retry_cart_roi(self) -> None:
        self.state["_cart_roi_mask_pending"] = None
        self.state["_cart_roi_pending_last_update_frame"] = -1
        self.state["_cart_roi_pending_polygon"] = None
        self.state["_cart_roi_pending_ratio"] = 0.0
        self.state["_cart_roi_preview_last_sent_ms"] = 0
        self.state["_cart_roi_confirmed"] = False
        self.state["phase"] = PHASE_ROI_CALIBRATING

    def force_checkout_running(self) -> None:
        self.state["_checkout_started"] = True
        self.state["phase"] = PHASE_CHECKOUT_RUNNING

    def set_cart_roi_auto_enabled(self, enabled: bool) -> None:
        self.state["_cart_roi_auto_enabled"] = bool(enabled)
        self.state["_checkout_start_mode"] = "auto_roi" if bool(enabled) else "no_roi"
        self.state["_checkout_user_message"] = None
        self.state["_cart_roi_last_segmenter"] = None
        self.state["_cart_roi_last_error"] = None
        self.state["_cart_roi_invalid_reason"] = None
        self.state["_cart_roi_calib_log_last_ms"] = 0
        self.state["_checkout_started"] = False
        self.state["phase"] = PHASE_IDLE
        self.state["_cart_roi_confirmed"] = False
        self.state["_cart_roi_mask"] = None
        self.state["_cart_roi_last_update_frame"] = -1
        self.state["_cart_roi_mask_pending"] = None
        self.state["_cart_roi_pending_last_update_frame"] = -1
        self.state["_cart_roi_pending_polygon"] = None
        self.state["_cart_roi_pending_ratio"] = 0.0
        self.state["_cart_roi_preview_last_sent_ms"] = 0
        if bool(enabled):
            # Auto ROI mode starts from fresh calibration result.
            self.roi_poly_norm = None
        else:
            # Legacy behavior: keep inference running with full-frame ROI.
            self.roi_poly_norm = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def start_checkout_with_mode(self, mode: str) -> str:
        start_mode = "auto_roi" if str(mode) == "auto_roi" else "no_roi"
        if start_mode == "auto_roi":
            self.state["_cart_roi_auto_enabled"] = True
            self.state["_checkout_start_mode"] = "auto_roi"
            self.state["_checkout_user_message"] = None
            self.state["_cart_roi_last_segmenter"] = None
            self.state["_cart_roi_last_error"] = None
            self.state["_cart_roi_invalid_reason"] = None
            self.state["_cart_roi_calib_log_last_ms"] = 0
            self.state["_checkout_started"] = True
            self.state["phase"] = PHASE_ROI_CALIBRATING
            self.state["_cart_roi_confirmed"] = False
            self.state["_cart_roi_mask"] = None
            self.state["_cart_roi_last_update_frame"] = -1
            self.state["_cart_roi_mask_pending"] = None
            self.state["_cart_roi_pending_last_update_frame"] = -1
            self.state["_cart_roi_pending_polygon"] = None
            self.state["_cart_roi_pending_ratio"] = 0.0
            self.state["_cart_roi_preview_last_sent_ms"] = 0
            self.roi_poly_norm = None
            return PHASE_ROI_CALIBRATING

        # no_roi start: preserve legacy full-frame ROI and run immediately.
        self.state["_cart_roi_auto_enabled"] = False
        self.state["_checkout_start_mode"] = "no_roi"
        self.state["_checkout_user_message"] = None
        self.state["_cart_roi_last_segmenter"] = None
        self.state["_cart_roi_last_error"] = None
        self.state["_cart_roi_invalid_reason"] = None
        self.state["_cart_roi_calib_log_last_ms"] = 0
        self.state["_checkout_started"] = True
        self.state["phase"] = PHASE_CHECKOUT_RUNNING
        self.state["_cart_roi_confirmed"] = False
        self.state["_cart_roi_mask"] = None
        self.state["_cart_roi_last_update_frame"] = -1
        self.state["_cart_roi_mask_pending"] = None
        self.state["_cart_roi_pending_last_update_frame"] = -1
        self.state["_cart_roi_pending_polygon"] = None
        self.state["_cart_roi_pending_ratio"] = 0.0
        self.state["_cart_roi_preview_last_sent_ms"] = 0
        self.roi_poly_norm = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        return PHASE_CHECKOUT_RUNNING

    def get_roi_polygon(self, frame_shape: tuple[int, ...]) -> np.ndarray | None:
        """Convert normalized ROI to pixel coordinates for the given frame."""
        if not self.roi_poly_norm or len(self.roi_poly_norm) < 3:
            return None
        h, w = frame_shape[:2]
        pts = []
        for x_norm, y_norm in self.roi_poly_norm:
            x = int(max(0.0, min(1.0, x_norm)) * w)
            y = int(max(0.0, min(1.0, y_norm)) * h)
            pts.append([x, y])
        return np.array(pts, dtype=np.int32)


class SessionManager:
    """Manages checkout sessions with TTL expiration."""

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 50) -> None:
        self._sessions: dict[str, CheckoutSession] = {}
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions

    def create(self) -> CheckoutSession:
        self.cleanup_expired()
        if len(self._sessions) >= self._max_sessions:
            # Evict oldest inactive session
            oldest = min(self._sessions.values(), key=lambda s: s.last_active)
            del self._sessions[oldest.session_id]

        sid = str(uuid.uuid4())
        session = CheckoutSession(session_id=sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> CheckoutSession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            k for k, v in self._sessions.items()
            if now - v.last_active > self._ttl
        ]
        for k in expired:
            del self._sessions[k]

    @property
    def active_count(self) -> int:
        return len(self._sessions)
