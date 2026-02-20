from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("backend.roi.cart_segmenter")


class RoboflowCartSegmenter:
    """Roboflow semantic segmentation client for cart ROI mask generation."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = "smartcart-fd4z1",
        version: int = 4,
        every_n_frames: int = 10,
        debug: bool = False,
        debug_dir: str | None = None,
        timeout_seconds: float = 8.0,
        target_class_name: str = "cartline",
        class_aliases: str | list[str] | tuple[str, ...] = ("cartline", "cart", "shopping_cart", "trolley"),
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.endpoint = str(endpoint).strip()
        self.version = int(version)
        self.every_n_frames = max(1, int(every_n_frames))
        self.debug = bool(debug)
        self.debug_dir = str(debug_dir or "debug_cart_roi")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.target_class_name = str(target_class_name or "cartline").strip().lower()
        self.class_aliases = self._parse_aliases(class_aliases, self.target_class_name)
        self.enabled = bool(self.api_key)
        self._endpoint_url = f"https://serverless.roboflow.com/{self.endpoint}/{self.version}"

    def get_or_update_mask(
        self,
        *,
        frame_bgr: np.ndarray,
        frame_count: int,
        frame_id: int | None,
        session_id: str | None,
        state: MutableMapping[str, Any],
        mask_key: str = "_cart_roi_mask",
        last_update_key: str = "_cart_roi_last_update_frame",
    ) -> np.ndarray | None:
        """Return cached ROI mask and refresh it every N frames."""
        telemetry: dict[str, Any] = {
            "ts_ms": int(time.time() * 1000),
            "frame_count": int(frame_count),
            "frame_id": int(frame_id) if frame_id is not None else None,
            "frame_shape": [int(frame_bgr.shape[0]), int(frame_bgr.shape[1])],
            "called": False,
            "status": "init",
            "skip_reason": None,
            "roboflow_response_ok": False,
            "roboflow_latency_ms": None,
            "decoded_mask_shape": None,
            "class_map_has_target": False,
            "matched_class_name": None,
            "class_map_values_sample": [],
            "target_id": None,
        }
        if not self.enabled:
            telemetry["status"] = "skipped"
            telemetry["skip_reason"] = "disabled"
            state["_cart_roi_last_segmenter"] = telemetry
            state["_cart_roi_last_error"] = "segmenter_disabled"
            return None

        cached = state.get(mask_key)
        if isinstance(cached, np.ndarray):
            if cached.shape[:2] != frame_bgr.shape[:2]:
                cached = None
                state.pop(mask_key, None)

        last_update = int(state.get(last_update_key, -10_000_000))
        should_refresh = cached is None or (int(frame_count) - last_update) >= self.every_n_frames
        if not should_refresh:
            telemetry["status"] = "skipped"
            telemetry["skip_reason"] = "nframe"
            state["_cart_roi_last_segmenter"] = telemetry
            return cached

        telemetry["called"] = True
        decoded = self._request_and_decode(frame_bgr)
        telemetry["roboflow_response_ok"] = bool(decoded.get("response_ok", False))
        telemetry["roboflow_latency_ms"] = decoded.get("latency_ms")
        semantic_mask = decoded.get("semantic_mask")
        class_map = decoded.get("class_map")
        if not isinstance(semantic_mask, np.ndarray):
            telemetry["status"] = "error"
            telemetry["skip_reason"] = str(decoded.get("error_reason", "decode_failed"))
            state["_cart_roi_last_segmenter"] = telemetry
            state["_cart_roi_last_error"] = str(telemetry["skip_reason"])
            return cached
        telemetry["decoded_mask_shape"] = [int(semantic_mask.shape[0]), int(semantic_mask.shape[1])]
        if not isinstance(class_map, dict) or not class_map:
            telemetry["status"] = "error"
            telemetry["skip_reason"] = "class_map_missing"
            state["_cart_roi_last_segmenter"] = telemetry
            state["_cart_roi_last_error"] = "class_map_missing"
            return cached

        cart_id, matched_class_name = self._find_target_class_id(class_map)
        telemetry["class_map_has_target"] = cart_id is not None
        telemetry["matched_class_name"] = matched_class_name
        telemetry["class_map_values_sample"] = self._class_name_candidates(class_map)
        telemetry["target_id"] = int(cart_id) if cart_id is not None else None
        if cart_id is None:
            telemetry["status"] = "error"
            telemetry["skip_reason"] = "cart_class_not_found"
            logger.warning(
                "Cart ROI class id not found. target=%s aliases=%s class_names=%s",
                self.target_class_name,
                sorted(self.class_aliases),
                telemetry["class_map_values_sample"],
            )
            state["_cart_roi_last_segmenter"] = telemetry
            state["_cart_roi_last_error"] = "cart_class_not_found"
            return cached

        roi_mask = self._build_cart_roi_mask(semantic_mask, cart_id, frame_bgr.shape[:2])
        if roi_mask is None:
            telemetry["status"] = "error"
            telemetry["skip_reason"] = "empty_cart_mask"
            state["_cart_roi_last_segmenter"] = telemetry
            state["_cart_roi_last_error"] = "empty_cart_mask"
            return cached

        roi_mask = self._postprocess_mask(roi_mask)
        state[mask_key] = roi_mask
        state[last_update_key] = int(frame_count)
        telemetry["status"] = "updated"
        telemetry["skip_reason"] = None
        state["_cart_roi_last_segmenter"] = telemetry
        state["_cart_roi_last_error"] = None

        if self.debug:
            self._debug_log_and_save(
                frame_bgr=frame_bgr,
                roi_mask=roi_mask,
                frame_id=frame_id,
                session_id=session_id,
                state=state,
            )
        return roi_mask

    def _request_and_decode(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {
            "semantic_mask": None,
            "class_map": {},
            "response_ok": False,
            "latency_ms": None,
            "error_reason": None,
        }
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            logger.warning("Cart ROI segmentation skipped: failed to JPEG-encode frame")
            out["error_reason"] = "jpeg_encode_failed"
            return out

        url = f"{self._endpoint_url}?{urllib.parse.urlencode({'api_key': self.api_key})}"
        boundary = f"----RoboflowBoundary{int(time.time() * 1000)}"
        body = self._multipart_body(boundary=boundary, image_bytes=encoded.tobytes())
        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
            out["response_ok"] = True
            out["latency_ms"] = int((time.time() - started) * 1000.0)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                detail = ""
            logger.warning(
                "Cart ROI Roboflow HTTP error: code=%s reason=%s detail=%s",
                exc.code,
                exc.reason,
                detail,
            )
            out["error_reason"] = "http_error"
            return out
        except Exception as exc:
            logger.warning("Cart ROI Roboflow request failed: %s", exc)
            out["error_reason"] = "request_failed"
            out["latency_ms"] = int((time.time() - started) * 1000.0)
            return out

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            logger.warning("Cart ROI Roboflow JSON decode failed: %s", exc)
            out["error_reason"] = "json_decode_failed"
            return out

        semantic_mask = self._extract_semantic_mask(payload)
        class_map = self._extract_class_map(payload)
        if semantic_mask is None:
            logger.warning("Cart ROI semantic mask missing in Roboflow response")
            out["error_reason"] = "semantic_mask_missing"
            return out
        if not class_map:
            logger.warning("Cart ROI class_map missing in Roboflow response")
            out["error_reason"] = "class_map_missing"
            return out
        out["semantic_mask"] = semantic_mask
        out["class_map"] = class_map
        return out

    @staticmethod
    def _multipart_body(*, boundary: str, image_bytes: bytes) -> bytes:
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        return head + image_bytes + tail

    def _extract_semantic_mask(self, payload: Any) -> np.ndarray | None:
        if isinstance(payload, dict):
            for key in ("mask", "segmentation_mask", "semantic_mask", "prediction_mask"):
                if key in payload:
                    return self._decode_mask(payload.get(key))
            preds = payload.get("predictions")
            if isinstance(preds, dict):
                for key in ("mask", "segmentation_mask", "semantic_mask", "prediction_mask"):
                    if key in preds:
                        return self._decode_mask(preds.get(key))
            if isinstance(preds, list):
                for item in preds:
                    if not isinstance(item, dict):
                        continue
                    for key in ("mask", "segmentation_mask", "semantic_mask", "prediction_mask"):
                        if key in item:
                            return self._decode_mask(item.get(key))
        return None

    def _extract_class_map(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if isinstance(payload.get("class_map"), dict):
                return payload["class_map"]
            preds = payload.get("predictions")
            if isinstance(preds, dict) and isinstance(preds.get("class_map"), dict):
                return preds["class_map"]
        return {}

    def _decode_mask(self, raw_mask: Any) -> np.ndarray | None:
        if raw_mask is None:
            return None
        if isinstance(raw_mask, list):
            try:
                arr = np.array(raw_mask, dtype=np.int32)
                if arr.ndim == 2:
                    return arr
            except Exception:
                return None
            return None
        if isinstance(raw_mask, str):
            b64 = raw_mask
            if b64.startswith("data:image"):
                comma = b64.find(",")
                b64 = b64[comma + 1 :] if comma >= 0 else b64
            try:
                decoded = base64.b64decode(b64)
            except Exception:
                return None
            nparr = np.frombuffer(decoded, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img.astype(np.int32)
        return None

    def _build_cart_roi_mask(
        self,
        semantic_mask: np.ndarray,
        cart_id: int,
        frame_shape_hw: tuple[int, int],
    ) -> np.ndarray | None:
        mask = semantic_mask
        frame_h, frame_w = frame_shape_hw
        if mask.shape[:2] != (frame_h, frame_w):
            mask = cv2.resize(mask.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)

        roi = (mask == int(cart_id)).astype(np.uint8)
        if int(np.sum(roi)) <= 0:
            return None
        return roi

    @staticmethod
    def _parse_aliases(raw_aliases: str | list[str] | tuple[str, ...], target_class_name: str) -> set[str]:
        aliases: set[str] = set()
        if isinstance(raw_aliases, str):
            values = raw_aliases.split(",")
        else:
            values = list(raw_aliases)
        for name in values:
            norm = str(name or "").strip().lower()
            if norm:
                aliases.add(norm)
        target_norm = str(target_class_name or "").strip().lower()
        if target_norm:
            aliases.add(target_norm)
        return aliases

    @staticmethod
    def _class_name_candidates(class_map: dict[str, Any], limit: int = 8) -> list[str]:
        out: list[str] = []
        for k, v in class_map.items():
            key = str(k).strip().lower()
            val = str(v).strip().lower()
            if key and not key.isdigit():
                out.append(key)
            if val and not val.isdigit():
                out.append(val)
            if len(out) >= limit:
                break
        # preserve order, de-duplicate
        dedup: list[str] = []
        for item in out:
            if item not in dedup:
                dedup.append(item)
        return dedup[:limit]

    def _find_target_class_id(self, class_map: dict[str, Any]) -> tuple[int | None, str | None]:
        # Case 1: {"cartline": 1}
        for k, v in class_map.items():
            name = str(k).strip().lower()
            if name in self.class_aliases:
                try:
                    return int(v), name
                except Exception:
                    continue
        # Case 2: {"1": "cartline"} or {"0":"background","1":"cartline"}
        for k, v in class_map.items():
            name = str(v).strip().lower()
            if name in self.class_aliases:
                try:
                    return int(k), name
                except Exception:
                    continue
        return None, None

    @staticmethod
    def _postprocess_mask(mask01: np.ndarray) -> np.ndarray:
        bin255 = (mask01.astype(np.uint8) * 255).astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        closed = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        return (opened > 0).astype(np.uint8)

    def _debug_log_and_save(
        self,
        *,
        frame_bgr: np.ndarray,
        roi_mask: np.ndarray,
        frame_id: int | None,
        session_id: str | None,
        state: MutableMapping[str, Any],
    ) -> None:
        ratio = float(np.count_nonzero(roi_mask)) / float(max(1, roi_mask.size))
        logger.info(
            "Cart ROI debug: session=%s frame_id=%s roi_ratio=%.4f",
            session_id,
            frame_id,
            ratio,
        )

        now_ms = int(time.time() * 1000)
        last_save_ms = int(state.get("_cart_roi_last_debug_save_ms", 0))
        # Avoid writing debug images too often.
        if now_ms - last_save_ms < 5000:
            return
        state["_cart_roi_last_debug_save_ms"] = now_ms

        try:
            out_dir = Path(self.debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            overlay = frame_bgr.copy()
            overlay[roi_mask > 0] = (0.65 * overlay[roi_mask > 0] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
            safe_sid = str(session_id or "na")
            safe_fid = int(frame_id) if frame_id is not None else -1
            out_path = out_dir / f"cart_roi_s{safe_sid}_f{safe_fid}_{now_ms}.jpg"
            cv2.imwrite(str(out_path), overlay)
        except Exception:
            logger.exception("Failed to save cart ROI debug overlay")


def segment_single_image(image_path: str) -> np.ndarray | None:
    """Small helper for local smoke tests."""
    from backend import config

    if not config.ROBOFLOW_API_KEY:
        logger.warning("ROBOFLOW_API_KEY is empty; cart ROI smoke test disabled")
        return None
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Failed to read image for cart ROI smoke test: %s", image_path)
        return None
    segmenter = RoboflowCartSegmenter(
        api_key=config.ROBOFLOW_API_KEY,
        endpoint="smartcart-fd4z1",
        version=4,
        every_n_frames=1,
        debug=bool(config.CART_ROI_DEBUG),
        debug_dir=config.CART_ROI_DEBUG_DIR,
    )
    state: dict[str, Any] = {}
    return segmenter.get_or_update_mask(
        frame_bgr=image,
        frame_count=1,
        frame_id=1,
        session_id="smoke",
        state=state,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cart ROI semantic segmentation smoke runner")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save", default="", help="Optional output path for roi overlay image")
    args = parser.parse_args()

    mask = segment_single_image(args.image)
    if mask is None:
        print("cart_roi_mask: none")
        raise SystemExit(1)

    ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
    print(f"cart_roi_mask: shape={mask.shape} ratio={ratio:.4f}")

    if args.save:
        frame = cv2.imread(args.image)
        overlay = frame.copy()
        overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save, overlay)
        print(f"overlay_saved: {args.save}")
