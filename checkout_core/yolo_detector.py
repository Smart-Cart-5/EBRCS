"""YOLO object detector for hand and product detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO-based object detector for product and hand detection.

    Detects both products and hands in the frame, returning normalized bounding boxes.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.3, device: str = "cpu"):
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO .pt model file
            conf_threshold: Confidence threshold for detections (0-1)
            device: Device to run inference on ("cpu" or "cuda")
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

        # Move model to device
        self.model.to(device)

        logger.info(
            "YOLO detector initialized: model=%s, conf=%.2f, device=%s",
            model_path, conf_threshold, device
        )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLO detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of detections, each containing:
            - box: [x1, y1, x2, y2] normalized coordinates (0-1)
            - class: "product" or "hand"
            - confidence: detection confidence (0-1)

        Example:
            [
                {
                    "box": [0.2, 0.3, 0.6, 0.7],
                    "class": "product",
                    "confidence": 0.95
                },
                {
                    "box": [0.1, 0.1, 0.3, 0.4],
                    "class": "hand",
                    "confidence": 0.88
                }
            ]
        """
        h, w = frame.shape[:2]

        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )

        detections = []

        # Parse results
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Normalize coordinates to 0-1 range
                x1_norm = float(x1 / w)
                y1_norm = float(y1 / h)
                x2_norm = float(x2 / w)
                y2_norm = float(y2 / h)

                # Get class ID and confidence
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                # Map class ID to name (0: hand, 1: object/product)
                class_name = "hand" if cls_id == 0 else "product"

                detections.append({
                    "box": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "class": class_name,
                    "confidence": conf
                })

        return detections

    def extract_crop(self, frame: np.ndarray, box: list[float]) -> np.ndarray | None:
        """Extract a cropped region from the frame using normalized coordinates.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            box: [x1, y1, x2, y2] normalized coordinates (0-1)

        Returns:
            Cropped BGR image, or None if invalid
        """
        h, w = frame.shape[:2]

        x1, y1, x2, y2 = box

        # Convert to pixel coordinates
        x1_px = int(x1 * w)
        y1_px = int(y1 * h)
        x2_px = int(x2 * w)
        y2_px = int(y2 * h)

        # Clamp to image bounds
        x1_px = max(0, min(x1_px, w - 1))
        y1_px = max(0, min(y1_px, h - 1))
        x2_px = max(0, min(x2_px, w))
        y2_px = max(0, min(y2_px, h))

        # Check if valid crop
        if x2_px <= x1_px or y2_px <= y1_px:
            return None

        # Extract crop
        crop = frame[y1_px:y2_px, x1_px:x2_px]

        # Check minimum size (at least 20x20)
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        return crop
