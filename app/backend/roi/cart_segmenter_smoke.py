from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from backend.roi.cart_segmenter import segment_single_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for cart ROI segmentation")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--save", default="", help="Optional path to save overlay image")
    args = parser.parse_args()

    mask = segment_single_image(args.image)
    if mask is None:
        print("cart_roi_mask: none")
        return 1

    ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
    print(f"cart_roi_mask: shape={mask.shape} ratio={ratio:.4f}")

    if args.save:
        frame = cv2.imread(args.image)
        if frame is not None:
            overlay = frame.copy()
            overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.save, overlay)
            print(f"overlay_saved: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
