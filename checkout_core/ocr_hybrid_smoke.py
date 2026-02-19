from __future__ import annotations

import numpy as np
import cv2

from checkout_core.frame_processor import run_ocr_hybrid


def main() -> None:
    img = np.full((320, 320, 3), 255, dtype=np.uint8)
    cv2.putText(img, "jjapa saeu hwangtae", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    result = run_ocr_hybrid(img, frame_id=0, session_id="smoke")
    print("OCR hybrid smoke result:", result)


if __name__ == "__main__":
    main()
