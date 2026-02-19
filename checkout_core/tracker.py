"""
DeepSORT 기반 객체 추적 모듈 (경량 모드)

배경차분으로 탐지된 bbox를 프레임 간 연결하여 고유 Track ID를 부여합니다.
embedder=None으로 IOU만 사용하여 기존 인식 속도를 유지합니다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None


class ObjectTracker:
    """DeepSORT 래퍼 - IOU 기반 bbox 트래킹 + Track ID 관리"""

    def __init__(
        self,
        *,
        max_age: int = 30,
        n_init: int = 1,
        max_iou_distance: float = 0.7,
        embedder: str | None = None,
        half: bool = True,
        bgr: bool = True,
    ):
        if DeepSort is None:
            raise ImportError(
                "deep-sort-realtime not installed. "
                "Run: pip install deep-sort-realtime"
            )

        kwargs: dict[str, Any] = dict(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
        )
        self._use_embedder = embedder is not None
        if self._use_embedder:
            kwargs.update(embedder=embedder, half=half, bgr=bgr)
        else:
            # IOU-only 모드: dummy embedder 사용 시 embeds 직접 전달
            kwargs["embedder"] = None

        self._max_age = max_age
        self._n_init = n_init
        self._max_iou_distance = max_iou_distance
        self.tracker = DeepSort(**kwargs)

        # Track ID → 상품명 매핑
        self.track_labels: dict[int, str] = {}
        # 카운트 완료된 Track ID
        self.counted_tracks: set[int] = set()

    def update(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int]],
    ) -> list[dict[str, Any]]:
        """bbox 리스트를 받아 Track ID를 부여하고 반환합니다."""
        raw_detections = [
            ([x, y, w, h], 1.0, None)
            for x, y, w, h in detections
        ]

        if self._use_embedder:
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        else:
            # IOU-only: 단위 벡터 dummy embedding (0벡터는 정규화 시 NaN 발생)
            dummy_embeds = [np.ones(128, dtype=np.float32) for _ in raw_detections]
            tracks = self.tracker.update_tracks(raw_detections, embeds=dummy_embeds)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltwh = track.to_ltwh()
            results.append({
                "track_id": track.track_id,
                "bbox": (int(ltwh[0]), int(ltwh[1]), int(ltwh[2]), int(ltwh[3])),
            })
        return results

    def find_track_for_bbox(
        self,
        tracks: list[dict[str, Any]],
        target_bbox: tuple[int, int, int, int],
    ) -> int | None:
        """주어진 bbox와 가장 IOU가 높은 Track ID를 반환합니다."""
        if not tracks:
            return None

        best_iou = 0.0
        best_id = None
        tx, ty, tw, th = target_bbox

        for t in tracks:
            bx, by, bw, bh = t["bbox"]
            iou = self._compute_iou(
                (tx, ty, tx + tw, ty + th),
                (bx, by, bx + bw, by + bh),
            )
            if iou > best_iou:
                best_iou = iou
                best_id = t["track_id"]

        return best_id

    @staticmethod
    def _compute_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """두 bbox (x1,y1,x2,y2)의 IOU를 계산합니다."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    def should_count(self, track_id: int) -> bool:
        """이 Track이 아직 카운트되지 않았으면 True, 등록 후 반환."""
        if track_id in self.counted_tracks:
            return False
        self.counted_tracks.add(track_id)
        return True

    def register_label(self, track_id: int, label: str) -> None:
        self.track_labels[track_id] = label

    def get_label(self, track_id: int) -> str | None:
        return self.track_labels.get(track_id)

    def reset(self) -> None:
        self.tracker = DeepSort(
            max_age=self._max_age,
            n_init=self._n_init,
            max_iou_distance=self._max_iou_distance,
            embedder=None,
        )
        self.track_labels.clear()
        self.counted_tracks.clear()
