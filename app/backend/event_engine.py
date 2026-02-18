"""Hand-driven ADD event engine (MVP, single candidate)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class EventUpdate:
    state: str
    status: str
    add_confirmed: bool = False
    remove_confirmed: bool = False
    track_box: list[float] | None = None


class AddEventEngine:
    IDLE = "IDLE"
    GRASP = "GRASP"
    PLACE_CHECK = "PLACE_CHECK"
    IN_CART = "IN_CART"
    PICK_CHECK = "PICK_CHECK"
    REMOVE_CHECK = "REMOVE_CHECK"

    def __init__(
        self,
        t_grasp_min_frames: int = 4,
        t_place_stable_frames: int = 12,
        t_remove_confirm_frames: int = 45,
        roi_hysteresis_inset_ratio: float = 0.05,
        roi_hysteresis_outset_ratio: float = 0.05,
    ):
        self.t_grasp_min_frames = max(1, int(t_grasp_min_frames))
        self.t_place_stable_frames = max(1, int(t_place_stable_frames))
        self.t_remove_confirm_frames = max(1, int(t_remove_confirm_frames))
        self.roi_hysteresis_inset_ratio = max(0.0, float(roi_hysteresis_inset_ratio))
        self.roi_hysteresis_outset_ratio = max(0.0, float(roi_hysteresis_outset_ratio))
        self.state = self.IDLE
        self.grasp_frames = 0
        self.place_stable_frames = 0
        self.place_missing_frames = 0
        self.pick_frames = 0
        self.remove_confirm_frames = 0
        self.last_box: list[float] | None = None

    @staticmethod
    def _center(box: list[float]) -> tuple[float, float]:
        return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

    def _find_track_box(
        self,
        products: list[dict[str, Any]],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
    ) -> list[float] | None:
        if not products:
            return None
        if self.last_box is None:
            for product in products:
                box = product.get("box")
                if box and self._inside_roi(box, roi_poly, frame_shape):
                    return box
            return products[0].get("box")

        last_cx, last_cy = self._center(self.last_box)
        best = None
        best_dist = 1e9
        for product in products:
            box = product.get("box")
            if not box:
                continue
            cx, cy = self._center(box)
            dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
            if dist < best_dist:
                best = box
                best_dist = dist
        return best

    def _inside_roi(
        self,
        box: list[float],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
        *,
        mode: str = "normal",
    ) -> bool:
        if roi_poly is None:
            return True
        h, w = frame_shape[:2]
        poly = roi_poly.astype(np.float32)
        if mode in {"in", "out"} and len(poly) >= 3:
            center = np.mean(poly, axis=0)
            if mode == "in":
                ratio = max(0.0, 1.0 - self.roi_hysteresis_inset_ratio)
            else:
                ratio = 1.0 + self.roi_hysteresis_outset_ratio
            poly = (poly - center) * ratio + center
        cx = ((box[0] + box[2]) * 0.5) * w
        cy = ((box[1] + box[3]) * 0.5) * h
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

    def _visible_in_roi(
        self,
        products: list[dict[str, Any]],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
    ) -> bool:
        for product in products:
            box = product.get("box")
            if box and self._inside_roi(box, roi_poly, frame_shape, mode="in"):
                return True
        return False

    def update(
        self,
        *,
        best_pair: dict[str, Any] | None,
        products: list[dict[str, Any]],
        roi_poly: np.ndarray | None,
        frame_shape: tuple[int, ...],
    ) -> EventUpdate:
        paired_box = best_pair.get("product_box") if best_pair else None
        if paired_box is not None:
            self.last_box = paired_box

        if self.state == self.IDLE:
            if paired_box is not None:
                self.grasp_frames += 1
                if self.grasp_frames >= self.t_grasp_min_frames:
                    self.state = self.GRASP
                    return EventUpdate(self.state, "GRASP", track_box=self.last_box)
                return EventUpdate(self.state, "IDLE(손-상품 탐색)", track_box=self.last_box)
            self.grasp_frames = 0
            return EventUpdate(self.state, "IDLE")

        if self.state == self.GRASP:
            if paired_box is not None:
                self.grasp_frames += 1
                return EventUpdate(self.state, "GRASP", track_box=self.last_box)
            self.state = self.PLACE_CHECK
            self.place_stable_frames = 0
            self.place_missing_frames = 0
            return EventUpdate(self.state, "PLACE_CHECK", track_box=self.last_box)

        if self.state == self.PLACE_CHECK:
            track_box = self._find_track_box(products, roi_poly, frame_shape)
            if track_box is None:
                self.place_missing_frames += 1
                if self.place_missing_frames > self.t_place_stable_frames:
                    self.state = self.IDLE
                    self.grasp_frames = 0
                    self.last_box = None
                    return EventUpdate(self.state, "IDLE(후보 소실)")
                return EventUpdate(self.state, "PLACE_CHECK(대상 탐색)")

            self.last_box = track_box
            hand_separated = paired_box is None
            in_roi = self._inside_roi(track_box, roi_poly, frame_shape, mode="in")
            if hand_separated and in_roi:
                self.place_stable_frames += 1
            else:
                self.place_stable_frames = 0

            if self.place_stable_frames >= self.t_place_stable_frames:
                self.state = self.IN_CART
                self.grasp_frames = 0
                self.pick_frames = 0
                return EventUpdate(self.state, "ADD 확정", add_confirmed=True, track_box=track_box)

            return EventUpdate(self.state, "PLACE_CHECK", track_box=track_box)

        if self.state == self.IN_CART:
            if paired_box is not None and self._inside_roi(paired_box, roi_poly, frame_shape, mode="in"):
                self.pick_frames += 1
                self.last_box = paired_box
                if self.pick_frames >= self.t_grasp_min_frames:
                    self.state = self.PICK_CHECK
                    self.remove_confirm_frames = 0
                    return EventUpdate(self.state, "PICK_CHECK", track_box=self.last_box)
            else:
                self.pick_frames = 0
            return EventUpdate(self.state, "IN_CART", track_box=self.last_box)

        if self.state == self.PICK_CHECK:
            if paired_box is not None:
                self.last_box = paired_box
                outside_out = not self._inside_roi(paired_box, roi_poly, frame_shape, mode="out")
                if outside_out:
                    self.state = self.REMOVE_CHECK
                    self.remove_confirm_frames = 1
                    return EventUpdate(self.state, "REMOVE_CHECK", track_box=self.last_box)
                return EventUpdate(self.state, "PICK_CHECK", track_box=self.last_box)

            # Hand-product pair lost after pick; begin remove confirmation.
            self.state = self.REMOVE_CHECK
            self.remove_confirm_frames = 1
            return EventUpdate(self.state, "REMOVE_CHECK", track_box=self.last_box)

        # REMOVE_CHECK
        if paired_box is not None and self._inside_roi(paired_box, roi_poly, frame_shape, mode="in"):
            self.state = self.IN_CART
            self.pick_frames = 0
            self.remove_confirm_frames = 0
            return EventUpdate(self.state, "REMOVE 취소", track_box=paired_box)

        if self._visible_in_roi(products, roi_poly, frame_shape):
            self.state = self.IN_CART
            self.pick_frames = 0
            self.remove_confirm_frames = 0
            return EventUpdate(self.state, "REMOVE 취소", track_box=self.last_box)

        self.remove_confirm_frames += 1
        if self.remove_confirm_frames >= self.t_remove_confirm_frames:
            self.state = self.IN_CART
            self.pick_frames = 0
            self.remove_confirm_frames = 0
            return EventUpdate(self.state, "REMOVE 확정", remove_confirmed=True, track_box=self.last_box)

        return EventUpdate(self.state, "REMOVE_CHECK", track_box=self.last_box)
