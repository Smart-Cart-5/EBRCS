from __future__ import annotations

import unittest

import numpy as np

from backend.hand_event_engine import HandEventEngine


def _roi() -> np.ndarray:
    return np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)


def _frame_result(frame_id: int, hands: list[list[float]], objects: list[list[float]]) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp_ms": frame_id * 100,
        "hands": [{"box": b, "class": "hand", "confidence": 0.9} for b in hands],
        "objects": [{"box": b, "class": "product", "confidence": 0.9} for b in objects],
        "roi_poly": _roi(),
        "frame_shape": (100, 100, 3),
    }


class HandEventEngineTest(unittest.TestCase):
    def _engine(self) -> HandEventEngine:
        return HandEventEngine(
            association_stable_frames=1,
            association_min_score=0.01,
            association_min_iou=0.0,
            allow_det_fallback=True,
            det_fallback_min_score=0.005,
            add_evidence_frames=2,
            remove_evidence_frames=2,
            candidate_timeout_ms=1000,
            cooldown_ms=0,
            track_iou_match_threshold=0.0,
            hand_representative_point="center",
            object_representative_point="bottom_center",
        )

    def test_add_confirmed_with_late_binding_after_crossing(self) -> None:
        engine = self._engine()
        seq = [
            _frame_result(0, [[0.20, 0.35, 0.32, 0.55]], []),
            _frame_result(1, [[0.34, 0.35, 0.46, 0.55]], []),  # OUT_TO_IN, but no object yet
            _frame_result(2, [[0.36, 0.35, 0.48, 0.55]], [[0.42, 0.40, 0.57, 0.68]]),  # late bind
            _frame_result(3, [[0.37, 0.35, 0.49, 0.55]], [[0.43, 0.40, 0.58, 0.68]]),
        ]
        add_events: list[dict[str, object]] = []
        saw_late_bind = False
        for frame_result in seq:
            out = engine.update(frame_result)
            add_events.extend([e for e in out["events"] if e.get("event_type") == "ADD"])
            saw_late_bind = saw_late_bind or any(
                t.get("reason") == "late_bind_candidate" for t in out["transitions"]
            )
        self.assertTrue(saw_late_bind)
        self.assertEqual(len(add_events), 1)
        self.assertIsNotNone(add_events[0].get("object_track_id"))

    def test_add_confirmed_with_det_fallback_when_tracked_empty(self) -> None:
        engine = self._engine()
        seq = [
            _frame_result(0, [[0.20, 0.35, 0.32, 0.55]], []),
            _frame_result(1, [[0.34, 0.35, 0.46, 0.55]], []),  # OUT_TO_IN
            {
                **_frame_result(2, [[0.36, 0.35, 0.48, 0.55]], []),
                "objects_fallback": [{"box": [0.42, 0.40, 0.57, 0.68], "confidence": 0.5}],
            },
            {
                **_frame_result(3, [[0.37, 0.35, 0.49, 0.55]], []),
                "objects_fallback": [{"box": [0.43, 0.40, 0.58, 0.68], "confidence": 0.5}],
            },
        ]
        add_events: list[dict[str, object]] = []
        used_fallback = False
        for frame_result in seq:
            out = engine.update(frame_result)
            add_events.extend([e for e in out["events"] if e.get("event_type") == "ADD"])
            for assoc in out.get("associations", []):
                counts = assoc.get("counts", {})
                if int(counts.get("det_fallback_used", 0)) > 0:
                    used_fallback = True
        self.assertTrue(used_fallback)
        self.assertEqual(len(add_events), 1)

    def test_add_timeout_cancel_when_candidate_never_appears(self) -> None:
        engine = self._engine()
        seq = [
            _frame_result(0, [[0.18, 0.35, 0.30, 0.55]], []),
            _frame_result(1, [[0.34, 0.35, 0.46, 0.55]], []),  # OUT_TO_IN
            _frame_result(2, [[0.35, 0.35, 0.47, 0.55]], []),
            _frame_result(3, [[0.36, 0.35, 0.48, 0.55]], []),
            _frame_result(12, [[0.36, 0.35, 0.48, 0.55]], []),  # timeout
        ]
        timeout_reasons: list[str] = []
        for frame_result in seq:
            out = engine.update(frame_result)
            timeout_reasons.extend(
                [str(t.get("reason")) for t in out["transitions"] if t.get("to") == "IDLE"]
            )
        self.assertIn("timeout_no_candidate", timeout_reasons)

    def test_remove_confirmed_on_in_to_out_and_object_lost(self) -> None:
        engine = self._engine()
        seq = [
            _frame_result(0, [[0.40, 0.35, 0.52, 0.55]], [[0.42, 0.40, 0.58, 0.68]]),
            _frame_result(1, [[0.42, 0.35, 0.54, 0.55]], [[0.43, 0.40, 0.59, 0.68]]),
            _frame_result(2, [[0.72, 0.35, 0.84, 0.55]], []),  # IN_TO_OUT
            _frame_result(3, [[0.73, 0.35, 0.85, 0.55]], []),
        ]
        remove_events: list[dict[str, object]] = []
        for frame_result in seq:
            out = engine.update(frame_result)
            remove_events.extend([e for e in out["events"] if e.get("event_type") == "REMOVE"])
        self.assertEqual(len(remove_events), 1)
        self.assertIsNotNone(remove_events[0].get("object_track_id"))


if __name__ == "__main__":
    unittest.main()
