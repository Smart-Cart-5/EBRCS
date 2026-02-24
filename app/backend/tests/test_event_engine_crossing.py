from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.event_engine import AddEventEngine


def _box_inside() -> list[float]:
    return [0.45, 0.45, 0.55, 0.55]


def _box_outside() -> list[float]:
    return [0.05, 0.05, 0.15, 0.15]


def _roi_poly() -> np.ndarray:
    return np.array(
        [[30.0, 30.0], [70.0, 30.0], [70.0, 70.0], [30.0, 70.0]],
        dtype=np.float32,
    )


def _update(engine: AddEventEngine, frame_id: int, inside: bool):
    return engine.update(
        best_pair=None,
        products=[
            {
                "track_id": 12,
                "box": _box_inside() if inside else _box_outside(),
                "confidence": 0.91,
            }
        ],
        roi_poly=_roi_poly(),
        frame_shape=(100, 100, 3),
        frame_id=frame_id,
        session_id="s1",
    )


def test_add_only_on_outside_to_inside_crossing() -> None:
    engine = AddEventEngine(transition_confirm_frames=2, event_cooldown_sec=0.0)
    assert not _update(engine, 1, inside=False).add_confirmed
    assert not _update(engine, 2, inside=True).add_confirmed
    ev = _update(engine, 3, inside=True)
    assert ev.add_confirmed
    assert ev.event_payload is not None
    assert ev.event_payload["event_type"] == "ADD"
    assert ev.event_payload["prev_inside"] is False
    assert ev.event_payload["curr_inside"] is True
    assert ev.event_payload["track_id"] == 12


def test_remove_only_on_inside_to_outside_crossing() -> None:
    engine = AddEventEngine(transition_confirm_frames=2, event_cooldown_sec=0.0)
    assert not _update(engine, 1, inside=True).remove_confirmed
    assert not _update(engine, 2, inside=False).remove_confirmed
    ev = _update(engine, 3, inside=False)
    assert ev.remove_confirmed
    assert ev.event_payload is not None
    assert ev.event_payload["event_type"] == "REMOVE"
    assert ev.event_payload["prev_inside"] is True
    assert ev.event_payload["curr_inside"] is False


def test_no_add_when_track_starts_inside_and_stays_inside() -> None:
    engine = AddEventEngine(transition_confirm_frames=2, event_cooldown_sec=0.0)
    assert not _update(engine, 1, inside=True).add_confirmed
    assert not _update(engine, 2, inside=True).add_confirmed
    assert not _update(engine, 3, inside=True).add_confirmed


def test_cooldown_suppresses_rapid_flip_events() -> None:
    engine = AddEventEngine(transition_confirm_frames=1, event_cooldown_sec=1.0)
    with patch("backend.event_engine.time.time", side_effect=[1.0, 1.2, 2.5]):
        assert not _update(engine, 1, inside=False).add_confirmed
        ev_add = _update(engine, 2, inside=True)
        assert ev_add.add_confirmed
        ev_remove_suppressed = _update(engine, 3, inside=False)
        assert not ev_remove_suppressed.remove_confirmed
        ev_add_again = _update(engine, 4, inside=True)
        assert ev_add_again.add_confirmed
