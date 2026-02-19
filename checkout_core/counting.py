from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any


def ensure_last_seen_at_state(
    session_state: MutableMapping[str, Any],
    *,
    key: str = "last_seen_at",
    legacy_key: str = "last_seen",
) -> dict[str, float]:
    """Ensure timestamp-based dedup state exists.

    Legacy frame-based state is dropped because it cannot be safely converted
    to wall-clock timestamps.
    """
    current = session_state.get(key)
    if not isinstance(current, dict):
        session_state[key] = {}

    if legacy_key in session_state:
        session_state.pop(legacy_key, None)

    return session_state[key]


def should_count_product(
    last_seen_at: MutableMapping[str, float],
    name: str,
    *,
    now: float | None = None,
    cooldown_seconds: float,
) -> bool:
    """Return True if product can be counted now, otherwise False."""
    if now is None:
        now = time.time()

    cooldown = max(0.0, float(cooldown_seconds))
    last = float(last_seen_at.get(name, -1e9))
    if now - last >= cooldown:
        last_seen_at[name] = now
        return True

    return False


def should_count_track(
    counted_tracks: MutableMapping,
    track_id: int | str,
    product_name: str,
) -> bool:
    """
    Track ID 기반 중복 방지 (DeepSORT 사용 시)

    같은 Track ID는 한 번만 카운트됩니다.

    Args:
        counted_tracks: Track ID → 상품명 매핑 딕셔너리
        track_id: DeepSORT Track ID
        product_name: 인식된 상품명

    Returns:
        True if 새로운 Track (카운트 가능), False otherwise
    """
    if track_id in counted_tracks:
        # 이미 카운트된 Track
        return False

    # 새로운 Track 등록
    counted_tracks[track_id] = product_name
    return True
