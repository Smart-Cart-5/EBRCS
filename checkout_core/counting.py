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
