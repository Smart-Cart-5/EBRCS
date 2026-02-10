"""Shared application state and dependency injection.

Holds the singleton model bundle, FAISS index, and session manager.
Populated during FastAPI lifespan startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
from aiorwlock import RWLock

from backend.services.session_manager import SessionManager


@dataclass
class AppState:
    """Global application state shared across all requests.

    Uses RWLock for concurrent read (inference) and exclusive write (product updates).
    - Reader lock: Multiple inference requests can read FAISS index simultaneously
    - Writer lock: Only one product update at a time, blocks all readers
    """

    model_bundle: dict[str, Any] = field(default_factory=dict)
    weighted_db: np.ndarray | None = None
    labels: np.ndarray | None = None
    faiss_index: faiss.IndexFlatIP | None = None

    # RWLock: allows multiple readers OR single writer
    index_rwlock: RWLock = field(default_factory=RWLock)

    session_manager: SessionManager = field(default_factory=SessionManager)


# Singleton -- populated in main.py lifespan
app_state = AppState()
