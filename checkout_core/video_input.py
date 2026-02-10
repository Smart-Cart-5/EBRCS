from __future__ import annotations

import os
import tempfile
from pathlib import Path


def persist_uploaded_video(uploaded_file, prefix: str = "checkout_video_") -> str:
    suffix = Path(getattr(uploaded_file, "name", "video.mp4")).suffix or ".mp4"
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(uploaded_file.getbuffer())
    return path
