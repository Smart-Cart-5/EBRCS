"""Minimal Streamlit shim so checkout_core can be imported without Streamlit.

checkout_core/inference.py uses:
  - @st.cache_resource(show_spinner=False)
  - @st.cache_data(show_spinner=False)
  - st.secrets (fallback after os.getenv)

This module installs a lightweight stub into sys.modules["streamlit"]
BEFORE checkout_core is imported. The stub makes decorators pass-through
and st.secrets return empty, so os.getenv path is used for HF_TOKEN.

Usage:
    import backend.st_shim  # noqa: F401  -- must be first
    from checkout_core.inference import load_models, load_db, ...
"""

from __future__ import annotations

import sys
import types


def _noop_decorator(show_spinner: bool = True):
    """Return a pass-through decorator (replaces cache_resource / cache_data)."""
    def wrapper(fn):
        return fn
    return wrapper


def _install_shim() -> None:
    if "streamlit" in sys.modules:
        # If real Streamlit is already loaded, skip
        return

    st = types.ModuleType("streamlit")
    st.cache_resource = _noop_decorator
    st.cache_data = _noop_decorator

    # st.secrets behaves like an empty dict so get_hf_token() falls to os.getenv
    st.secrets = {}

    sys.modules["streamlit"] = st


_install_shim()
