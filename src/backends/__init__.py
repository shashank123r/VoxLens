"""Backend factory."""

from __future__ import annotations

import os

from backends.base import InferenceBackend
from backends.local import LocalGemmaBackend
from backends.nim import NIMBackend


def create_backend() -> InferenceBackend:
    backend_name = os.environ.get("MODEL_BACKEND", "nim").lower().strip()
    if backend_name == "nim":
        return NIMBackend()
    if backend_name == "local":
        return LocalGemmaBackend()
    raise ValueError(f"Unknown MODEL_BACKEND={backend_name!r}; expected 'local' or 'nim'")
