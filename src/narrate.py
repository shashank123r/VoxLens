"""Scene-change detection for Narrate Mode."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def decode_jpeg_b64(blob: str) -> np.ndarray:
    if blob.startswith("data:"):
        blob = blob.split(",", 1)[1]
    raw = base64.b64decode(blob)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.float32)


def structural_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM without scipy/skimage dependency."""
    if a.shape != b.shape:
        b_img = Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]))
        b = np.asarray(b_img, dtype=np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(num / den) if den else 1.0


class SceneChangeDetector:
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def should_narrate(self, image_blob: str) -> tuple[bool, float]:
        frame = decode_jpeg_b64(image_blob)
        if self._previous is None:
            self._previous = frame
            return False, 1.0
        score = structural_similarity(self._previous, frame)
        changed = score < self.threshold
        if changed:
            self._previous = frame
        return changed, score
