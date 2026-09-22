"""Render normalized attribution maps over unmodified display images."""
from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, opacity: float = 0.45) -> Image.Image:
    """Overlay a red-yellow heatmap on a detached RGB copy of the upload."""
    display = image.convert("RGB").copy()
    map_image = Image.fromarray(np.clip(heatmap * 255, 0, 255).astype(np.uint8)).resize(display.size, Image.Resampling.BILINEAR)
    values = np.asarray(map_image, dtype=np.float32) / 255.0
    # A compact, dependency-free warm palette: black -> red -> yellow.
    colors = np.stack((np.minimum(1.0, 2 * values), np.minimum(1.0, 2 * np.maximum(values - .5, 0)), np.zeros_like(values)), axis=-1)
    base = np.asarray(display, dtype=np.float32) / 255.0
    alpha = np.clip(values[..., None] * float(opacity), 0.0, 1.0)
    composite = base * (1 - alpha) + colors * alpha
    return Image.fromarray(np.clip(composite * 255, 0, 255).astype(np.uint8), "RGB")
