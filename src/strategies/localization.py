"""Macro-photo lesion localization, crop generation, and localization metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


def parse_bounding_box(value) -> tuple[float, float, float, float] | None:
    """Parse an xyxy box from a manifest value without inferring missing labels.

    Accepted values are ``[x_min, y_min, x_max, y_max]`` or a mapping with those
    names. Coordinates are pixel coordinates in the uncropped source image.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        value = [value[key] for key in ("x_min", "y_min", "x_max", "y_max")]
    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("bounding_box must be xyxy [x_min, y_min, x_max, y_max]")
    x1, y1, x2, y2 = map(float, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding_box must have x_max > x_min and y_max > y_min")
    return x1, y1, x2, y2


def segmentation_bounding_box(mask: Image.Image | str | Path) -> tuple[int, int, int, int] | None:
    """Return the tight xyxy foreground box of a supplied macro-photo mask."""
    close = False
    if not isinstance(mask, Image.Image):
        mask, close = Image.open(mask), True
    try:
        array = np.asarray(mask.convert("L")) > 0
        ys, xs = np.where(array)
        return None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    finally:
        if close:
            mask.close()


def expand_box(box, image_size: tuple[int, int], margin: float = .15) -> tuple[int, int, int, int]:
    """Expand an xyxy box by a fractional side margin and clamp it to an image."""
    if not 0 <= margin <= 1:
        raise ValueError("margin must be between 0 and 1")
    x1, y1, x2, y2 = parse_bounding_box(box)
    width, height = image_size
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    expanded = (max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))), min(width, int(np.ceil(x2 + dx))), min(height, int(np.ceil(y2 + dy))))
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        raise ValueError("box is outside image boundaries")
    return expanded


def lesion_crop(image: Image.Image | str | Path, box, margin: float = .15) -> Image.Image:
    """Crop a valid localization box with context; caller owns the returned image."""
    close = False
    if not isinstance(image, Image.Image):
        image, close = Image.open(image), True
    try:
        return image.convert("RGB").crop(expand_box(box, image.size, margin))
    finally:
        if close:
            image.close()


def box_iou(predicted, target) -> float:
    """Compute IoU for two xyxy localization boxes."""
    px1, py1, px2, py2 = parse_bounding_box(predicted); tx1, ty1, tx2, ty2 = parse_bounding_box(target)
    intersection = max(0, min(px2, tx2) - max(px1, tx1)) * max(0, min(py2, ty2) - max(py1, ty1))
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - intersection
    return intersection / union if union else 0.0


def localization_metrics(predicted_boxes, target_boxes, iou_threshold: float = .5) -> dict:
    """Report box IoU and recall only for samples with real localization labels."""
    if len(predicted_boxes) != len(target_boxes):
        raise ValueError("predicted_boxes and target_boxes must have the same length")
    ious = [box_iou(predicted, target) for predicted, target in zip(predicted_boxes, target_boxes)]
    return {"support": len(ious), "mean_iou": float(np.mean(ious)) if ious else None, "localization_recall": float(np.mean(np.asarray(ious) >= iou_threshold)) if ious else None, "iou_threshold": iou_threshold}


def localization_annotation_requirements() -> str:
    """Describe the non-fabricated annotation contract expected by localization."""
    return ("Use ordinary clinical/macro-photo rows only. Provide either manifest bounding_box as pixel xyxy "
            "[x_min, y_min, x_max, y_max], or segmentation_mask_path to an aligned binary lesion mask. "
            "Missing annotations remain missing and are excluded from localization supervision.")
