"""Validation-only selection helpers for the final diagnostic system."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json

import numpy as np
import pandas as pd

from .metrics import classification_metrics


def stable_sample_ids(metadata: Sequence[Mapping]) -> tuple[str, ...]:
    """Return deterministic IDs, rejecting rows that cannot be safely aligned."""
    ids = tuple(str(row.get("image_id") or row.get("image_path") or "") for row in metadata)
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Predictions need unique non-empty image_id or image_path values for alignment")
    return ids


def align_prediction_collections(collections: Mapping[str, Mapping]) -> dict[str, dict]:
    """Align prediction collections by stable ID and verify labels exactly match."""
    if not collections:
        raise ValueError("At least one prediction collection is required")
    reference_name, reference = next(iter(collections.items()))
    reference_ids = stable_sample_ids(reference["metadata"])
    reference_targets = np.asarray(reference["targets"])
    aligned = {reference_name: dict(reference, sample_ids=reference_ids)}
    for name, collection in list(collections.items())[1:]:
        ids = stable_sample_ids(collection["metadata"])
        target = np.asarray(collection["targets"])
        if set(ids) != set(reference_ids):
            raise ValueError(f"Prediction sample IDs disagree between {reference_name} and {name}")
        lookup = {sample_id: index for index, sample_id in enumerate(ids)}
        order = np.asarray([lookup[sample_id] for sample_id in reference_ids])
        if not np.array_equal(target[order], reference_targets):
            raise ValueError(f"Prediction labels disagree between {reference_name} and {name}")
        aligned[name] = {
            **collection,
            "targets": target[order],
            "probabilities": np.asarray(collection["probabilities"])[order],
            "logits": np.asarray(collection.get("logits", []))[order] if len(collection.get("logits", [])) else collection.get("logits"),
            "metadata": [collection["metadata"][index] for index in order],
            "sample_ids": reference_ids,
        }
    return aligned


def binary_metrics(targets, malignant_probability, threshold: float) -> dict:
    """Use the shared metric contract at one explicit malignant threshold."""
    positive = np.asarray(malignant_probability, dtype=float)
    probabilities = np.column_stack((1.0 - positive, positive))
    prediction = (positive >= threshold).astype(int)
    return classification_metrics(targets, prediction, probabilities, labels=[0, 1], class_names=["benign", "malignant"])


def threshold_search(targets, malignant_probability, *, split: str, thresholds=None) -> list[dict]:
    """Evaluate a predeclared threshold grid; fitting is forbidden outside validation."""
    if split != "validation":
        raise ValueError("Threshold selection is permitted only on the validation split")
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if thresholds is None else np.asarray(thresholds, dtype=float)
    if not len(grid) or (grid <= 0).any() or (grid >= 1).any():
        raise ValueError("Thresholds must be between 0 and 1")
    rows = []
    for threshold in grid:
        metrics = binary_metrics(targets, malignant_probability, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def select_threshold(rows: Sequence[Mapping]) -> dict:
    """Select macro-F1 first, preferring 0.50 then the nearest stable threshold."""
    if not rows:
        raise ValueError("Threshold search produced no rows")
    best = max(float(row["macro_f1"]) for row in rows if row["macro_f1"] is not None)
    tied = [dict(row) for row in rows if abs(float(row["macro_f1"]) - best) < 1e-12]
    return min(tied, key=lambda row: (abs(float(row["threshold"]) - .5), float(row["threshold"])))


def prediction_relationships(aligned: Mapping[str, Mapping]) -> dict:
    """Describe diversity without using labels to alter any probability weights."""
    names = list(aligned)
    probabilities = np.column_stack([np.asarray(aligned[name]["probabilities"])[:, 1] for name in names])
    target = np.asarray(next(iter(aligned.values()))["targets"])
    output = {"models": names, "probability_correlation": pd.DataFrame(probabilities, columns=names).corr().to_dict(), "pairs": []}
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            a, b = probabilities[:, first] >= .5, probabilities[:, second] >= .5
            output["pairs"].append({
                "first": names[first], "second": names[second],
                "disagreement_rate": float((a != b).mean()),
                "first_correct_second_wrong": int(((a == target) & (b != target)).sum()),
                "second_correct_first_wrong": int(((b == target) & (a != target)).sum()),
            })
    return output


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
