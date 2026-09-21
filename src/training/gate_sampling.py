"""Configurable, non-predictive sampling and confidence weights for the gate."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


DEFAULT_LABEL_WEIGHTS = {
    "strong": 1.0,
    "moderate": 0.75,
    "weak": 0.40,
    "auxiliary": 0.25,
    "ambiguous": 0.0,
}


def label_confidence_weights(strengths, configured: dict | None = None) -> np.ndarray:
    """Return configured per-label reliability weights; unknown/missing is weak."""
    values = {**DEFAULT_LABEL_WEIGHTS, **(configured or {})}
    return np.asarray(
        [float(values.get(str(value).lower(), values["weak"])) for value in strengths],
        dtype=float,
    )


def gate_sampling_weights(frame: pd.DataFrame, *, confidence_weights: dict | None = None,
                          hard_negative_fraction: float = .30,
                          source_balance: bool = True) -> np.ndarray:
    """Create replacement-sampling weights with balanced classes and negative subtypes.

    The desired negative mix is applied only when both subtypes exist. Within a
    class/subtype stratum, optional source balancing and label reliability shape
    exposure. These fields are never passed through the model input path.
    """
    if not 0 <= hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be between 0 and 1")
    if "lesion_present" not in frame:
        raise KeyError("Gate sampling requires lesion_present")
    labels = pd.to_numeric(frame.lesion_present, errors="raise").astype(int).to_numpy()
    if set(labels) - {0, 1}:
        raise ValueError("Gate sampling requires binary labels")
    subtype = frame.get("gate_negative_subtype", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().to_numpy()
    sources = frame.get("source_dataset", frame.get("dataset", pd.Series("<missing>", index=frame.index))).fillna("<missing>").astype(str).to_numpy()
    strengths = frame.get("gate_label_strength", pd.Series("weak", index=frame.index)).fillna("weak")
    confidence = label_confidence_weights(strengths, confidence_weights)
    weights = np.zeros(len(frame), dtype=float)
    positive = np.flatnonzero(labels == 1)
    healthy = np.flatnonzero((labels == 0) & (subtype == "healthy_no_visible_lesion"))
    hard = np.flatnonzero((labels == 0) & (subtype == "other_skin_condition"))
    fallback_negative = np.flatnonzero((labels == 0) & ~np.isin(np.arange(len(frame)), np.r_[healthy, hard]))
    strata = [(positive, .5)]
    available_negative = [(healthy, 1 - hard_negative_fraction), (hard, hard_negative_fraction), (fallback_negative, 0.0)]
    present = [(indices, share) for indices, share in available_negative if len(indices)]
    if present:
        total_share = sum(share for _, share in present)
        if total_share == 0:
            present = [(indices, 1 / len(present)) for indices, _ in present]
        else:
            present = [(indices, share / total_share) for indices, share in present]
        strata.extend((indices, .5 * share) for indices, share in present)
    if not len(positive) or not present:
        # Preserve usable behavior for a temporary/synthetic one-class fixture.
        strata = [(np.arange(len(frame)), 1.0)]
    for indices, budget in strata:
        local = confidence[indices].clip(min=0)
        if source_balance:
            counts = defaultdict(int)
            for source in sources[indices]:
                counts[source] += 1
            local = local * np.asarray([1 / counts[source] for source in sources[indices]])
        if not local.sum():
            local = np.ones(len(indices), dtype=float)
        weights[indices] = budget * local / local.sum()
    return weights


__all__ = ["DEFAULT_LABEL_WEIGHTS", "gate_sampling_weights", "label_confidence_weights"]
