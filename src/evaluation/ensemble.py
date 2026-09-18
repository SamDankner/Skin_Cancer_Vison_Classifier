"""Leakage-safe probability averaging for task-compatible classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EnsembleMember:
    """Describe aligned probabilities from one task-compatible model."""
    name: str
    task: str
    class_order: tuple[str, ...]
    probabilities: np.ndarray
    sample_ids: tuple[str, ...] | None = None
    requires_metadata: bool = False


def _validate_members(members: Sequence[EnsembleMember]) -> tuple[int, int]:
    if not members:
        raise ValueError("At least one ensemble member is required")
    task, order = members[0].task, members[0].class_order
    sample_ids = members[0].sample_ids
    shape = np.asarray(members[0].probabilities).shape
    if len(shape) != 2 or shape[1] != len(order):
        raise ValueError("Each member needs [samples, classes] probabilities matching class_order")
    for member in members:
        if member.task != task:
            raise ValueError("Cannot ensemble models trained for incompatible tasks")
        if member.class_order != order:
            raise ValueError("Ensemble members must use exactly the same class order")
        if np.asarray(member.probabilities).shape != shape:
            raise ValueError("Ensemble members must predict the same samples and classes")
        if sample_ids is not None and member.sample_ids != sample_ids:
            raise ValueError("Ensemble member sample IDs are not identically ordered")
        values = np.asarray(member.probabilities, dtype=float)
        missing = np.isnan(values).all(axis=1)
        if (np.isnan(values).any(axis=1) & ~missing).any():
            raise ValueError("A member probability row must be entirely present or entirely NaN")
        available = values[~missing]
        if (available < 0).any() or not np.isfinite(available).all() or not np.allclose(available.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Member outputs must be finite non-negative probabilities summing to one")
    return shape


def normalize_weights(weights, member_count: int) -> np.ndarray:
    """Validate and normalize non-negative ensemble weights."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (member_count,) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, non-zero, and match the member count")
    return values / values.sum()


def average_probabilities(
    members: Sequence[EnsembleMember],
    weights=None,
    *,
    missing_member_policy: str = "renormalize_available",
) -> tuple[np.ndarray, dict]:
    """Average probabilities, explicitly renormalizing around ineligible members.

    A member marks an ineligible sample with an all-NaN probability row. This is
    useful when legitimately required metadata is unavailable; metadata is never
    fabricated. Any partially NaN row is rejected.
    """
    sample_count, _ = _validate_members(members)
    weight = normalize_weights(np.ones(len(members)) if weights is None else weights, len(members))
    stack = np.stack([np.asarray(member.probabilities, dtype=float) for member in members])
    row_missing = np.isnan(stack).all(axis=2)
    partially_missing = np.isnan(stack).any(axis=2) & ~row_missing
    if partially_missing.any():
        raise ValueError("A member probability row must be entirely present or entirely NaN")
    if missing_member_policy not in {"renormalize_available", "error"}:
        raise ValueError("missing_member_policy must be renormalize_available or error")
    if missing_member_policy == "error" and row_missing.any():
        raise ValueError("An ensemble member is ineligible for at least one sample")

    # Per-sample renormalization preserves a probability distribution when a
    # compatible member cannot use legitimately absent metadata.
    available_weight = (~row_missing) * weight[:, None]
    denominators = available_weight.sum(axis=0)
    if (denominators == 0).any():
        raise ValueError("No ensemble member can produce a valid prediction for at least one sample")
    combined = np.nansum(stack * available_weight[:, :, None], axis=0) / denominators[:, None]
    if not np.allclose(combined.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Member outputs are not valid normalized probabilities")
    details = {
        "member_names": [member.name for member in members],
        "weights": weight.tolist(),
        "task": members[0].task,
        "class_order": list(members[0].class_order),
        "sample_count": sample_count,
        "available_member_count": (~row_missing).sum(axis=0).tolist(),
        "missing_member_policy": missing_member_policy,
    }
    return combined, details


def equal_weight_ensemble(members: Sequence[EnsembleMember], **kwargs) -> tuple[np.ndarray, dict]:
    """Average compatible member probabilities with equal initial weights."""
    return average_probabilities(members, np.ones(len(members)), **kwargs)


def fit_validation_weights(
    members: Sequence[EnsembleMember],
    targets,
    *,
    split: str,
    iterations: int = 1000,
    learning_rate: float = 0.1,
    minimum_weight: float = 0.0,
) -> dict:
    """Fit non-negative normalized weights by validation cross-entropy only."""
    if split != "validation":
        raise ValueError("Ensemble weights may be fit only on the validation split")
    _, class_count = _validate_members(members)
    stack = np.stack([np.asarray(member.probabilities, dtype=float) for member in members])
    if np.isnan(stack).any():
        raise ValueError("Weight fitting needs validation predictions from every member for every sample")
    target = np.asarray(targets, dtype=int)
    if target.shape != (stack.shape[1],) or (target < 0).any() or (target >= class_count).any():
        raise ValueError("targets must be class indices aligned to validation predictions")
    weights = np.full(len(members), 1.0 / len(members))
    true_probabilities = stack[:, np.arange(len(target)), target]
    for iteration in range(iterations):
        mixture = np.clip(weights @ true_probabilities, 1e-12, 1.0)
        gradient = -np.mean(true_probabilities / mixture, axis=1)
        step = learning_rate / np.sqrt(iteration + 1)
        weights *= np.exp(np.clip(-step * (gradient - gradient.mean()), -50, 50))
        if minimum_weight:
            weights = np.maximum(weights, minimum_weight)
        weights /= weights.sum()
    combined, _ = average_probabilities(members, weights)
    nll = -float(np.mean(np.log(np.clip(combined[np.arange(len(target)), target], 1e-12, 1.0))))
    return {
        "method": "validation_nll_probability_averaging",
        "fit_split": split,
        "member_names": [member.name for member in members],
        "task": members[0].task,
        "class_order": list(members[0].class_order),
        "weights": weights.tolist(),
        "validation_nll": nll,
    }


def apply_fitted_ensemble(members: Sequence[EnsembleMember], fitted: dict, **kwargs) -> tuple[np.ndarray, dict]:
    """Apply persisted weights only when names, task, and class order still match."""
    if fitted.get("member_names") != [member.name for member in members]:
        raise ValueError("Persisted ensemble member names/order do not match")
    if fitted.get("task") != members[0].task or tuple(fitted.get("class_order", ())) != members[0].class_order:
        raise ValueError("Persisted ensemble task or class order does not match")
    return average_probabilities(members, fitted["weights"], **kwargs)
