"""Small, deterministic probability aggregation policies."""
from __future__ import annotations

from typing import Mapping
import numpy as np


def normalized(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / weights.sum()


def equal(probabilities: Mapping[str, float], _: Mapping | None = None) -> dict[str, float]:
    return dict(zip(probabilities, normalized(np.ones(len(probabilities)))))


def static(probabilities: Mapping[str, float], policy: Mapping) -> dict[str, float]:
    base = policy.get("weights", {})
    return dict(zip(probabilities, normalized([float(base.get(name, 1.0)) for name in probabilities])))


def closest_pair_consensus(probabilities: Mapping[str, float], policy: Mapping) -> dict[str, float]:
    """Downweight one member only for a close-pair / distant-third pattern.

    The two gates make ordinary spread retain static validation weights: the
    nearest pair must meet ``pair_max_distance`` and the third prediction must
    be at least ``separation_ratio`` times farther from that pair.
    """
    names, values = list(probabilities), np.asarray(list(probabilities.values()), dtype=float)
    if len(names) != 3:
        return static(probabilities, policy)
    pairs = [(abs(values[i] - values[j]), i, j) for i in range(3) for j in range(i + 1, 3)]
    pair_distance, first, second = min(pairs)
    outlier = next(index for index in range(3) if index not in {first, second})
    separation = min(abs(values[outlier] - values[first]), abs(values[outlier] - values[second]))
    base = np.asarray([float(policy.get("weights", {}).get(name, 1.0)) for name in names])
    if pair_distance <= float(policy.get("pair_max_distance", 0.10)) and separation >= float(policy.get("separation_ratio", 2.0)) * max(pair_distance, 1e-8):
        base[outlier] *= float(policy.get("outlier_weight_multiplier", 0.20))
    return dict(zip(names, normalized(base)))


def robust_mad(probabilities: Mapping[str, float], policy: Mapping) -> dict[str, float]:
    """Huber-like inverse deviation weights around the sample median."""
    names, values = list(probabilities), np.asarray(list(probabilities.values()), dtype=float)
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(policy.get("minimum_scale", 0.05)))
    z = np.abs(values - median) / scale
    cutoff = float(policy.get("huber_cutoff", 1.5))
    robust = np.where(z <= cutoff, 1.0, cutoff / np.maximum(z, 1e-12))
    base = np.asarray([float(policy.get("weights", {}).get(name, 1.0)) for name in names])
    return dict(zip(names, normalized(base * robust)))


def aggregate(probabilities: Mapping[str, float], policy: Mapping) -> tuple[float, dict[str, float]]:
    name = policy.get("strategy", "equal_probability_average")
    methods = {"equal_probability_average": equal, "static_validation_weighted": static,
               "closest_pair_consensus": closest_pair_consensus, "robust_mad_huber": robust_mad}
    if name not in methods:
        raise ValueError(f"Unsupported production aggregation strategy: {name}")
    weights = methods[name](probabilities, policy)
    return float(sum(probabilities[key] * weights[key] for key in probabilities)), weights
