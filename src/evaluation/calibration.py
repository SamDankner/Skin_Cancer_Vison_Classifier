"""Validation-only threshold selection and temperature scaling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from .metrics import classification_metrics


def _require_validation(split: str) -> None:
    if split != "validation":
        raise ValueError("Thresholds, calibration parameters, and ensemble weights may be fit only on the validation split")


def _binary_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def select_binary_threshold(
    targets,
    positive_probabilities,
    *,
    split: str,
    objective: str = "macro_f1",
    minimum_sensitivity: float | None = None,
    grid_size: int = 1001,
) -> dict:
    """Select a binary threshold on validation predictions only."""
    _require_validation(split)
    y_true, scores = np.asarray(targets, dtype=int), np.asarray(positive_probabilities, dtype=float)
    if y_true.shape != scores.shape or y_true.ndim != 1:
        raise ValueError("targets and positive_probabilities must be equal one-dimensional arrays")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold selection requires both binary classes on validation")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Positive probabilities must be finite values in [0, 1]")
    supported = {"macro_f1", "youden_j", "balanced_accuracy", "sensitivity_constrained"}
    if objective not in supported:
        raise ValueError(f"objective must be one of {sorted(supported)}")
    if objective == "sensitivity_constrained" and not 0 <= (minimum_sensitivity if minimum_sensitivity is not None else -1) <= 1:
        raise ValueError("minimum_sensitivity in [0, 1] is required for sensitivity-constrained selection")

    # Include observed scores and 0.5 so ties are resolved on attainable
    # decision boundaries while retaining the conventional default threshold.
    thresholds = np.unique(
        np.concatenate((np.linspace(0.0, 1.0, grid_size), scores, [0.5]))
    )
    candidates = []
    for threshold in thresholds:
        metrics = classification_metrics(
            y_true,
            _binary_predictions(scores, threshold),
            np.column_stack((1 - scores, scores)),
            labels=[0, 1],
        )
        if objective == "youden_j":
            value = metrics["sensitivity"] + metrics["specificity"] - 1
        elif objective == "sensitivity_constrained":
            value = metrics["specificity"] if metrics["sensitivity"] >= minimum_sensitivity else None
        else:
            value = metrics[objective]
        if value is not None:
            candidates.append((float(value), -abs(float(threshold) - 0.5), float(threshold), metrics))
    if not candidates:
        raise ValueError("No threshold satisfies the requested objective/constraint")
    best = max(candidates, key=lambda row: (row[0], row[1], -row[2]))
    return {
        "threshold": best[2],
        "objective": objective,
        "objective_value": best[0],
        "minimum_sensitivity": minimum_sensitivity,
        "fit_split": split,
        "metrics": best[3],
    }


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return np.log(clipped)


def apply_temperature(values, temperature: float, *, input_type: str = "logits") -> np.ndarray:
    """Apply a previously fitted temperature without modifying fitted state."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    array = np.asarray(values, dtype=float)
    logits = _probabilities_to_logits(array) if input_type == "probabilities" else array
    if input_type not in {"logits", "probabilities"} or logits.ndim != 2:
        raise ValueError("values must be a two-dimensional logits/probabilities array")
    # Subtracting the row maximum keeps the softmax numerically stable without
    # changing its probabilities.
    scaled = logits / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    return exponent / exponent.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class TemperatureCalibration:
    """Persist and apply one validation-fitted calibration temperature."""
    temperature: float
    class_order: tuple[str, ...]
    fit_split: str = "validation"
    method: str = "temperature_scaling"

    def apply(self, values, *, input_type: str = "logits") -> np.ndarray:
        return apply_temperature(values, self.temperature, input_type=input_type)

    def to_dict(self) -> dict:
        values = asdict(self)
        values["class_order"] = list(self.class_order)
        return values


def fit_temperature(
    logits,
    targets,
    *,
    class_order: Sequence[str],
    split: str,
    max_iter: int = 100,
) -> tuple[TemperatureCalibration, dict]:
    """Fit one positive temperature using validation logits and NLL."""
    _require_validation(split)
    values, y_true = np.asarray(logits, dtype=float), np.asarray(targets, dtype=int)
    if values.ndim != 2 or values.shape[0] != len(y_true) or values.shape[1] != len(class_order):
        raise ValueError("Logit shape must be [samples, len(class_order)] and match targets")
    if set(np.unique(y_true)) != set(range(len(class_order))):
        raise ValueError("Calibration validation data must contain every declared class")
    tensor_logits = torch.tensor(values, dtype=torch.float64)
    tensor_targets = torch.tensor(y_true, dtype=torch.long)
    log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.05,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(tensor_logits / log_temperature.exp(), tensor_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
    calibration = TemperatureCalibration(temperature, tuple(map(str, class_order)), split)
    before = apply_temperature(values, 1.0)
    after = calibration.apply(values)
    before_metrics = classification_metrics(y_true, probabilities=before, labels=list(range(len(class_order))), class_names=class_order)
    after_metrics = classification_metrics(y_true, probabilities=after, labels=list(range(len(class_order))), class_names=class_order)
    comparison = {
        "before": {**{key: before_metrics[key] for key in ("brier_score", "calibration_error")}, "reliability": reliability_table(y_true, before)},
        "after": {**{key: after_metrics[key] for key in ("brier_score", "calibration_error")}, "reliability": reliability_table(y_true, after)},
        "recommended": bool(after_metrics["brier_score"] <= before_metrics["brier_score"]),
    }
    return calibration, comparison


def reliability_table(targets, probabilities, *, n_bins: int = 10) -> list[dict]:
    """Return reliability-bin counts, confidence, and observed accuracy."""
    y_true, probability = np.asarray(targets), np.asarray(probabilities, dtype=float)
    confidence, predicted = probability.max(axis=1), probability.argmax(axis=1)
    rows, edges = [], np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        selected = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == n_bins - 1 else confidence < edges[index + 1])
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]), "count": int(selected.sum()),
            "mean_confidence": float(confidence[selected].mean()) if selected.any() else None,
            "accuracy": float((predicted[selected] == y_true[selected]).mean()) if selected.any() else None,
        })
    return rows
