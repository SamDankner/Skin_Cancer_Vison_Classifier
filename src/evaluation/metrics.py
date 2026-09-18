"""Robust classification, calibration, and grouped confidence-interval metrics."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean_defined(values: Sequence[float | None], weights: Sequence[float] | None = None) -> float | None:
    valid = np.asarray([value is not None and np.isfinite(value) for value in values])
    if not valid.any():
        return None
    data = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    if weights is None:
        return float(np.nanmean(data))
    weight = np.asarray(weights, dtype=float)[valid]
    return float(np.average(data[valid], weights=weight)) if weight.sum() else None


def expected_calibration_error(targets, probabilities, n_bins: int = 10) -> float | None:
    """Return top-label ECE; empty bins do not affect the result."""
    y_true = np.asarray(targets)
    probability = np.asarray(probabilities, dtype=float)
    if not len(y_true) or probability.ndim != 2 or probability.shape[0] != len(y_true):
        return None
    confidence, predicted = probability.max(axis=1), probability.argmax(axis=1)
    edges, total = np.linspace(0.0, 1.0, n_bins + 1), 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (confidence <= upper if index == n_bins - 1 else confidence < upper)
        if selected.any():
            accuracy = float((predicted[selected] == y_true[selected]).mean())
            total += selected.mean() * abs(accuracy - float(confidence[selected].mean()))
    return float(total)


def _validate_probabilities(probabilities, sample_count: int, class_count: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1 and class_count == 2:
        values = np.column_stack((1.0 - values, values))
    if values.shape != (sample_count, class_count):
        raise ValueError(f"Expected probabilities with shape {(sample_count, class_count)}, got {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one")
    return values


def classification_metrics(
    targets,
    predictions=None,
    probabilities=None,
    *,
    labels: Sequence[Any] | None = None,
    class_names: Sequence[str] | None = None,
    loss: float | None = None,
    calibration_bins: int = 10,
) -> dict:
    """Calculate a common metric contract without inventing undefined values.

    ``labels`` fixes the expected class order. Missing expected classes are
    reported explicitly; metrics requiring full coverage remain ``None``.
    """
    y_true = np.asarray(targets)
    if y_true.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if labels is None:
        labels = sorted(np.unique(y_true).tolist(), key=str)
    labels = list(labels)
    if not labels:
        raise ValueError("At least one expected class is required")
    class_names = [str(label) for label in labels] if class_names is None else list(class_names)
    if len(class_names) != len(labels):
        raise ValueError("class_names and labels must have equal length")
    unknown = set(np.unique(y_true).tolist()) - set(labels)
    if unknown:
        raise ValueError(f"Targets contain classes outside the declared class order: {sorted(unknown, key=str)}")

    probability = None if probabilities is None else _validate_probabilities(probabilities, len(y_true), len(labels))
    if predictions is None:
        if probability is None:
            raise ValueError("Provide predictions or probabilities")
        y_pred = np.asarray([labels[index] for index in probability.argmax(axis=1)])
    else:
        y_pred = np.asarray(predictions)
    if y_pred.shape != y_true.shape:
        raise ValueError("targets and predictions must have equal shape")
    unknown_predictions = set(np.unique(y_pred).tolist()) - set(labels)
    if unknown_predictions:
        raise ValueError(f"Predictions contain classes outside the declared class order: {sorted(unknown_predictions, key=str)}")

    observed = set(np.unique(y_true).tolist())
    missing = [label for label in labels if label not in observed]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_rows, precision_values, recall_values, f1_values, supports = [], [], [], [], []
    for index, (label, name) in enumerate(zip(labels, class_names)):
        tp = float(matrix[index, index])
        support, predicted_count = int(matrix[index, :].sum()), int(matrix[:, index].sum())
        precision, recall = _safe_ratio(tp, predicted_count), _safe_ratio(tp, support)
        fp, fn = predicted_count - tp, support - tp
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
        label_value = label.item() if isinstance(label, np.generic) else label
        class_rows.append({"class": label_value, "class_name": str(name), "precision": precision, "recall": recall, "f1": f1, "support": support})
        precision_values.append(precision); recall_values.append(recall); f1_values.append(f1); supports.append(support)

    result = {
        "loss": None if loss is None else float(loss),
        "sample_count": int(len(y_true)),
        "labels": [label.item() if isinstance(label, np.generic) else label for label in labels],
        "class_names": list(map(str, class_names)),
        "class_coverage_complete": not missing,
        "missing_classes": [label.item() if isinstance(label, np.generic) else label for label in missing],
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": _mean_defined(recall_values) if len(observed) > 1 else None,
        "macro_precision": _mean_defined(precision_values),
        "macro_recall": _mean_defined(recall_values),
        "macro_f1": _mean_defined(f1_values),
        "weighted_f1": _mean_defined(f1_values, supports),
        "confusion_matrix": matrix.tolist(),
        "class_metrics": class_rows,
        "roc_auc": None, "pr_auc": None,
        "sensitivity": None, "specificity": None,
        "positive_predictive_value": None, "negative_predictive_value": None,
        "brier_score": None, "calibration_error": None,
    }
    if len(labels) == 2:
        tn, fp, fn, tp = matrix.ravel()
        result.update({
            "sensitivity": _safe_ratio(tp, tp + fn),
            "specificity": _safe_ratio(tn, tn + fp),
            "positive_predictive_value": _safe_ratio(tp, tp + fp),
            "negative_predictive_value": _safe_ratio(tn, tn + fn),
        })
        if probability is not None and not missing:
            positive = (y_true == labels[1]).astype(int)
            result["roc_auc"] = float(roc_auc_score(positive, probability[:, 1]))
            result["pr_auc"] = float(average_precision_score(positive, probability[:, 1]))
            result["brier_score"] = float(np.mean((probability[:, 1] - positive) ** 2))
    elif probability is not None and not missing:
        encoded = label_binarize(y_true, classes=labels)
        result["roc_auc"] = float(roc_auc_score(encoded, probability, average="macro", multi_class="ovr"))
        result["pr_auc"] = float(average_precision_score(encoded, probability, average="macro"))
        indices = np.asarray([labels.index(value) for value in y_true])
        result["brier_score"] = float(np.mean(np.sum((probability - np.eye(len(labels))[indices]) ** 2, axis=1)))
    if probability is not None:
        encoded_targets = np.asarray([labels.index(value) for value in y_true])
        result["calibration_error"] = expected_calibration_error(encoded_targets, probability, calibration_bins)
    return result


def lesion_presence_metrics(targets, predictions=None, probabilities=None, **kwargs) -> dict:
    """Metrics for lesion=1 versus true normal skin=0; both classes are explicit."""
    kwargs.setdefault("labels", [0, 1])
    kwargs.setdefault("class_names", ["normal_skin", "lesion_present"])
    result = classification_metrics(targets, predictions, probabilities, **kwargs)
    result.update({
        "lesion_sensitivity": result["sensitivity"],
        "normal_skin_specificity": result["specificity"],
        "normal_skin_false_positive_rate": None if result["specificity"] is None else 1.0 - result["specificity"],
        "precision": result["positive_predictive_value"],
        "recall": result["sensitivity"],
        "f1": result["class_metrics"][1]["f1"],
    })
    return result


def source_stratified_metrics(targets, predictions, probabilities, sources, *, metric_function=classification_metrics, **kwargs) -> dict:
    """Compute the identical metric contract per declared source dataset."""
    y_true, y_pred, source = np.asarray(targets), np.asarray(predictions), np.asarray(sources, dtype=str)
    probability = None if probabilities is None else np.asarray(probabilities)
    if len(source) != len(y_true):
        raise ValueError("sources and targets must have equal length")
    return {
        name: metric_function(y_true[index], y_pred[index], None if probability is None else probability[index], **kwargs)
        for name in sorted(set(source))
        for index in [np.flatnonzero(source == name)]
    }


def bootstrap_confidence_intervals(
    targets,
    predictions,
    probabilities=None,
    *,
    labels: Sequence[Any] | None = None,
    groups=None,
    metrics: Sequence[str] = ("roc_auc", "sensitivity", "specificity", "macro_f1"),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    metric_function: Callable[..., dict] = classification_metrics,
) -> dict:
    """Bootstrap percentile intervals, resampling patient/group IDs when supplied."""
    y_true, y_pred = np.asarray(targets), np.asarray(predictions)
    probability = None if probabilities is None else np.asarray(probabilities)
    group_values = np.arange(len(y_true)) if groups is None else np.asarray(["<MISSING>" if value is None else str(value) for value in groups])
    if len(group_values) != len(y_true):
        raise ValueError("groups and targets must have equal length")
    unique_groups, rng = np.unique(group_values), np.random.default_rng(seed)
    samples = {name: [] for name in metrics}
    for _ in range(n_resamples):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(group_values == group) for group in chosen])
        values = metric_function(y_true[indices], y_pred[indices], None if probability is None else probability[indices], labels=labels)
        for name in metrics:
            value = values.get(name)
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    return {
        name: ({"lower": float(np.quantile(values, alpha)), "upper": float(np.quantile(values, 1 - alpha)), "confidence": confidence, "valid_resamples": len(values)} if values else None)
        for name, values in samples.items()
    }
