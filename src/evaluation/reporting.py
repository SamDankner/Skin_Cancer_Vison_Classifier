"""Experiment comparison and compact machine/readable evaluation reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


COMPARISON_COLUMNS = [
    "run_name", "strategy", "backbone", "dataset_combination", "task", "input_mode",
    "image_resolution", "validation_macro_f1", "test_macro_f1", "balanced_accuracy",
    "roc_auc", "pr_auc", "sensitivity", "specificity", "training_time_seconds",
    "class_coverage_complete", "leakage_free", "status", "checkpoint",
]


def _first(record: Mapping, *paths, default=None):
    for path in paths:
        if path in record and record[path] is not None:
            return record[path]
        value = record
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return default


def normalize_run_record(record: Mapping) -> dict:
    """Normalize nested run JSON or flattened experiment-summary rows."""
    flat = dict(record)
    return {
        "run_name": _first(record, "run_name"),
        "strategy": _first(record, "config.strategy_name", "strategy", "strategy_name"),
        "backbone": _first(record, "config.backbone", "backbone"),
        "dataset_combination": _first(record, "config.dataset_combination", "dataset_combination", "config.datasets", "datasets"),
        "task": _first(record, "config.task", "task"),
        "input_mode": _first(record, "config.input_mode", "input_mode", default="full_image"),
        "image_resolution": _first(record, "config.image_size", "image_size", "image_resolution"),
        "validation_macro_f1": _first(record, "validation_metrics.macro_f1", "metrics.validation_macro_f1", "metrics.macro_f1", "validation_macro_f1", "metrics.macro_f1"),
        "test_macro_f1": _first(record, "test_metrics.macro_f1", "metrics.test_macro_f1", "test_macro_f1"),
        "balanced_accuracy": _first(record, "validation_metrics.balanced_accuracy", "metrics.balanced_accuracy", "balanced_accuracy"),
        "roc_auc": _first(record, "validation_metrics.roc_auc", "metrics.roc_auc", "roc_auc"),
        "pr_auc": _first(record, "validation_metrics.pr_auc", "metrics.pr_auc", "pr_auc"),
        "sensitivity": _first(record, "validation_metrics.sensitivity", "metrics.sensitivity", "sensitivity"),
        "specificity": _first(record, "validation_metrics.specificity", "metrics.specificity", "specificity"),
        "training_time_seconds": _first(record, "timing.training_seconds", "training_time_seconds"),
        "class_coverage_complete": bool(_first(record, "validation_metrics.class_coverage_complete", "metrics.class_coverage_complete", "class_coverage_complete", default=False)),
        "leakage_free": bool(_first(record, "leakage_free", default=False)),
        "status": _first(record, "status", default="complete"),
        "checkpoint": _first(record, "best_checkpoint", "config.checkpoint_path", "checkpoint"),
        "source_record": flat,
    }


def load_experiment_records(
    summary_path: str | Path = "results/experiment_summary.csv",
    runs_root: str | Path = "results/runs",
) -> pd.DataFrame:
    """Read persisted runs, preferring detailed run.json records by run name."""
    records = {}
    summary = Path(summary_path)
    if summary.is_file() and summary.stat().st_size:
        for row in pd.read_csv(summary).replace({np.nan: None}).to_dict("records"):
            normalized = normalize_run_record(row)
            records[normalized["run_name"] or f"summary-{len(records)}"] = normalized
    root = Path(runs_root)
    if root.exists():
        for path in root.glob("*/run.json"):
            normalized = normalize_run_record(json.loads(path.read_text(encoding="utf-8")))
            records[normalized["run_name"] or path.parent.name] = normalized
    if not records:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return pd.DataFrame(records.values())


def compare_experiments(
    records: pd.DataFrame,
    *,
    task: str,
    filters: Mapping[str, object] | None = None,
    sort_by: str = "validation_macro_f1",
    ascending: bool = False,
    valid_only: bool = False,
) -> pd.DataFrame:
    """Compare only one compatible task category at a time."""
    if not task:
        raise ValueError("task is required; incompatible task families must not be ranked together")
    result = records.loc[records.task.eq(task)].copy()
    for column, value in (filters or {}).items():
        if column not in result:
            raise KeyError(column)
        values = value if isinstance(value, (list, tuple, set)) else [value]
        result = result.loc[result[column].isin(values)]
    if valid_only:
        result = result.loc[result.class_coverage_complete & result.leakage_free & result.status.eq("complete")]
    if sort_by not in result:
        raise KeyError(sort_by)
    return result.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(drop=True)


def select_development_models(records: pd.DataFrame, *, task: str, count: int = 1, sort_by: str = "validation_macro_f1") -> pd.DataFrame:
    """Select complete, leakage-free, class-complete models using development records."""
    ranked = compare_experiments(records, task=task, sort_by=sort_by, valid_only=True)
    ranked = ranked.loc[ranked.checkpoint.notna()]
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} valid {task!r} candidate runs are available")
    return ranked.head(count).copy()


def save_comparison(frame: pd.DataFrame, output_directory: str | Path) -> dict[str, Path]:
    """Save an experiment comparison as CSV and JSON."""
    destination = Path(output_directory); destination.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = destination / "experiment_comparison.csv", destination / "experiment_comparison.json"
    export = frame.drop(columns=["source_record"], errors="ignore")
    export.to_csv(csv_path, index=False)
    json_path.write_text(export.to_json(orient="records", indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def compare_crop_strategies(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    class_order: Sequence[str],
    crop_provenance: Mapping[str, str],
) -> pd.DataFrame:
    """Compare full/crop/fusion predictions on their identical sample intersection."""
    from .metrics import classification_metrics

    if set(prediction_frames) != set(crop_provenance):
        raise ValueError("Every strategy needs explicit crop provenance: none, automatic, or ground_truth_oracle")
    if not prediction_frames:
        raise ValueError("At least one crop strategy is required")
    allowed = {"none", "automatic", "ground_truth_oracle"}
    if set(crop_provenance.values()) - allowed:
        raise ValueError(f"crop_provenance values must be in {sorted(allowed)}")
    common = set.intersection(*(set(frame.sample_id.astype(str)) for frame in prediction_frames.values()))
    if not common:
        raise ValueError("Crop strategies have no common eligible samples")
    rows = []
    reference_labels, reference_task = None, None
    for name, original in prediction_frames.items():
        frame = original.assign(sample_id=original.sample_id.astype(str)).set_index("sample_id").loc[sorted(common)]
        if frame.task.nunique() != 1:
            raise ValueError(f"{name} mixes incompatible tasks")
        task = frame.task.iloc[0]
        if reference_task is not None and task != reference_task:
            raise ValueError("Crop strategies must use the same task")
        reference_task = task
        labels = frame.true_label.astype(str).to_numpy()
        if reference_labels is not None and not np.array_equal(labels, reference_labels):
            raise ValueError("True labels differ across crop strategies on common samples")
        reference_labels = labels
        probabilities = np.asarray([[json.loads(value)[label] for label in class_order] for value in frame.class_probabilities])
        targets = np.asarray([class_order.index(value) for value in labels])
        metrics = classification_metrics(targets, probabilities=probabilities, labels=range(len(class_order)), class_names=class_order)
        rows.append({"strategy": name, "crop_provenance": crop_provenance[name], "common_sample_count": len(common), **{key: metrics[key] for key in ("macro_f1", "balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity")}})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(matrix, class_names: Sequence[str], path: str | Path) -> Path:
    """Save a labelled confusion-matrix figure and close it."""
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="True")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.tight_layout()
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150); plt.close(figure)
    return destination


def plot_training_history(history: pd.DataFrame, path: str | Path) -> Path:
    """Save loss, validation performance, and learning-rate histories."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for prefix in ("train", "validation"):
        if f"{prefix}_loss" in history:
            axes[0].plot(history.epoch, history[f"{prefix}_loss"], label=prefix)
        if f"{prefix}_macro_f1" in history:
            axes[1].plot(history.epoch, history[f"{prefix}_macro_f1"], label=prefix)
    if "learning_rate" in history:
        axes[2].plot(history.epoch, history.learning_rate, label="learning rate", color="tab:green")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="Training and validation performance")
    axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="Learning-rate schedule")
    for axis in axes:
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    figure.tight_layout()
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150); plt.close(figure)
    return destination


def plot_parameter_search(frame: pd.DataFrame, path: str | Path, metric: str) -> Path:
    """Save validation performance and runtime by staged-search candidate."""
    if frame.empty or "validation_metric" not in frame:
        raise ValueError("Parameter-search results are empty or incomplete")
    labels = frame["run_name"].astype(str).tolist()
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.2), 4))
    axes[0].bar(positions, frame["validation_metric"].astype(float))
    axes[0].set(title=f"Validation {metric} by configuration", xlabel="Configuration", ylabel=metric)
    if "training_seconds" in frame:
        axes[1].bar(positions, frame["training_seconds"].astype(float), color="tab:orange")
    axes[1].set(title="Runtime by configuration", xlabel="Configuration", ylabel="Training seconds")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=45, ha="right")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_binary_curves(targets, positive_probabilities, path: str | Path, *, calibration_bins: int = 10) -> Path:
    """Save ROC, precision-recall, and reliability panels for a binary result."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    target, scores = np.asarray(targets), np.asarray(positive_probabilities, dtype=float)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Binary curves require both classes")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    RocCurveDisplay.from_predictions(target, scores, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(target, scores, ax=axes[1])
    observed, predicted = calibration_curve(target, scores, n_bins=calibration_bins, strategy="uniform")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[2].plot(predicted, observed, marker="o")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="Reliability")
    figure.tight_layout()
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150); plt.close(figure)
    return destination


def plot_localization_overlay(image: str | Path | Image.Image, path: str | Path, *, predicted_box=None, target_box=None) -> Path:
    """Save a clearly labelled automatic-versus-ground-truth box overlay."""
    from src.strategies.localization import parse_bounding_box

    source = Image.open(image) if not isinstance(image, Image.Image) else image
    try:
        figure, axis = plt.subplots(figsize=(6, 6)); axis.imshow(source.convert("RGB")); axis.axis("off")
        for box, color, label in ((target_box, "lime", "ground truth"), (predicted_box, "red", "automatic prediction")):
            parsed = parse_bounding_box(box)
            if parsed is not None:
                x1, y1, x2, y2 = parsed
                axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2, label=label))
        if axis.patches:
            axis.legend()
        figure.tight_layout()
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150); plt.close(figure)
        return destination
    finally:
        if not isinstance(image, Image.Image):
            source.close()
