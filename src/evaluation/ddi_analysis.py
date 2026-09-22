"""Non-tuning utilities for post-hoc analysis of protected DDI predictions.

These functions consume already-exported predictions.  They deliberately do
not fit thresholds, calibrators, ensemble weights, or models.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import classification_metrics


def malignant_probability(frame: pd.DataFrame) -> pd.Series:
    """Extract the persisted malignant probability without changing it."""
    return frame["class_probabilities"].map(lambda value: float(json.loads(value)["malignant"]))


def canonical_error_table(
    ensemble: pd.DataFrame,
    members: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Align preserved prediction exports with official DDI metadata by file."""
    meta = metadata.rename(columns={"DDI_file": "sample_id", "DDI_ID": "ddi_id", "disease": "original_diagnosis"}).copy()
    result = ensemble.copy()
    result["ensemble_malignant_probability"] = malignant_probability(result)
    result = result.merge(meta[[column for column in ("sample_id", "ddi_id", "original_diagnosis", "skin_tone", "age", "sex", "anatomical_site") if column in meta]], on="sample_id", how="left", validate="one_to_one")
    for name, frame in members.items():
        if set(frame.sample_id) != set(result.sample_id):
            raise ValueError(f"{name} predictions are not aligned to ensemble predictions")
        probability = frame[["sample_id", "class_probabilities"]].copy()
        probability[f"{name}_malignant_probability"] = malignant_probability(probability)
        result = result.merge(probability[["sample_id", f"{name}_malignant_probability"]], on="sample_id", how="left", validate="one_to_one")
    result["true_binary"] = result.true_label.eq("malignant").astype(int)
    result["predicted_binary"] = result.predicted_label.eq("malignant").astype(int)
    result["correct"] = result.true_binary.eq(result.predicted_binary)
    result["error_type"] = np.select(
        [(result.true_binary == 0) & (result.predicted_binary == 0), (result.true_binary == 0) & (result.predicted_binary == 1),
         (result.true_binary == 1) & (result.predicted_binary == 0)], ["TN", "FP", "FN"], default="TP"
    )
    probability_columns = [f"{name}_malignant_probability" for name in members]
    result["models_predicting_malignant"] = (result[probability_columns] >= threshold).sum(axis=1)
    result["model_probability_spread"] = result[probability_columns].max(axis=1) - result[probability_columns].min(axis=1)
    result["model_probability_std"] = result[probability_columns].std(axis=1, ddof=0)
    result["distance_from_frozen_threshold"] = (result.ensemble_malignant_probability - threshold).abs()
    result["confidence"] = np.maximum(result.ensemble_malignant_probability, 1 - result.ensemble_malignant_probability)
    return result


def binary_subgroup_metrics(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return fixed-threshold subgroup metrics; undefined values remain null."""
    rows = []
    for value, subset in frame.groupby(group, dropna=False, sort=True):
        label = "<missing>" if pd.isna(value) else str(value)
        probability = subset.ensemble_malignant_probability.to_numpy()
        metric = classification_metrics(subset.true_binary, subset.predicted_binary, np.column_stack([1 - probability, probability]), labels=[0, 1], class_names=["benign", "malignant"])
        matrix = metric["confusion_matrix"]
        rows.append({group: label, "total_n": len(subset), "malignant_n": int(subset.true_binary.sum()), "benign_n": int((1 - subset.true_binary).sum()),
                     "tn": matrix[0][0], "fp": matrix[0][1], "fn": matrix[1][0], "tp": matrix[1][1],
                     **{key: metric[key] for key in ("sensitivity", "specificity", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc")},
                     "small_sample_flag": len(subset) < 20})
    return pd.DataFrame(rows)


def probability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize frozen ensemble probabilities by confusion-matrix outcome."""
    rows = []
    for error_type, subset in frame.groupby("error_type", sort=False):
        values = subset.ensemble_malignant_probability
        rows.append({"error_type": error_type, "n": len(values), "mean": values.mean(), "median": values.median(), "std": values.std(ddof=0),
                     "q1": values.quantile(.25), "q3": values.quantile(.75), "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def false_negative_diagnoses(frame: pd.DataFrame) -> pd.DataFrame:
    """Diagnosis-level fixed-system sensitivity, including diagnoses without FNs."""
    malignant = frame.loc[frame.true_binary.eq(1)]
    rows = []
    for diagnosis, subset in malignant.groupby("original_diagnosis", dropna=False, sort=True):
        detected = int(subset.predicted_binary.sum())
        rows.append({"original_diagnosis": "<missing>" if pd.isna(diagnosis) else str(diagnosis), "malignant_n": len(subset), "correctly_detected": detected,
                     "missed_fn": len(subset) - detected, "sensitivity": detected / len(subset), "mean_ensemble_malignant_probability": subset.ensemble_malignant_probability.mean(),
                     "median_ensemble_malignant_probability": subset.ensemble_malignant_probability.median(), "small_sample_flag": len(subset) < 10})
    return pd.DataFrame(rows)


def model_disagreement(frame: pd.DataFrame, members: Iterable[str], threshold: float) -> pd.DataFrame:
    """Per-sample member votes and whether averaging overruled a positive vote."""
    result = frame.copy()
    columns = [f"{name}_malignant_probability" for name in members]
    for name, column in zip(members, columns):
        result[f"{name}_predicted_malignant"] = result[column] >= threshold
    result["member_vote_pattern"] = result[[f"{name}_predicted_malignant" for name in members]].astype(int).astype(str).agg("|".join, axis=1)
    result["any_member_detected_malignant"] = result["models_predicting_malignant"] > 0
    result["ensemble_overruled_member_positive"] = result.predicted_binary.eq(0) & result.any_member_detected_malignant
    return result
