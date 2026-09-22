"""Validation-only search for the separately persisted v2 deployment policy.

This script deliberately reads only the three row-aligned validation exports in
``results/final_selection``. It never discovers or opens external-test paths.
"""
from __future__ import annotations
import argparse, ast, json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ensemble.weighting import aggregate
from src.evaluation.metrics import classification_metrics

NAMES = ("convnext", "efficientnet", "multimodal")


def load_validation_predictions(root: Path) -> pd.DataFrame:
    frames = []
    for name in NAMES:
        path = root / "results" / "final_selection" / f"validation_predictions_{name}.csv"
        frame = pd.read_csv(path)
        if set(frame.split.astype(str)) != {"validation"} or set(frame.task.astype(str)) != {"diagnosis_binary"}:
            raise ValueError(f"{path} is not the expected validation diagnosis export")
        frame = frame[["sample_id", "true_label", "class_probabilities"]].copy()
        frame[name] = frame.class_probabilities.map(lambda value: float(json.loads(value)["malignant"]))
        frames.append(frame.drop(columns="class_probabilities"))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["sample_id", "true_label"], how="inner", validate="one_to_one")
    if len(merged) != len(frames[0]) or merged.sample_id.duplicated().any() or merged.true_label.nunique() != 2:
        raise ValueError("Validation prediction exports are not completely ID/label aligned")
    merged["target"] = merged.true_label.eq("malignant").astype(int)
    return merged


def performance_weights(frame: pd.DataFrame, names: tuple[str, ...]) -> dict[str, float]:
    scores = []
    for name in names:
        pred = (frame[name].to_numpy() >= .51).astype(int)
        scores.append(classification_metrics(frame.target, pred, frame[name].to_numpy(), labels=[0, 1])["balanced_accuracy"])
    scores = np.maximum(np.asarray(scores, dtype=float), 1e-6)
    scores /= scores.sum()
    return dict(zip(names, map(float, scores)))


def policy_score(frame: pd.DataFrame, names: tuple[str, ...], policy: dict, train: pd.DataFrame | None = None) -> np.ndarray:
    policy = dict(policy)
    if policy["strategy"] in {"static_validation_weighted", "closest_pair_consensus", "robust_mad_huber"}:
        policy["weights"] = performance_weights(train if train is not None else frame, names)
    return np.asarray([aggregate(dict(zip(names, row)), policy)[0] for row in frame.loc[:, names].to_numpy()], dtype=float)


def metric_row(target, scores, threshold=.51) -> dict:
    values = classification_metrics(target, (scores >= threshold).astype(int), np.c_[1 - scores, scores], labels=[0, 1], class_names=["benign", "malignant"])
    return {key: values[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier_score")}


def candidates(names: tuple[str, ...]) -> list[dict]:
    if len(names) == 2:
        return [{"strategy": "equal_probability_average"}, {"strategy": "static_validation_weighted"}, {"strategy": "learned_logistic_stacker"}]
    choices = [{"strategy": "equal_probability_average"}, {"strategy": "static_validation_weighted"}]
    for pair in (.05, .10, .15):
        for ratio in (2., 3., 4.):
            choices.append({"strategy": "closest_pair_consensus", "pair_max_distance": pair, "separation_ratio": ratio, "outlier_weight_multiplier": .20})
    for cutoff in (1., 1.5, 2.):
        choices.append({"strategy": "robust_mad_huber", "huber_cutoff": cutoff, "minimum_scale": .05})
    return choices + [{"strategy": "learned_logistic_stacker"}]


def evaluate(frame: pd.DataFrame, names: tuple[str, ...], policy: dict, seed: int) -> tuple[dict, list[dict]]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, frame.target)):
        if policy["strategy"] == "learned_logistic_stacker":
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            model.fit(frame.iloc[train_idx].loc[:, names], frame.target.iloc[train_idx])
            score = model.predict_proba(frame.iloc[test_idx].loc[:, names])[:, 1]
        else:
            score = policy_score(frame.iloc[test_idx], names, policy, frame.iloc[train_idx])
        rows.append({"fold": fold, **metric_row(frame.target.iloc[test_idx], score), "strategy": policy["strategy"], "parameters": json.dumps(policy, sort_keys=True)})
    result = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0] if key in {"accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc", "pr_auc", "brier_score"}}
    result.update({"strategy": policy["strategy"], "parameters": json.dumps(policy, sort_keys=True), "balanced_accuracy_std": float(np.nanstd([row["balanced_accuracy"] for row in rows], ddof=1))})
    return result, rows


def select(results: list[tuple[dict, dict]], *, require_disagreement_policy: bool = False) -> tuple[dict, dict]:
    # A one-standard-error simplicity rule prevents deploying a fragile winner.
    best = max(results, key=lambda item: item[0]["balanced_accuracy"])[0]
    within = [item for item in results if item[0]["balanced_accuracy"] >= best["balanced_accuracy"] - best["balanced_accuracy_std"]]
    if require_disagreement_policy:
        adaptive = [item for item in within if item[0]["strategy"] in {"closest_pair_consensus", "robust_mad_huber"}]
        if adaptive:
            # The adaptive contender is statistically tied under the predeclared
            # one-SE rule; choose the more interpretable consensus gate.
            return min(adaptive, key=lambda item: (0 if item[0]["strategy"] == "closest_pair_consensus" else 1, -item[0]["macro_f1"]))
    order = {"equal_probability_average": 0, "static_validation_weighted": 1, "closest_pair_consensus": 2, "robust_mad_huber": 3, "learned_logistic_stacker": 4}
    return min(within, key=lambda item: (order[item[0]["strategy"]], -item[0]["macro_f1"]))


def threshold_analysis(frame: pd.DataFrame, names: tuple[str, ...], policy: dict) -> tuple[float, dict, dict]:
    scores = policy_score(frame, names, policy)
    fixed = metric_row(frame.target, scores)
    options = [(metric_row(frame.target, scores, threshold), threshold) for threshold in np.arange(.30, .701, .01)]
    eligible = [item for item in options if item[0]["sensitivity"] >= fixed["sensitivity"] - .01 and item[0]["specificity"] >= fixed["specificity"] - .01]
    tuned, threshold = max(eligible, key=lambda item: (item[0]["balanced_accuracy"], item[0]["macro_f1"], -abs(item[1] - .51)))
    return (float(threshold) if tuned["balanced_accuracy"] > fixed["balanced_accuracy"] + .005 else .51), fixed, tuned


def write_config(root: Path, three: dict, two: dict, threshold: float, source: str) -> Path:
    frozen = yaml.safe_load((root / "configs" / "final_model.yaml").read_text(encoding="utf-8"))
    v1 = root / "configs" / "deployments" / "v1_frozen"; v2 = root / "configs" / "deployments" / "v2_adaptive"
    v1.mkdir(parents=True, exist_ok=True); v2.mkdir(parents=True, exist_ok=True)
    (v1 / "ensemble.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    (v1 / "manifest.yaml").write_text("variant: v1_frozen\nsource: configs/final_model.yaml\n", encoding="utf-8")
    config = {"version": 2, "status": "experimental", "identifier": "v2_adaptive", "task": "diagnosis_binary", "class_order": ["benign", "malignant"], "models": frozen["models"],
              "ensemble": {"member_names": list(NAMES), "no_metadata_policy": two, "metadata_available_policy": three},
              "threshold": {"threshold": threshold, "fit_split": "validation", "objective": "balanced_accuracy"},
              "selection": {"data": "row-aligned validation prediction exports only", "external_evaluation": "not performed", "report": source}}
    path = v2 / "ensemble.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (v2 / "manifest.yaml").write_text("variant: v2_adaptive\nstatus: experimental\nvalidation_only: true\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search v2 ensemble policies using validation predictions only.")
    parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    frame = load_validation_predictions(ROOT); output = ROOT / "results" / "ensemble_search" / "v2_adaptive"; output.mkdir(parents=True, exist_ok=True)
    all_results, folds, selected = {}, [], {}
    for label, names in (("metadata_available", NAMES), ("no_metadata", NAMES[:2])):
        evaluated = []
        for policy in candidates(names):
            summary, fold_rows = evaluate(frame, names, policy, args.seed); evaluated.append((summary, policy)); folds.extend([{**row, "mode": label} for row in fold_rows])
        summary, policy = select(evaluated, require_disagreement_policy=(label == "metadata_available")); policy = dict(policy)
        if policy["strategy"] != "equal_probability_average": policy["weights"] = performance_weights(frame, names)
        selected[label] = policy; all_results[label] = [item[0] | {"mode": label} for item in evaluated]
    threshold, fixed, tuned = threshold_analysis(frame, NAMES, selected["metadata_available"])
    pd.DataFrame(sum(all_results.values(), [])).to_csv(output / "strategy_comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "cross_validation_results.csv", index=False)
    pd.DataFrame(sum(all_results.values(), [])).to_csv(output / "hyperparameter_search.csv", index=False)
    config = write_config(ROOT, selected["metadata_available"], selected["no_metadata"], threshold, "results/ensemble_search/v2_adaptive/weighting_report.md")
    examples = []
    for values in ((.86, .92, .05), (.20, .40, .65)):
        probability, weights = aggregate(dict(zip(NAMES, values)), selected["metadata_available"]); examples.append({"probabilities": list(values), "weights": weights, "ensemble_probability": probability})
    report = ["# v2 adaptive ensemble weighting search", "", "Validation-only repeated stratified 5-fold CV (3 repeats, seed 42) evaluated fixed neural-network prediction rows. DDI/external-test files were not read.", "", "## Selection", f"Three-model policy: `{selected['metadata_available']}`.", f"Two-model policy: `{selected['no_metadata']}`.", f"Threshold fixed analysis at 0.51: {fixed}; tuned analysis: {tuned}; deployed threshold: {threshold}.", "", "## Candidate theory", "- Equal averaging: transparent baseline.", "- Static validation weighting: normalized per-model validation balanced accuracy.", "- Robust MAD/Huber: median-centered influence reduction for large deviations.", "- Closest-pair consensus: downweights one member only when a close pair and separation-ratio gates are both satisfied.", "- Logistic stacking was intentionally not deployed: it needs fitted coefficients and an out-of-fold serving contract; the simple policies were the production comparison.", "", "## Example behavior", *[json.dumps(item) for item in examples], "", "See strategy_comparison.csv and cross_validation_results.csv for every fixed-threshold fold result."]
    (output / "weighting_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {"selected": selected, "threshold": threshold, "fixed_051": fixed, "tuned": tuned, "config": str(config.relative_to(ROOT)), "examples": examples}
    (output / "selected_strategy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ROOT / "configs" / "deployments" / "v2_adaptive" / "weighting_search_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
