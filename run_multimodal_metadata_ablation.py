"""Run the validation-only, fixed-seed multimodal metadata ablation.

This intentionally uses no development-test predictions.  It creates a fresh
subdirectory under ``results/ablations/multimodal_metadata`` and refuses to
reuse one, so it cannot overwrite a serious run.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.datasets import select_task_manifest, validate_manifest
from src.evaluation.metrics import classification_metrics
from src.strategies.multimodal_strategy import ablation_config, train
from src.utils.seed import seed_everything


VARIANTS = {
    "A_image_only": (),
    "B_core_metadata": ("age", "sex", "anatomical_site"),
    "C_core_plus_skin_tone": ("age", "sex", "anatomical_site", "skin_tone"),
}


def _metric_subset(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"sample_count": 0, "metrics": None}
    probability = np.asarray([json.loads(value)["malignant"] for value in frame.class_probabilities])
    target = frame.true_label.map({"benign": 0, "malignant": 1}).to_numpy()
    prediction = frame.predicted_label.map({"benign": 0, "malignant": 1}).to_numpy()
    return {"sample_count": int(len(frame)), "metrics": classification_metrics(target, prediction, np.column_stack((1 - probability, probability)), labels=[0, 1], class_names=["benign", "malignant"])}


def _availability_report(manifest: pd.DataFrame, validation_predictions: pd.DataFrame) -> dict:
    validation = manifest.loc[manifest.split.eq("validation")].copy()
    validation["skin_tone_present"] = validation.skin_tone.notna() & validation.skin_tone.astype(str).str.strip().ne("")
    merged = validation_predictions.merge(validation[["image_id", "dataset", "binary_target", "skin_tone_present"]], on="image_id", how="left", validate="one_to_one")
    availability = {
        "by_source": validation.groupby("dataset").skin_tone_present.agg(["size", "sum", "mean"]).rename(columns={"size": "examples", "sum": "available", "mean": "available_fraction"}).reset_index().to_dict("records"),
        "by_label": validation.groupby("binary_target").skin_tone_present.agg(["size", "sum", "mean"]).rename(columns={"size": "examples", "sum": "available", "mean": "available_fraction"}).reset_index().to_dict("records"),
        "by_split": manifest.assign(skin_tone_present=manifest.skin_tone.notna() & manifest.skin_tone.astype(str).str.strip().ne("")).groupby("split").skin_tone_present.agg(["size", "sum", "mean"]).rename(columns={"size": "examples", "sum": "available", "mean": "available_fraction"}).reset_index().to_dict("records"),
    }
    availability["association"] = {
        "source_equals_PAD_given_present": float(merged.loc[merged.skin_tone_present, "dataset"].eq("PAD-UFES-20").mean()) if merged.skin_tone_present.any() else None,
        "label_malignant_rate_present": float(merged.loc[merged.skin_tone_present, "binary_target"].mean()) if merged.skin_tone_present.any() else None,
        "label_malignant_rate_missing": float(merged.loc[~merged.skin_tone_present, "binary_target"].mean()) if (~merged.skin_tone_present).any() else None,
    }
    return availability, merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/development_manifest.csv")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--output", default="results/ablations/multimodal_metadata")
    parser.add_argument("--include-masked-skin-tone", action="store_true")
    parser.add_argument("--max-per-source-label-split", type=int, default=None,
                        help="Fixed deterministic screening cap per source/label/split stratum.")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild the JSON report from existing validation predictions without training.")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.report_only:
        raise FileExistsError(f"Refusing to overwrite ablation output: {root}")
    root.mkdir(parents=True, exist_ok=args.report_only)
    manifest = select_task_manifest(validate_manifest(pd.read_csv(args.manifest, low_memory=False)), "diagnosis_binary")
    if manifest.dataset.fillna("").str.upper().eq("DDI").any():
        raise PermissionError("DDI is prohibited from the ablation")
    sampling = None
    if args.max_per_source_label_split is not None:
        if args.max_per_source_label_split < 1:
            raise ValueError("--max-per-source-label-split must be positive")
        sampled = []
        for _, group in manifest.groupby(["split", "dataset", "binary_target"], dropna=False):
            sampled.append(group.sample(min(len(group), args.max_per_source_label_split), random_state=42))
        manifest = pd.concat(sampled, ignore_index=True)
        sampling = {"method": "fixed seed 42 sample within split/source/label strata", "max_per_source_label_split": args.max_per_source_label_split, "examples": int(len(manifest))}
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    base.update({"image_size": 196, "epochs": 5, "seed": 42, "evaluate_development_test": False, "models_root": str(root / "models"), "summary_path": str(root / "summary.csv")})
    variants = dict(VARIANTS)
    if args.include_masked_skin_tone:
        variants["D_skin_tone_architecture_all_missing"] = VARIANTS["C_core_plus_skin_tone"]
    results, skin_tone_audit = {}, None
    for name, fields in variants.items():
        working = manifest.copy()
        if name.startswith("D_"):
            working["skin_tone"] = pd.NA
        cfg = ablation_config(base, fields, image_only=not fields)
        cfg.update({"run_name": name, "run_directory": str(root / name)})
        if args.report_only:
            saved = json.loads((root / name / "run.json").read_text(encoding="utf-8"))
            result_metrics, result_timing, best_epoch = saved["metrics"], saved["timing"], saved["best_epoch"]
        else:
            seed_everything(42, deterministic=False)
            result = train(working, cfg, run_name=name)
            result_metrics, result_timing, best_epoch = result.metrics, result.timing, result.best_epoch
        predictions = pd.read_csv(root / name / "validation_predictions.csv")
        validation_rows = manifest.loc[manifest.split.eq("validation"), ["image_id", "dataset"]]
        source_merged = predictions.merge(validation_rows, on="image_id", how="left", validate="one_to_one")
        row = {"configuration": "image" if not fields else "image + " + " + ".join(fields), "metadata_fields": list(fields), "validation_metrics": result_metrics, "training_time_seconds": result_timing["training_seconds"], "best_epoch": best_epoch, "source_validation": {source: _metric_subset(group) for source, group in source_merged.groupby("dataset")}}
        if name == "C_core_plus_skin_tone":
            skin_tone_audit, merged = _availability_report(manifest, predictions)
            row["skin_tone_present_validation"] = _metric_subset(merged.loc[merged.skin_tone_present])
            row["skin_tone_missing_validation"] = _metric_subset(merged.loc[~merged.skin_tone_present])
            row["source_validation"] = {source: _metric_subset(group) for source, group in merged.groupby("dataset")}
            probabilities = np.asarray([json.loads(value)["malignant"] for value in merged.class_probabilities])
            row["mean_malignant_probability_present"] = float(probabilities[merged.skin_tone_present.to_numpy()].mean())
            row["mean_malignant_probability_missing"] = float(probabilities[~merged.skin_tone_present.to_numpy()].mean())
        results[name] = row
    report = {"purpose": "validation-only controlled metadata ablation; no development-test or DDI data used", "sampling": sampling, "controlled_setup": {key: base.get(key) for key in ("seed", "image_size", "epochs", "backbone", "optimizer", "loss", "scheduler", "augmentation", "early_stopping_patience", "early_stopping_min_delta")}, "variants": results, "skin_tone_availability_audit": skin_tone_audit}
    (root / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(root / "ablation_report.json")


if __name__ == "__main__":
    main()
