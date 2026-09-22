"""Reproduce final validation-only selection and frozen development evaluation.

DDI is intentionally not an input to this command.  Use the separately guarded
external-evaluation API only after reviewing the generated frozen record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.datasets import select_task_manifest, validate_manifest
from src.evaluation.ensemble import EnsembleMember, average_probabilities
from src.evaluation.evaluator import (build_evaluation_loader, collect_predictions,
    export_predictions, freeze_final_configuration, major_result_report,
    prediction_frame, load_checkpoint_bundle)
from src.evaluation.final_selection import (align_prediction_collections, binary_metrics,
    prediction_relationships, select_threshold, stable_sample_ids, threshold_search, write_json)


STRATEGIES = ("efficientnet", "convnext", "dinov2", "multimodal")


def serious_runs(root: Path) -> list[dict]:
    """Read completed diagnostic records, excluding screens and skin-tone defaults."""
    rows = []
    for path in root.rglob("run.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg, metrics = payload.get("config", {}), payload.get("metrics", {})
        strategy = cfg.get("strategy_name")
        checkpoint = cfg.get("checkpoint_path") or payload.get("best_checkpoint")
        if (payload.get("status") != "complete" or cfg.get("task") != "diagnosis_binary"
                or strategy not in STRATEGIES or not checkpoint or not Path(checkpoint).is_file()
                or "screening" in path.parts):
            continue
        # Skin tone remains ablation-only because it was observed to encode source.
        if strategy == "multimodal" and "skin_tone" in (cfg.get("metadata_fields") or []):
            continue
        rows.append({"run_json": str(path), "strategy": strategy, "checkpoint": checkpoint,
                     "run_id": cfg.get("run_name") or path.parent.name,
                     "best_epoch": payload.get("best_epoch"), "config": cfg, "metrics": metrics,
                     "timing": payload.get("timing") or {}})
    return rows


def row(record: dict) -> dict:
    m, c = record["metrics"], record["config"]
    class_metrics = m.get("class_metrics", [{}, {}])
    return {"run_id": record["run_id"], "architecture": record["strategy"], "checkpoint": record["checkpoint"],
            "best_epoch": record["best_epoch"], "validation_macro_f1": m.get("macro_f1"),
            "validation_balanced_accuracy": m.get("balanced_accuracy"), "validation_accuracy": m.get("accuracy"),
            "validation_roc_auc": m.get("roc_auc"), "validation_pr_auc": m.get("pr_auc"),
            "sensitivity": m.get("sensitivity"), "specificity": m.get("specificity"),
            "benign_f1": class_metrics[0].get("f1"), "malignant_f1": class_metrics[1].get("f1"),
            "brier_score": m.get("brier_score"), "calibration_error": m.get("calibration_error"),
            "inference_latency_ms": ((record.get("timing") or {}).get("inference_model_only_batch_size_1") or {}).get("mean_ms"),
            "training_seconds": (record.get("timing") or {}).get("training_seconds"),
            "unfreeze_last_blocks": c.get("unfreeze_last_blocks"), "metadata_fields": json.dumps(c.get("metadata_fields") or []),
            "historical_baseline": "20260921T060517413359Z" in record["checkpoint"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/development_manifest.csv")
    parser.add_argument("--results", default="results/final_selection")
    parser.add_argument("--evaluation-results", default="results/final_evaluation")
    parser.add_argument("--frozen-config", default="configs/final_model.yaml")
    args = parser.parse_args()
    out, evaluation = Path(args.results), Path(args.evaluation_results)
    out.mkdir(parents=True, exist_ok=False); evaluation.mkdir(parents=True, exist_ok=False)
    candidates = serious_runs(Path("results/runs"))
    if not candidates:
        raise RuntimeError("No serious diagnostic run records with available checkpoints were found")
    winner = {strategy: max((r for r in candidates if r["strategy"] == strategy), key=lambda r: r["metrics"]["macro_f1"]) for strategy in STRATEGIES}
    all_rows = pd.DataFrame([row(r) for r in candidates]).sort_values("validation_macro_f1", ascending=False)
    all_rows.to_csv(out / "model_leaderboard.csv", index=False)
    write_json(out / "model_leaderboard.json", {"selection_split": "validation", "candidates": all_rows.to_dict("records"), "winners": {key: row(value) for key, value in winner.items()}})
    manifest = select_task_manifest(validate_manifest(pd.read_csv(args.manifest)), "diagnosis_binary")
    collections, bundles = {}, {}
    for name, record in winner.items():
        bundle = load_checkpoint_bundle(record["checkpoint"], strategy=name)
        if bundle.task != "diagnosis_binary" or bundle.class_order != ["benign", "malignant"]:
            raise ValueError(f"{name} checkpoint has incompatible task or class order")
        bundles[name] = bundle
        loader = build_evaluation_loader(manifest, bundle, split="validation")
        collections[name] = collect_predictions(bundle.model, loader, class_order=bundle.class_order)
    aligned = align_prediction_collections(collections)
    targets = next(iter(aligned.values()))["targets"]
    for name, values in aligned.items():
        export_predictions(prediction_frame(targets, values["probabilities"], values["metadata"], class_order=["benign", "malignant"], task="diagnosis_binary", strategy=name, checkpoint_id=Path(winner[name]["checkpoint"]).stem, threshold=.5), out / f"validation_predictions_{name}.csv")
    definitions = {name: members for name, members in {
        "convnext_plus_efficientnet": ["convnext", "efficientnet"],
        "convnext_plus_multimodal": ["convnext", "multimodal"],
        "convnext_plus_efficientnet_plus_multimodal": ["convnext", "efficientnet", "multimodal"],
        "all_four": ["convnext", "efficientnet", "multimodal", "dinov2"],
    }.items()}
    comparison, combined = [], {}
    for name, values in aligned.items():
        comparison.append({"system": name, "members": name, "weights": [1.0], **binary_metrics(targets, values["probabilities"][:, 1], .5)})
    for name, member_names in definitions.items():
        members = [EnsembleMember(item, "diagnosis_binary", ("benign", "malignant"), aligned[item]["probabilities"], aligned[item]["sample_ids"], item == "multimodal") for item in member_names]
        probability, details = average_probabilities(members)
        combined[name] = (probability, details)
        comparison.append({"system": name, "members": "+".join(member_names), "weights": details["weights"], **binary_metrics(targets, probability[:, 1], .5)})
    comparison_frame = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    comparison_frame.to_csv(out / "ensemble_validation_results.csv", index=False)
    relationships = prediction_relationships(aligned)
    write_json(out / "ensemble_validation_results.json", {"selection_split": "validation", "comparison": comparison_frame.to_dict("records"), "relationships": relationships})
    chosen_name = comparison_frame.iloc[0]["system"]
    if chosen_name in aligned:
        final_probability = aligned[chosen_name]["probabilities"]
        final_members, final_weights = [chosen_name], [1.0]
    else:
        final_probability, details = combined[chosen_name]
        final_members, final_weights = definitions[chosen_name], details["weights"]
    search = threshold_search(targets, final_probability[:, 1], split="validation")
    threshold = select_threshold(search)
    pd.DataFrame(search).to_csv(out / "threshold_search.csv", index=False)
    write_json(out / "threshold_search.json", {"selection_split": "validation", "objective": "macro_f1", "selected": threshold, "rows": search})
    plt.plot([item["threshold"] for item in search], [item["macro_f1"] for item in search]); plt.axvline(threshold["threshold"], color="red"); plt.xlabel("Malignant probability threshold"); plt.ylabel("Validation macro-F1"); plt.tight_layout(); plt.savefig(out / "threshold_plot.png", dpi=150); plt.close()
    models = [{"name": name, "strategy": name, "checkpoint": winner[name]["checkpoint"], "metadata_fields": winner[name]["config"].get("metadata_fields", [])} for name in final_members]
    frozen = {"version": 1, "status": "frozen", "task": "diagnosis_binary", "class_order": ["benign", "malignant"], "models": models,
              "ensemble": {"identifier": chosen_name, "method": "equal_weight_probability_average", "member_names": final_members, "weights": final_weights, "missing_member_policy": "error", "fit_split": "validation"},
              "threshold": {"threshold": threshold["threshold"], "objective": "macro_f1", "fit_split": "validation", "selected_metrics": threshold},
              "calibration": {"enabled": False, "fit_split": "validation"},
              "preprocessing": {"image_resolution": 224, "normalization": "ImageNet weights transforms via build_transforms", "metadata_fields": winner.get("multimodal", {}).get("config", {}).get("metadata_fields", []), "metadata_preprocessor": "checkpoint-persisted training-only state"},
              "dataset_mappings": {"development": "persisted diagnosis_binary mapping", "DDI": "must be supplied and validated before external final test"}, "development_datasets": sorted(manifest.dataset.dropna().unique().tolist()), "seed": 42, "selection_split": "validation"}
    frozen_path = freeze_final_configuration(frozen, out / "frozen_final_model.yaml")
    config_path = Path(args.frozen_config); config_path.parent.mkdir(parents=True, exist_ok=True); config_path.write_text(frozen_path.read_text(encoding="utf-8"), encoding="utf-8")
    # No post-test decisions: use the frozen component identities, weights, and threshold unchanged.
    test_collections = {}
    for name in final_members:
        loader = build_evaluation_loader(manifest, bundles[name], split="test")
        test_collections[name] = collect_predictions(bundles[name].model, loader, class_order=["benign", "malignant"])
    test_aligned = align_prediction_collections(test_collections)
    reference = next(iter(test_aligned.values()))
    test_members = [EnsembleMember(name, "diagnosis_binary", ("benign", "malignant"), test_aligned[name]["probabilities"], test_aligned[name]["sample_ids"], name == "multimodal") for name in final_members]
    test_probability, _ = average_probabilities(test_members, final_weights, missing_member_policy="error")
    test_prediction = (test_probability[:, 1] >= threshold["threshold"]).astype(int)
    test_metrics = major_result_report(reference["targets"], test_prediction, test_probability, class_order=["benign", "malignant"], groups=[row.get("patient_id") or row.get("lesion_id") or row.get("image_id") for row in reference["metadata"]], n_resamples=1000, seed=42)
    write_json(evaluation / "dev_test_metrics.json", {"evaluation_scope": "internal_development_test", "frozen_config": str(config_path), "metrics": test_metrics})
    export_predictions(prediction_frame(reference["targets"], test_probability, reference["metadata"], class_order=["benign", "malignant"], task="diagnosis_binary", strategy="frozen_system", ensemble_id=chosen_name, threshold=threshold["threshold"]), evaluation / "dev_test_predictions.csv")
    winners_table = all_rows.sort_values("validation_macro_f1", ascending=False).groupby("architecture", as_index=False).first()
    def markdown_table(frame):
        columns = list(frame.columns)
        return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] + ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in frame.iterrows()])
    report = ["# Final model selection report", "", "All model and threshold choices used validation data only. DDI was not accessed by this workflow.", "", "## Architecture winners", "", markdown_table(winners_table[["architecture", "run_id", "validation_macro_f1", "checkpoint"]]), "", "## Ensemble selection", "", markdown_table(comparison_frame[["system", "macro_f1", "balanced_accuracy", "roc_auc", "sensitivity", "specificity"]]), "", f"Selected frozen system: `{chosen_name}` at threshold `{threshold['threshold']:.2f}` (validation macro-F1 {threshold['macro_f1']:.4f}).", "", "## DINOv2 and metadata", "", "Frozen DINOv2 and last-two-block tuning were essentially tied; frozen was retained. The multimodal component uses age, sex, and anatomical site only. Skin tone remains excluded because the prior ablation showed source confounding.", "", "## Limitations", "", "This diagnostic research model expects an appropriate focal clinical/macro lesion photograph; it is not a general classifier for healthy skin, rashes, acne, cuts, or arbitrary images. Source/domain shift, class imbalance, demographic/acquisition coverage, and external clinical validation remain material limitations. It is not a clinical diagnostic device."]
    (out / "model_selection_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"winners": {key: value["checkpoint"] for key, value in winner.items()}, "selected": chosen_name, "threshold": threshold["threshold"], "dev_test_macro_f1": test_metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
