"""Isolated, confirmation-gated post-hoc evaluation of v2 on DDI.

This entry point deliberately never calls the v1 final-test runner and never
writes beneath ``results/final_evaluation``.  DDI is opened only after the
explicit acknowledgement flag is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.data.ddi import build_ddi_manifest, validate_ddi_evaluation_manifest
from src.evaluation.evaluator import (
    build_evaluation_loader,
    collect_predictions,
    load_checkpoint_bundle,
)
from src.evaluation.metrics import classification_metrics
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble
from src.utils.device import get_device

ROOT = Path(__file__).resolve().parent
DEFAULT_V1_OUTPUT = ROOT / "results/final_evaluation/ddi"
DEFAULT_V1_CONFIG = ROOT / "configs/deployments/v1_frozen/ensemble.yaml"
DEFAULT_V2_CONFIG = ROOT / "configs/deployments/v2_adaptive/ensemble.yaml"
DEFAULT_OUTPUT = ROOT / "results/posthoc_evaluation/ddi/v2_adaptive"
POSTHOC_NOTICE = (
    "This is a post-hoc evaluation of the v2 adaptive ensemble on DDI. DDI had previously been "
    "inspected during analysis of v1 and therefore these results do not constitute a new untouched "
    "external validation. V2 weighting parameters were selected using validation data only; DDI was "
    "not used to numerically tune those weights. DDI findings nevertheless influenced the subsequent "
    "research direction. Historical clean DDI results belong to v1 only."
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def guard_paths(*, output: Path, v1_output: Path, v1_config: Path, v2_config: Path) -> None:
    """Reject every output location that could affect immutable v1 artifacts."""
    if output.resolve() == v1_output.resolve() or _is_relative_to(output, v1_output):
        raise ValueError("Post-hoc v2 output must be distinct from and outside the protected v1 DDI directory")
    if v1_config.resolve() == v2_config.resolve():
        raise ValueError("v1 and v2 deployment configuration paths must be distinct")
    if not v1_output.is_dir():
        raise FileNotFoundError(f"Expected protected v1 DDI directory was not found: {v1_output}")
    if not v1_config.is_file() or not v2_config.is_file():
        raise FileNotFoundError("Both the v1 reference config and v2 deployment config must exist")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing v2 post-hoc result directory")


def _load_v2_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("identifier") != "v2_adaptive" or config.get("task") != "diagnosis_binary":
        raise ValueError("Expected the v2_adaptive diagnosis_binary deployment configuration")
    policy = (config.get("ensemble") or {}).get("no_metadata_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("v2 configuration has no no_metadata_policy")
    definitions = {item.get("name"): item for item in config.get("models", [])}
    active_names = ("convnext", "efficientnet")
    if any(name not in definitions for name in (*active_names, "multimodal")):
        raise ValueError("v2 configuration must declare ConvNeXt, EfficientNet, and multimodal members")
    if tuple((config.get("ensemble") or {}).get("member_names") or ()) != ("convnext", "efficientnet", "multimodal"):
        raise ValueError("v2 member ordering is not the saved deployment contract")
    threshold = (config.get("threshold") or {}).get("threshold")
    if threshold is None:
        raise ValueError("v2 configuration has no saved threshold")
    return config, dict(policy), [definitions[name] for name in active_names]


def preflight(*, ddi_root: Path, output: Path, v1_output: Path, v1_config: Path, v2_config: Path) -> dict[str, Any]:
    """Perform all read-only checks; this function never creates result files."""
    guard_paths(output=output, v1_output=v1_output, v1_config=v1_config, v2_config=v2_config)
    config, no_metadata_policy, active_definitions = _load_v2_policy(v2_config)
    manifest, dataset_report = build_ddi_manifest(ddi_root)
    eligible = validate_ddi_evaluation_manifest(manifest.loc[manifest.binary_target.notna()].copy())
    counts = eligible.binary_target.astype(int).value_counts().to_dict()
    if len(eligible) != 656 or counts.get(0) != 485 or counts.get(1) != 171:
        raise ValueError(f"Unexpected DDI cohort: {len(eligible)} samples, benign={counts.get(0, 0)}, malignant={counts.get(1, 0)}")
    v1_report = v1_output / "ddi_report.json"
    if not v1_report.is_file():
        raise FileNotFoundError(f"Expected historical v1 report was not found: {v1_report}")
    hashes = {"v2_config": _sha256(v2_config), "v1_config": _sha256(v1_config), "v1_report": _sha256(v1_report)}
    for definition in active_definitions:
        checkpoint = _resolve(str(definition["checkpoint"]).replace("\\", "/"))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"v2 checkpoint not found: {checkpoint}")
        actual = _sha256(checkpoint)
        if definition.get("checkpoint_sha256") and actual != definition["checkpoint_sha256"]:
            raise ValueError(f"v2 checkpoint hash mismatch: {checkpoint}")
        hashes[f"{definition['name']}_checkpoint"] = actual
    return {"config": config, "policy": no_metadata_policy,
            "metadata_available_policy": dict(config["ensemble"]["metadata_available_policy"]), "active_definitions": active_definitions,
            "eligible": eligible, "dataset_report": dataset_report, "hashes": hashes,
            "v1_report": v1_report, "v1_config": v1_config, "threshold": float(config["threshold"]["threshold"])}


def _prediction_csv(reference: dict[str, Any], individual: Mapping[str, dict[str, Any]], final_probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    targets = reference["targets"].astype(int)
    predicted = (final_probabilities >= threshold).astype(int)
    outcome = np.where((targets == 0) & (predicted == 0), "TN", np.where((targets == 0), "FP", np.where(predicted == 0, "FN", "TP")))
    metadata = reference["metadata"]
    return pd.DataFrame({
        "ddi_id": [row.get("image_id") for row in metadata], "filename": [Path(row.get("image_path", "")).name for row in metadata],
        "ground_truth": targets, "convnext_malignant_probability": individual["convnext"]["probabilities"][:, 1],
        "efficientnet_malignant_probability": individual["efficientnet"]["probabilities"][:, 1],
        "multimodal_status": "inactive", "multimodal_malignant_probability": pd.NA,
        "final_v2_malignant_probability": final_probabilities, "predicted_class": predicted,
        "threshold": threshold, "correct": targets == predicted, "confusion_outcome": outcome,
    })


def _with_rates(metrics: dict[str, Any], predictions: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(metrics["confusion_matrix"])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {**metrics, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "false_positive_rate": None if tn + fp == 0 else fp / (tn + fp),
            "false_negative_rate": None if fn + tp == 0 else fn / (fn + tp),
            "predicted_malignant_rate": float(np.mean(predictions))}


def _comparison(v1_report: Path, v2_metrics: Mapping[str, Any]) -> dict[str, Any]:
    v1 = json.loads(v1_report.read_text(encoding="utf-8")).get("metrics", {})
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier_score")
    rows = [{"metric": key, "v1_frozen": v1.get(key), "v2_posthoc": v2_metrics.get(key),
             "v2_minus_v1": None if v1.get(key) is None or v2_metrics.get(key) is None else v2_metrics[key] - v1[key]} for key in keys]
    return {"comparison_type": "descriptive_only", "important_policy_difference":
            "v1 ran ConvNeXt, EfficientNet, and multimodal DINOv2 with missing metadata; v2 intentionally runs only ConvNeXt and EfficientNet because DDI lacks age, sex, and anatomical site. This is not solely a weighting comparison.", "rows": rows}


def execute(plan: Mapping[str, Any], *, output: Path, v2_config: Path) -> Path:
    """Run inference in a sibling staging directory, then atomically publish it."""
    stage = output.parent / f".{output.name}_tmp"
    if stage.exists():
        raise FileExistsError(f"Staging directory already exists; preserve it for diagnostics: {stage}")
    stage.mkdir(parents=True)
    try:
        device = get_device()
        individual: dict[str, dict[str, Any]] = {}
        reference = None
        for definition in plan["active_definitions"]:  # deliberately excludes multimodal
            checkpoint = _resolve(str(definition["checkpoint"]).replace("\\", "/"))
            bundle = load_checkpoint_bundle(checkpoint, strategy=definition.get("strategy"), device=device)
            loader = build_evaluation_loader(plan["eligible"], bundle, split=None, allow_final_test=True, frozen_config_path=v2_config)
            collected = collect_predictions(bundle.model, loader, device=device, class_order=["benign", "malignant"])
            if reference is not None and not np.array_equal(reference["targets"], collected["targets"]):
                raise ValueError("v2 model predictions are not aligned to the same DDI manifest")
            reference = reference or collected
            individual[definition["name"]] = collected
        final_probability = np.asarray([AdaptiveEnsemble(plan["config"]).combine(
            {name: values["probabilities"][index, 1] for name, values in individual.items()},
            metadata={"age": None, "sex": None, "anatomical_site": None},
        ).malignant_probability for index in range(len(reference["targets"]))])
        predicted = (final_probability >= plan["threshold"]).astype(int)
        metrics = _with_rates(classification_metrics(reference["targets"], predicted, np.column_stack((1 - final_probability, final_probability)), labels=[0, 1], class_names=["benign", "malignant"]), predicted)
        _prediction_csv(reference, individual, final_probability, plan["threshold"]).to_csv(stage / "ddi_v2_posthoc_predictions.csv", index=False)
        comparison = _comparison(plan["v1_report"], metrics)
        # Detect a concurrent change rather than publishing a report whose v1
        # comparison/reference hashes no longer describe the historical files.
        if _sha256(Path(plan["v1_config"])) != plan["hashes"]["v1_config"] or _sha256(Path(plan["v1_report"])) != plan["hashes"]["v1_report"]:
            raise RuntimeError("Protected v1 configuration or report changed during v2 post-hoc evaluation; refusing to publish")
        report = {"evaluation_type": "post-hoc DDI evaluation of v2", "methodological_notice": POSTHOC_NOTICE,
                  "dataset": {"sample_count": len(reference["targets"]), "benign": 485, "malignant": 171, "metadata_available": False},
                  "active_models": ["convnext", "efficientnet"], "inactive_models": {"multimodal": "DDI lacks age, sex, and anatomical_site"},
                  "no_metadata_policy": plan["policy"], "metadata_available_policy_saved_but_not_used_on_ddi": plan["metadata_available_policy"],
                  "threshold": plan["threshold"], "metrics": metrics,
                  "integrity_hashes": plan["hashes"], "v1_reference_report": str(plan["v1_report"]), "comparison_policy_note": comparison["important_policy_difference"]}
        (stage / "ddi_v2_posthoc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (stage / "v1_vs_v2_posthoc_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        pd.DataFrame(comparison["rows"]).to_csv(stage / "v1_vs_v2_posthoc_comparison.csv", index=False)
        (stage / "EVALUATION_COMPLETE.json").write_text(json.dumps({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_type": "post-hoc DDI evaluation of v2"}, indent=2), encoding="utf-8")
        os.replace(stage, output)
        return output
    except Exception:
        raise


def _console(plan: Mapping[str, Any], *, output: Path, v1_output: Path, dry_run: bool) -> None:
    print("Evaluation type: POST-HOC\nModel variant: v2_adaptive\n")
    print("DDI samples: 656\nBenign: 485\nMalignant: 171\n\nMetadata available: no")
    print("Active models:\n  ConvNeXt-Tiny\n  EfficientNetV2-S\n\nInactive:\n  Multimodal DINOv2")
    print(f"\nNo-metadata aggregation: {plan['policy']['strategy']}\nThreshold: {plan['threshold']}")
    print(f"\nOutput:\n{output}\n\nProtected v1 output:\n{v1_output}\n\nV1 modifications planned: NONE")
    if dry_run: print("\nDry run complete. No inference performed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated post-hoc DDI evaluation of v2 adaptive ensemble.")
    parser.add_argument("--ddi-root", default="data/final_external_test/ddi")
    parser.add_argument("--output", default="results/posthoc_evaluation/ddi/v2_adaptive")
    parser.add_argument("--v1-output", default="results/final_evaluation/ddi")
    parser.add_argument("--v1-config", default="configs/deployments/v1_frozen/ensemble.yaml")
    parser.add_argument("--v2-config", default="configs/deployments/v2_adaptive/ensemble.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-posthoc-ddi", action="store_true")
    args = parser.parse_args()
    plan = preflight(ddi_root=_resolve(args.ddi_root), output=_resolve(args.output), v1_output=_resolve(args.v1_output), v1_config=_resolve(args.v1_config), v2_config=_resolve(args.v2_config))
    _console(plan, output=_resolve(args.output), v1_output=_resolve(args.v1_output), dry_run=args.dry_run)
    if args.dry_run: return
    if not args.confirm_posthoc_ddi:
        parser.error("Refusing DDI inference. Re-run with --confirm-posthoc-ddi after reviewing the dry-run report.")
    destination = execute(plan, output=_resolve(args.output), v2_config=_resolve(args.v2_config))
    print(f"Post-hoc v2 DDI evaluation complete: {destination}")


if __name__ == "__main__":
    main()
