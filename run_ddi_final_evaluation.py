"""Deliberate, non-tuning entry point for the frozen DDI final evaluation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.data.ddi import audit_ddi_overlap, build_ddi_manifest, validate_ddi_evaluation_manifest
from src.data.datasets import write_manifest
from src.evaluation.evaluator import evaluate_frozen_external_test

_PREFLIGHT_ARTIFACTS = {"ddi_dataset_report.json", "ddi_overlap_audit.json"}


def _guard_output_lifecycle(output: Path) -> None:
    """Allow only recoverable preflight files; completed/inference outputs lock DDI."""
    if not output.exists():
        return
    existing = {item.name for item in output.iterdir()}
    if existing and not existing <= _PREFLIGHT_ARTIFACTS:
        raise FileExistsError("Refusing to overwrite completed or unowned DDI final-test artifacts")

def main():
    parser = argparse.ArgumentParser(description="Run the frozen one-time DDI external evaluation.")
    parser.add_argument("--ddi-root", default="data/final_external_test/ddi")
    parser.add_argument("--development-manifest", default="data/processed/development_manifest.csv")
    parser.add_argument("--frozen-config", default="configs/final_model.yaml")
    parser.add_argument("--output", default="results/final_evaluation/ddi")
    parser.add_argument("--allow-final-test", action="store_true", help="Required acknowledgement before DDI images are opened.")
    args = parser.parse_args()
    if not args.allow_final_test:
        parser.error("DDI is final test data; explicitly pass --allow-final-test")
    out = Path(args.output)
    _guard_output_lifecycle(out)
    manifest, dataset_report = build_ddi_manifest(args.ddi_root)
    out.mkdir(parents=True)
    write_manifest(manifest, "data/final_external_test/ddi_manifest.csv", allow_final_test=True)
    (out / "ddi_dataset_report.json").write_text(json.dumps(dataset_report, indent=2), encoding="utf-8")
    overlap = audit_ddi_overlap(manifest, args.development_manifest)
    (out / "ddi_overlap_audit.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    if not overlap["clean_external_test"]: raise RuntimeError("DDI overlap detected; review audit before final inference")
    eligible = validate_ddi_evaluation_manifest(manifest.loc[manifest.binary_target.notna()].copy())
    result = evaluate_frozen_external_test(eligible, allow_final_test=True, frozen_config_path=args.frozen_config, output_directory=out, allow_existing_preflight=True)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
if __name__ == "__main__": main()
