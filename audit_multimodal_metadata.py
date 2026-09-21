"""Write an auditable metadata report for the approved development manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.datasets import select_task_manifest, validate_manifest
from src.data.metadata import metadata_audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/development_manifest.csv")
    parser.add_argument("--output", default="results/multimodal_metadata_audit.json")
    args = parser.parse_args()
    manifest = validate_manifest(pd.read_csv(args.manifest, low_memory=False))
    report = metadata_audit_report(select_task_manifest(manifest, "diagnosis_binary"))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
