"""Opt-in development-data manifest preparation.  Never handles DDI."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preparation import DEVELOPMENT_DATASETS, build_development_manifest, write_provenance


def main():
    """Record source provenance and build the development manifest."""
    parser = argparse.ArgumentParser(
        description="Build safe manifests from approved raw clinical-photo datasets."
    )
    parser.add_argument("--dataset", choices=DEVELOPMENT_DATASETS, action="append")
    parser.add_argument(
        "--all-development",
        action="store_true",
        help="Record provenance and build manifests; does not download.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    raw = root / "data" / "raw"
    selected = args.dataset or DEVELOPMENT_DATASETS

    for dataset in selected:
        write_provenance(
            raw,
            dataset,
            manual_required=dataset in {"Fitzpatrick17k", "ImageQX", "Muhaba", "ENCoDE"},
        )

    (root / "data" / "normal_skin").mkdir(parents=True, exist_ok=True)
    (root / "data" / "normal_skin" / "REQUIRES_DATA.md").write_text(
        "No approved normal-skin macro-photo dataset is configured. Do not relabel "
        "benign lesions as normal skin. Obtain an explicitly licensed clinical "
        "normal-skin source before enabling lesion-presence negatives.\n",
        encoding="utf-8",
    )

    report = build_development_manifest(root, datasets=selected)
    for dataset in report["datasets"]:
        print(f"{dataset['dataset']}: {dataset['status'].upper()}")
    print(report)

if __name__ == "__main__":
    main()
