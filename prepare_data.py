"""Opt-in development-data manifest preparation.  Never handles DDI."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preparation import DEVELOPMENT_DATASETS, build_development_manifest, write_provenance
from src.data.scin_download import download_scin_dataset


def main():
    """Record source provenance and build the development manifest."""
    parser = argparse.ArgumentParser(
        description="Build safe manifests from approved raw clinical-photo datasets."
    )
    parser.add_argument("--dataset", choices=DEVELOPMENT_DATASETS, action="append")
    parser.add_argument(
        "--all-development",
        action="store_true",
        help="Acquire approved automatic sources, record provenance, and build manifests.",
    )
    parser.add_argument(
        "--skip-scin-download",
        action="store_true",
        help="Build from local SCIN files without contacting the official bucket.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    raw = root / "data" / "raw"
    selected = args.dataset or DEVELOPMENT_DATASETS

    scin_acquisition = None
    if "SCIN" in selected and not args.skip_scin_download:
        scin_acquisition = download_scin_dataset(raw)

    for dataset in selected:
        write_provenance(
            raw,
            dataset,
            manual_required=dataset in {"Fitzpatrick17k", "ImageQX", "Muhaba", "ENCoDE"},
            acquisition_report=scin_acquisition if dataset == "SCIN" else None,
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
