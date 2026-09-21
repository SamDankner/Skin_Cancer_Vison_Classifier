"""Opt-in development-data manifest preparation.  Never handles DDI."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preparation import DEVELOPMENT_DATASETS, build_development_manifest, write_provenance
from src.data.mcsi_download import download_mcsi_dataset
from src.data.mendeley_download import download_mendeley_dataset
from src.data.msld_download import download_msld_dataset
from src.data.pad_download import download_pad_dataset
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
    pad_acquisition = None
    mcsi_acquisition = None
    msld_acquisition = None
    mendeley_acquisitions = {}
    if args.all_development and "PAD-UFES-20" in selected:
        pad_acquisition = download_pad_dataset(raw)
    if args.all_development and "MCSI" in selected:
        mcsi_acquisition = download_mcsi_dataset(raw)
    if args.all_development and "MSLD_v2" in selected:
        msld_acquisition = download_msld_dataset(raw)
    for dataset in ("ArsenicSkinImageBD", "MCVSLD", "MonkeyPox", "SkinDiseaseClassification"):
        if args.all_development and dataset in selected:
            mendeley_acquisitions[dataset] = download_mendeley_dataset(raw, dataset)
    if "SCIN" in selected and not args.skip_scin_download:
        scin_acquisition = download_scin_dataset(raw)

    for dataset in selected:
        write_provenance(
            raw,
            dataset,
            manual_required=dataset in {"Fitzpatrick17k", "ImageQX", "Muhaba", "ENCoDE"},
            acquisition_report=(
                scin_acquisition if dataset == "SCIN" else
                pad_acquisition if dataset == "PAD-UFES-20" else
                mcsi_acquisition if dataset == "MCSI" else
                msld_acquisition if dataset == "MSLD_v2" else None
                or mendeley_acquisitions.get(dataset)
            ),
        )

    (root / "data" / "normal_skin").mkdir(parents=True, exist_ok=True)
    (root / "data" / "normal_skin" / "REQUIRES_DATA.md").write_text(
        "The current core negative source is conservative SCIN LOOKS_HEALTHY, "
        "which is weak/self-reported. ImageQX and Muhaba require manual access. "
        "Do not relabel benign lesions or diffuse skin conditions as normal skin.\n",
        encoding="utf-8",
    )

    report = build_development_manifest(root, datasets=selected)
    for dataset in report["datasets"]:
        print(f"{dataset['dataset']}: {dataset['status'].upper()}")
    print(report)

if __name__ == "__main__":
    main()
