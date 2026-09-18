"""Reproducible, conservative ingestion for approved clinical-photo datasets.

This module deliberately does not download data on import and never reads DDI.
It builds a development manifest only from files already placed under ``data/raw``
or explicitly downloaded from a documented first-party URL.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Iterable
from hashlib import sha256

import pandas as pd
from PIL import Image, UnidentifiedImageError

# A preparation-only environment need not install PyTorch.  Prefer the public
# project interfaces, but retain an equivalent schema-only fallback so that
# data acquisition can happen before the training stack is installed.
try:
    from src.data.datasets import MANIFEST_COLUMNS, add_exact_hashes, empty_manifest, write_manifest
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    MANIFEST_COLUMNS = ["dataset", "image_path", "image_id", "patient_id", "lesion_id", "original_label", "harmonized_diagnosis", "binary_target", "lesion_present", "normal_skin", "image_quality_label", "localization_available", "bounding_box", "segmentation_mask_path", "supported_for_lesion_detection", "supported_for_diagnosis", "age", "sex", "anatomical_site", "skin_tone", "image_modality", "label_source", "ground_truth_method", "split"]
    def empty_manifest():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    def add_exact_hashes(manifest):
        result = manifest.copy()
        def digest(path):
            value = sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1_048_576), b""):
                    value.update(block)
            return value.hexdigest()
        result["file_sha256"] = result.image_path.map(digest)
        result["exact_duplicate_flag"] = result.file_sha256.duplicated(keep=False)
        return result
    def write_manifest(manifest, path, allow_final_test=False):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        manifest[MANIFEST_COLUMNS].to_csv(path, index=False)
        return path
from src.data.harmonize_labels import harmonize_manifest
try:
    from src.data.splits import class_coverage_report, leakage_report, make_group_splits
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    def _group(frame):
        patient = frame.patient_id.astype("string").fillna("")
        lesion = frame.lesion_id.astype("string").fillna("")
        return ("image:" + frame.image_id.astype(str)).where(lesion.eq(""), "lesion:" + lesion).where(patient.eq(""), "patient:" + patient)
    def make_group_splits(frame, train_fraction=.70, validation_fraction=.15, random_state=42):
        result = frame.copy(); groups = sorted(_group(result).unique())
        if len(groups) < 3:
            raise ValueError("Need at least three independent groups")
        # Stable ordering keeps reruns reproducible without adding a dependency.
        ordered = sorted(groups, key=lambda value: sha256(f"{random_state}:{value}".encode()).hexdigest())
        train_end, validation_end = round(len(ordered) * train_fraction), round(len(ordered) * (train_fraction + validation_fraction))
        labels = {group: "train" for group in ordered[:train_end]}
        labels.update({group: "validation" for group in ordered[train_end:validation_end]})
        labels.update({group: "test" for group in ordered[validation_end:]})
        result["split"] = _group(result).map(labels); return result
    def leakage_report(frame):
        return frame.assign(_group=_group(frame)).groupby("_group").split.nunique().rename("split_count").loc[lambda value: value > 1].reset_index()
    def class_coverage_report(frame, target_column, required_splits=("train", "validation")):
        classes = set(frame[target_column].dropna())
        return pd.DataFrame([{"split": split, "present_classes": sorted(set(frame.loc[frame.split.eq(split), target_column].dropna()), key=str), "missing_classes": sorted(classes - set(frame.loc[frame.split.eq(split), target_column].dropna()), key=str), "complete": classes == set(frame.loc[frame.split.eq(split), target_column].dropna())} for split in required_splits])

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEVELOPMENT_DATASETS = ("PAD-UFES-20", "MILK10k", "SCIN", "Fitzpatrick17k")
SOURCES = {
    "PAD-UFES-20": {"url": "https://data.mendeley.com/datasets/zr7vgbcyr2/1", "doi": "10.17632/zr7vgbcyr2.1", "terms": "CC BY 4.0"},
    "MILK10k": {"url": "https://api.isic-archive.com/doi/milk10k/", "doi": "10.34970/648456", "terms": "CC-BY-NC"},
    "SCIN": {"url": "https://github.com/google-research-datasets/scin", "doi": "10.1001/jamanetworkopen.2024.46615", "terms": "SCIN Data Use License"},
    "Fitzpatrick17k": {"url": "https://github.com/mattgroh/fitzpatrick17k", "doi": "Groh et al., CVPR 2021", "terms": "Images remain subject to their original-source terms"},
}

def _norm(value):
    return "" if pd.isna(value) else str(value).strip()

def _column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    return next((lookup[name] for name in names if name in lookup), None)

def _metadata_file(directory: Path) -> Path | None:
    csvs = sorted(directory.rglob("*.csv"), key=lambda p: ("metadata" not in p.name.lower(), -p.stat().st_size))
    return csvs[0] if csvs else None

def _image_id(path: Path) -> str:
    return path.stem

def _valid_images(directory: Path):
    valid, invalid = [], []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            if path.stat().st_size == 0:
                raise UnidentifiedImageError("zero-byte file")
            with Image.open(path) as im:
                im.verify()
            valid.append(path)
        except (OSError, UnidentifiedImageError) as exc:
            invalid.append({"image_path": str(path.resolve()), "reason": str(exc)})
    return valid, invalid

def _clinical_from_metadata(dataset: str, row: pd.Series | None) -> bool:
    """Require authoritative modality metadata for paired MILK10k images.

    PAD-UFES-20 and SCIN are released as ordinary clinical photographs.  The
    Fitzpatrick17k annotation file does not reliably declare modality, so its
    files are retained for provenance but excluded until the user supplies an
    official clinical-only bundle or review.
    """
    if dataset in {"PAD-UFES-20", "SCIN"}:
        return True
    if dataset == "Fitzpatrick17k" or row is None:
        return False
    modality = next((row[c] for c in row.index if str(c).lower() in {"modality", "image_modality", "image type", "image_type"}), None)
    value = _norm(modality).lower()
    return value in {"clinical", "macro", "clinical close-up", "clinical: close-up", "close-up"}

def _field(row: pd.Series | None, *names):
    if row is None:
        return None
    col = _column(pd.DataFrame([row]), names)
    return None if col is None else (None if not _norm(row[col]) else row[col])

def build_dataset_manifest(raw_root: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    """Build one conservative dataset manifest and an ingestion report."""
    directory = raw_root / dataset
    if not directory.exists():
        return empty_manifest(), {"dataset": dataset, "status": "not present", "valid_images": 0, "invalid_images": []}
    metadata_path = _metadata_file(directory)
    metadata = pd.read_csv(metadata_path, low_memory=False) if metadata_path else pd.DataFrame()
    id_col = _column(metadata, ("image_id", "image", "image_name", "isic_id", "filename", "file_name", "img_id"))
    metadata_by_id = {_norm(row[id_col]).removesuffix(Path(_norm(row[id_col])).suffix): row for _, row in metadata.iterrows()} if id_col else {}
    images, invalid = _valid_images(directory)
    rows = []
    for image in images:
        image_id = _image_id(image)
        row = metadata_by_id.get(image_id)
        clinical = _clinical_from_metadata(dataset, row)
        original = _field(row, "diagnosis_3", "diagnostic", "diagnosis", "dx", "label", "three_partition_label")
        item = {column: None for column in MANIFEST_COLUMNS}
        item.update({
            "dataset": dataset, "image_path": str(image.resolve()), "image_id": image_id,
            "patient_id": _field(row, "patient_id", "patient", "case_id", "case"),
            "lesion_id": _field(row, "lesion_id", "lesion"), "original_label": original,
            "lesion_present": True if clinical else None, "normal_skin": False if clinical else None,
            "supported_for_lesion_detection": clinical, "supported_for_diagnosis": clinical and original is not None,
            "age": _field(row, "age", "age_approx"), "sex": _field(row, "sex", "gender"),
            "anatomical_site": _field(row, "anatomical_site", "anatom_site_general", "location"),
            "skin_tone": _field(row, "skin_tone", "fitzpatrick", "fitzpatrick_skin_type", "mst"),
            "image_modality": "clinical" if clinical else "unknown",
            "label_source": "dermatologist" if dataset == "SCIN" else "dataset annotation",
            "ground_truth_method": _field(row, "ground_truth_method", "diagnosis_method", "diagnosis_confirm_type", "benign_malignant"),
        })
        rows.append(item)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not frame.empty:
        frame = harmonize_manifest(frame)
        # MILK10k's first diagnosis tier is the publisher's explicit benign /
        # malignant classification.  It is not inferred from image content.
        if dataset == "MILK10k" and "diagnosis_1" in metadata:
            tiers = frame.image_id.map(lambda image_id: _norm(metadata_by_id.get(image_id, {}).get("diagnosis_1", "")).lower())
            explicit_target = tiers.map({"benign": 0, "malignant": 1}).astype("Int64")
            frame.loc[explicit_target.notna(), "binary_target"] = explicit_target[explicit_target.notna()]
            frame.loc[explicit_target.notna(), "binary_mapping_reason"] = "MILK10k publisher benign/malignant tier"
        frame["supported_for_diagnosis"] &= frame.binary_target.notna() | frame.harmonized_diagnosis.notna()
    return frame, {"dataset": dataset, "status": "ready", "metadata_file": str(metadata_path) if metadata_path else None, "valid_images": len(images), "eligible_clinical_images": int(frame.image_modality.eq("clinical").sum()) if not frame.empty else 0, "invalid_images": invalid, "disk_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())}

def build_development_manifest(project_root: str | Path = ".") -> dict:
    """Combine approved sources, make safe splits, and persist reports."""
    root = Path(project_root).resolve(); raw = root / "data" / "raw"; reports = []
    frames = []
    for dataset in DEVELOPMENT_DATASETS:
        frame, report = build_dataset_manifest(raw, dataset); frames.append(frame); reports.append(report)
    manifest = pd.concat(frames, ignore_index=True) if any(not f.empty for f in frames) else empty_manifest()
    eligible = manifest.loc[manifest.image_modality.eq("clinical")].copy()
    if len(eligible) >= 3:
        eligible = make_group_splits(eligible)
    if not eligible.empty:
        eligible = add_exact_hashes(eligible)
    manifest.loc[eligible.index, MANIFEST_COLUMNS] = eligible[MANIFEST_COLUMNS]
    # Optional hash fields let the experiment runner detect exact duplicates
    # that cross otherwise independent patient/lesion groups.
    manifest["file_sha256"] = pd.NA
    manifest["exact_duplicate_flag"] = False
    if not eligible.empty:
        manifest.loc[eligible.index, "file_sha256"] = eligible.file_sha256
        manifest.loc[eligible.index, "exact_duplicate_flag"] = eligible.exact_duplicate_flag
    manifest_path = root / "data" / "processed" / "development_manifest.csv"
    write_manifest(manifest, manifest_path)
    duplicate_path = root / "data" / "processed" / "exact_duplicates.csv"
    (eligible.loc[eligible.exact_duplicate_flag] if "exact_duplicate_flag" in eligible else pd.DataFrame()).to_csv(duplicate_path, index=False)
    report = {"generated": date.today().isoformat(), "datasets": reports, "manifest": str(manifest_path), "rows": len(manifest), "eligible": len(eligible), "split_counts": eligible.split.value_counts().to_dict() if "split" in eligible else {}, "class_coverage": class_coverage_report(eligible, "binary_target").to_dict("records") if not eligible.empty else [], "leakage": leakage_report(eligible).to_dict("records") if not eligible.empty else []}
    (root / "data" / "processed" / "preparation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

def write_provenance(raw_root: Path, dataset: str, downloaded_files: list[str] | None = None, manual_required: bool = False) -> Path:
    """Write source, license, download, and exclusion provenance metadata."""
    folder = raw_root / dataset; folder.mkdir(parents=True, exist_ok=True)
    source = SOURCES[dataset]
    payload = {"dataset": dataset, "official_source_url": source["url"], "citation_or_doi": source["doi"], "license_or_terms": source["terms"], "download_date": date.today().isoformat(), "downloaded_files": downloaded_files or [], "extraction_method": "not performed by default", "exclusions": "MILK10k dermoscopy excluded; unknown modality is never eligible", "known_missing_files": [], "manual_access_required": manual_required}
    path = folder / "PROVENANCE.json"; path.write_text(json.dumps(payload, indent=2), encoding="utf-8"); return path
