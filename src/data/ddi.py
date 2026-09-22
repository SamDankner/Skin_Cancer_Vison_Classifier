"""DDI ingestion and leakage auditing for the one-time external final test.

This module intentionally accepts only an official DDI export already obtained
by an individually registered user.  It never downloads data or infers labels.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import hashlib
import json

import pandas as pd

from .datasets import MANIFEST_COLUMNS, file_sha256, validate_manifest

DDI_PORTAL = "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965"
DDI_PROJECT = "https://ddi-dataset.github.io/"
DDI_PAPER = "https://doi.org/10.1126/sciadv.abq6147"


def _first(frame: pd.DataFrame, names: tuple[str, ...], required: bool = False) -> str | None:
    for name in names:
        if name in frame:
            return name
    if required:
        raise ValueError(f"Official DDI metadata is missing one of {names}")
    return None


def build_ddi_manifest(root: str | Path) -> tuple[pd.DataFrame, dict]:
    """Build a DDI-only manifest from the official ``ddi_metadata.csv`` export.

    The official code specifies ``DDI_file`` and either ``malignant`` or
    ``malignancy(malig=1)``.  That direct source flag is the only binary mapping
    accepted here; diagnosis strings are retained for reporting, never guessed.
    """
    root = Path(root)
    metadata_path = root / "ddi_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Place the official ddi_metadata.csv at {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    filename = _first(metadata, ("DDI_file", "filename", "file_name"), True)
    malignancy = _first(metadata, ("malignant", "malignancy(malig=1)"), True)
    diagnosis = _first(metadata, ("disease", "diagnosis", "disease_name"))
    patient = _first(metadata, ("patient_id", "patient", "Patient_ID"))
    rows = []
    for index, item in metadata.iterrows():
        source_name = str(item[filename])
        matches = list((root / "images").glob(source_name)) + list(root.glob(source_name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one image for DDI metadata row {index}: {source_name}")
        raw_label = item[malignancy]
        if pd.isna(raw_label) or str(raw_label).strip() not in {"0", "1", "0.0", "1.0", "False", "True", "false", "true"}:
            target, status, reason = pd.NA, "excluded", "missing_or_invalid_official_malignancy_flag"
        else:
            target = int(float(raw_label)) if str(raw_label).strip().lower() not in {"true", "false"} else int(str(raw_label).strip().lower() == "true")
            status, reason = "included", "official DDI malignancy flag"
        values = {name: pd.NA for name in MANIFEST_COLUMNS}
        values.update({"dataset": "DDI", "source_dataset": "DDI", "image_path": str(matches[0]), "image_id": source_name,
                       "patient_id": item[patient] if patient else pd.NA, "original_label": item[diagnosis] if diagnosis else pd.NA,
                       "harmonized_diagnosis": item[diagnosis] if diagnosis else pd.NA, "binary_target": target,
                       # DDI consists of clinical photographs selected from
                       # pathology reports.  Marking a labelled DDI row as a
                       # lesion is required by this project's diagnosis-task
                       # selector; it does not alter its benign/malignant flag.
                       "lesion_present": status == "included", "supported_for_diagnosis": status == "included", "image_modality": "clinical",
                       "skin_tone": item.get("skin_tone", pd.NA), "age": item.get("age", pd.NA), "sex": item.get("sex", pd.NA),
                       "anatomical_site": item.get("anatomical_site", item.get("site", pd.NA)), "split": "final_external_test"})
        values["inclusion_status"], values["exclusion_reason"] = status, reason
        rows.append(values)
    frame = pd.DataFrame(rows)
    manifest = validate_manifest(frame[MANIFEST_COLUMNS], allow_final_test=True)
    report = {"source_url": DDI_PORTAL, "project_url": DDI_PROJECT, "paper": DDI_PAPER,
              "terms": "Stanford DDI Research Use Agreement; personal, non-commercial research only; do not redistribute.",
              "download_date": date.today().isoformat(), "metadata_file": str(metadata_path), "metadata_sha256": file_sha256(metadata_path),
              "image_count": len(frame), "eligible_binary_count": int(frame.inclusion_status.eq("included").sum()),
              "excluded_count": int(frame.inclusion_status.eq("excluded").sum())}
    return manifest, report


def validate_ddi_evaluation_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Fail before model loading unless final-test rows have usable binary targets."""
    if manifest.empty:
        raise ValueError("DDI external evaluation has zero eligible samples")
    target = pd.to_numeric(manifest["binary_target"], errors="coerce")
    if target.isna().any():
        raise ValueError("DDI eligible manifest contains null or non-numeric binary_target values")
    observed = set(target.astype(int))
    if observed != {0, 1}:
        raise ValueError(f"DDI external evaluation requires both canonical targets 0 and 1; observed {sorted(observed)}")
    if not target.isin([0, 1]).all():
        raise ValueError("DDI binary_target values must be only 0 or 1")
    missing = [path for path in manifest.image_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"DDI eligible manifest has {len(missing)} missing image paths; first: {missing[0]}")
    ids = manifest.image_id.astype(str)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("DDI sample IDs/image filenames must be non-empty and unique")
    result = manifest.copy()
    result["binary_target"] = target.astype("Int64")
    return result


def audit_ddi_overlap(manifest: pd.DataFrame, development_manifest: str | Path) -> dict:
    """Audit exact hashes and image IDs; no DDI image is silently removed."""
    dev = pd.read_csv(development_manifest)
    ddi_hashes = {str(row.image_id): file_sha256(row.image_path) for _, row in manifest.iterrows()}
    dev_hashes = {}
    for _, row in dev.dropna(subset=["image_path"]).iterrows():
        path = Path(row.image_path)
        if path.is_file():
            dev_hashes.setdefault(file_sha256(path), []).append(str(row.get("image_id", "")))
    exact = [{"ddi_image_id": key, "development_image_ids": dev_hashes[value]} for key, value in ddi_hashes.items() if value in dev_hashes]
    ids = sorted(set(manifest.image_id.astype(str)) & set(dev.get("image_id", pd.Series(dtype=str)).dropna().astype(str)))
    return {"ddi_samples": len(manifest), "development_manifest": str(development_manifest), "exact_hash_overlap": exact,
            "shared_image_ids": ids, "clean_external_test": not exact and not ids}
