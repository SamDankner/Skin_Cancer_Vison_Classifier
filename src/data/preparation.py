"""Reproducible, conservative ingestion for approved clinical-photo datasets.

This module deliberately does not download data on import and never reads DDI.
It builds a development manifest only from files already placed under ``data/raw``
or explicitly downloaded from a documented first-party URL.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import ast
import json
import math
from pathlib import Path
from typing import Iterable
from hashlib import sha256
import re

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# A preparation-only environment need not install PyTorch.  Prefer the public
# project interfaces, but retain an equivalent schema-only fallback so that
# data acquisition can happen before the training stack is installed.
try:
    from src.data.datasets import (
        MANIFEST_COLUMNS,
        add_exact_hashes,
        add_perceptual_duplicate_groups,
        empty_manifest,
        write_manifest,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    MANIFEST_COLUMNS = ["dataset", "source_dataset", "image_path", "image_id", "patient_id", "case_id", "lesion_id", "original_label", "harmonized_diagnosis", "binary_target", "lesion_present", "normal_skin", "normal_label_strength", "normal_label_method", "gate_mapping_reason", "gate_label_strength", "gate_label_source", "gate_negative_subtype", "other_skin_condition", "image_quality_label", "localization_available", "bounding_box", "segmentation_mask_path", "supported_for_lesion_detection", "supported_for_lesion_presence", "supported_for_diagnosis", "age", "age_group", "sex", "anatomical_site", "skin_tone", "monk_skin_tone", "image_modality", "label_source", "ground_truth_method", "self_reported_related_category", "dermatologist_skin_condition_label", "weighted_skin_condition_label", "dermatologist_fitzpatrick_skin_type", "symptoms", "lesion_diameter_mm", "perceptual_hash", "duplicate_group_id", "split"]
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
    def add_perceptual_duplicate_groups(manifest, max_distance=4):
        # Lightweight preparation-only fallback when the training stack is not
        # installed.  Exact byte duplicates remain grouped by SHA-256.
        result = manifest.copy()
        result["perceptual_hash"] = pd.NA
        result["duplicate_group_id"] = result.file_sha256.map(
            lambda value: f"sha256:{value}" if pd.notna(value) else None
        )
        return result, pd.DataFrame(
            columns=["left_index", "right_index", "distance", "possible_duplicate"]
        )
    def write_manifest(manifest, path, allow_final_test=False):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        manifest[MANIFEST_COLUMNS].to_csv(path, index=False)
        return path
from src.data.harmonize_labels import harmonize_manifest
from src.data.scin_download import (
    SCIN_KNOWN_MISSING_OBJECTS,
    expected_scin_image_objects,
    local_scin_path,
    scin_metadata_paths,
    validate_image_file,
)
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
SCIN_CORE_NEGATIVE = {"LOOKS_HEALTHY"}
SCIN_CORE_POSITIVE = {"GROWTH_OR_MOLE"}
SCIN_OTHER_CATEGORIES = {
    "ACNE", "RASH", "PIGMENTARY_PROBLEM", "NAIL_PROBLEM",
    "OTHER_HAIR_PROBLEM", "HAIR_LOSS", "OTHER_ISSUE_DESCRIPTION",
}
IMAGEQX_LABELS = {
    "lesion": "positive",
    "healthy skin": "negative",
    "healthy": "negative",
    "poor quality": "unsupported_quality",
    "no skin": "unsupported_no_skin",
}
MUHABA_HEALTHY_LABELS = {"healthy", "healthy skin", "normal", "normal skin"}
MUHABA_OTHER_LABELS = {
    "acne vulgaris", "acne", "atopic dermatitis", "dermatitis",
    "lichen planus", "onychomycosis", "tinea capitis", "tinea", "unknown",
}
DEVELOPMENT_DATASETS = ("PAD-UFES-20", "MILK10k", "SCIN", "MSLD_v2", "MCSI", "ArsenicSkinImageBD", "MCVSLD", "MonkeyPox", "SkinDiseaseClassification", "Fitzpatrick17k", "ImageQX", "Muhaba", "ENCoDE")
SOURCES = {
    "PAD-UFES-20": {"url": "https://data.mendeley.com/datasets/zr7vgbcyr2/1", "doi": "10.17632/zr7vgbcyr2.1", "terms": "CC BY 4.0"},
    "MILK10k": {"url": "https://api.isic-archive.com/doi/milk10k/", "doi": "10.34970/648456", "terms": "CC-BY-NC"},
    "SCIN": {"url": "https://github.com/google-research-datasets/scin", "doi": "10.1001/jamanetworkopen.2024.46615", "terms": "SCIN Data Use License"},
    "Fitzpatrick17k": {"url": "https://github.com/mattgroh/fitzpatrick17k", "doi": "Groh et al., CVPR 2021", "terms": "Images remain subject to their original-source terms"},
    "ImageQX": {"url": "https://doi.org/10.1089/tmj.2022.0405", "doi": "10.1089/tmj.2022.0405", "terms": "Manual permission required; do not download from mirrors"},
    "Muhaba": {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9060152/", "doi": "10.1002/ski2.81", "terms": "Available from corresponding author on reasonable request"},
    "ENCoDE": {"url": "https://physionet.org/content/encode-skin-color/1.0.0/", "doi": "10.13026/mcgk-1s42", "terms": "PhysioNet credentialed access and data-use agreement required"},
    "MSLD_v2": {"url": "https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20", "doi": "MSLD v2.0", "terms": "CC BY-NC 4.0; requires normal Kaggle authentication"},
    "MCSI": {"url": "https://zenodo.org/records/8360076", "doi": "10.5281/zenodo.8360076", "terms": "Open Zenodo record; curated/cropped source images"},
    "ArsenicSkinImageBD": {"url": "https://data.mendeley.com/datasets/x4hgnjj5gv/2", "doi": "10.17632/x4hgnjj5gv.2", "terms": "CC BY-NC 3.0; original smartphone photos only; augmented files excluded"},
    "MCVSLD": {"url": "https://data.mendeley.com/datasets/dfztdtfsxz/1", "doi": "10.17632/dfztdtfsxz.1", "terms": "CC BY 4.0; aggregate provenance and duplicate audit required before admission"},
    "MonkeyPox": {"url": "https://data.mendeley.com/datasets/st6kggjr23/1", "doi": "10.17632/st6kggjr23.1", "terms": "CC BY 4.0; provenance/augmentation audit required before admission"},
    "SkinDiseaseClassification": {"url": "https://data.mendeley.com/datasets/schhndjbjp/1", "doi": "10.17632/schhndjbjp.1", "terms": "CC BY 4.0; clinical/modality, provenance, and duplicate audit required before admission"},
}


def _json_safe(value, path: str = "report"):
    """Return a strict-JSON-safe report value without obscuring its structure."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path} has a non-string dictionary key {key!r}; "
                    "represent grouped data as nested dictionaries or records"
                )
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=lambda item: (type(item).__name__, str(item)))
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(ordered)]
    raise TypeError(f"{path} contains unsupported value {value!r} ({type(value).__name__})")


def _grouped_count_records(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict]:
    """Represent grouped counts as records instead of tuple-keyed dictionaries."""
    if frame.empty:
        return []
    return (
        frame.groupby(list(columns), dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records")
    )

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


def _bool(value) -> bool:
    """Parse explicit source booleans; absent/unrecognized remains false."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _nonempty(value) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip()) and str(value).strip().lower() not in {"[]", "{}", "none", "nan", "null"}


def _scin_body_site(row: pd.Series) -> str | None:
    sites = [str(name).removeprefix("body_parts_").replace("_", " ").lower()
             for name, value in row.items() if str(name).lower().startswith("body_parts_") and _bool(value)]
    return "; ".join(sites) if sites else None


def _scin_image_paths(directory: Path, row: pd.Series) -> list[tuple[str, Path, object]]:
    """Resolve valid schema-declared SCIN images; official objects have no suffix."""
    found: list[tuple[str, Path, object]] = []
    for column, value in row.items():
        name = str(column).lower()
        if not name.startswith("image_") or not name.endswith("_path") or not _nonempty(value):
            continue
        object_name = str(value).strip().lstrip("/")
        path = local_scin_path(directory, object_name)
        if validate_image_file(path) is None:
            shot = row.get(str(column).replace("_path", "_shot_type"))
            found.append((str(column), path, shot))
    return found


def _scin_dermatologist_fitzpatrick(label: pd.Series | None) -> str | None:
    if label is None:
        return None
    values = []
    for index in range(1, 4):
        value = _field(label, f"dermatologist_fitzpatrick_skin_type_label_{index}")
        if _nonempty(value) and str(value) not in values:
            values.append(str(value))
    return "; ".join(values) if values else None


def build_scin_manifests(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Ingest official SCIN CSVs with conservative, explicitly weak normal labels."""
    directory = raw_root / "SCIN"
    cases_path, labels_path = scin_metadata_paths(directory)
    if not cases_path.is_file():
        empty = empty_manifest()
        return empty, empty.copy(), empty.copy(), {
            "dataset": "SCIN", "status": "not_present", "total_cases": 0,
            "expected_image_objects": 0, "valid_images": 0,
            "missing_images": [], "corrupt_images": [], "invalid_images": [],
        }
    cases = pd.read_csv(cases_path, dtype={"case_id": "string"}, low_memory=False)
    labels = pd.read_csv(labels_path, dtype={"case_id": "string"}, low_memory=False) if labels_path.is_file() else pd.DataFrame(columns=["case_id"])
    labels_by_case = {str(row.get("case_id")): row for _, row in labels.iterrows()}
    expected_objects = expected_scin_image_objects(cases)
    missing, corrupt = [], []
    for object_name in expected_objects:
        error = validate_image_file(local_scin_path(directory, object_name))
        if error == "missing":
            missing.append(object_name)
        elif error:
            corrupt.append({"object": object_name, "reason": error})
    rows = []
    for _, case in cases.iterrows():
        case_id = _norm(case.get("case_id"))
        label = labels_by_case.get(case_id)
        related = _norm(case.get("related_category")).upper()
        dermatologist_label = _field(
            label,
            "dermatologist_skin_condition_on_label_name",
            "dermatologist_skin_condition_label_name",
        )
        weighted_label = _field(label, "weighted_skin_condition_label")
        has_dermatologist_condition = _nonempty(dermatologist_label) or _nonempty(weighted_label)
        # SCIN's related_category is structured but self-reported.  It supports
        # a weak core negative (LOOKS_HEALTHY), focal-growth positive
        # (GROWTH_OR_MOLE), or a controlled hard negative for only clear,
        # structured diffuse categories.  A false/absent dermatologist
        # gradability flag is never interpreted as healthy.
        weak_normal = related in SCIN_CORE_NEGATIVE and not has_dermatologist_condition
        growth = related in SCIN_CORE_POSITIVE
        hard_negative = related in {"ACNE", "RASH", "PIGMENTARY_PROBLEM"}
        other_condition = bool(
            related in SCIN_OTHER_CATEGORIES
            or (related in SCIN_CORE_NEGATIVE and has_dermatologist_condition)
            or (has_dermatologist_condition and not growth)
        )
        gate_target = 0 if (weak_normal or hard_negative) else (1 if growth else None)
        gate_eligible = weak_normal or growth or hard_negative
        gate_strength = (
            "moderate" if growth and has_dermatologist_condition
            else "weak" if gate_eligible
            else None
        )
        gate_reason = (
            "SCIN structured related_category=LOOKS_HEALTHY; self-reported and no dermatologist condition label"
            if weak_normal else
            "SCIN structured related_category=GROWTH_OR_MOLE"
            if growth else
            f"SCIN structured related_category={related} as non-target hard negative"
            if hard_negative else
            "SCIN non-target/diffuse or ambiguous category preserved as OTHER and excluded from core binary gate"
            if other_condition else
            "SCIN has no structured category that justifies a binary gate target"
        )
        for image_field, image, shot_type in _scin_image_paths(directory, case):
            item = {column: None for column in MANIFEST_COLUMNS}
            item.update({
                "dataset": "SCIN", "source_dataset": "SCIN", "image_path": str(image.resolve()),
                "image_id": image.stem, "case_id": case_id or None, "patient_id": case_id or None,
                "original_label": dermatologist_label or related or None,
                "lesion_present": gate_target,
                "normal_skin": weak_normal, "normal_label_strength": "weak" if weak_normal else None,
                "normal_label_method": "user_reported_related_category_LOOKS_HEALTHY" if weak_normal else None,
                "gate_negative_subtype": (
                    "healthy_no_visible_lesion" if weak_normal else
                    "other_skin_condition" if hard_negative else None
                ),
                "gate_mapping_reason": gate_reason,
                "gate_label_strength": gate_strength,
                "gate_label_source": (
                    "SCIN related_category plus dermatologist condition label"
                    if growth and has_dermatologist_condition else "SCIN self-reported related_category"
                    if gate_eligible else "SCIN structured categories"
                ),
                "other_skin_condition": other_condition,
                "supported_for_lesion_detection": gate_eligible,
                "supported_for_lesion_presence": gate_eligible,
                "supported_for_diagnosis": False,
                "age_group": _field(case, "age_group"), "sex": _field(case, "sex_at_birth"),
                "anatomical_site": _scin_body_site(case),
                "skin_tone": _field(case, "fitzpatrick_skin_type"),
                "monk_skin_tone": _field(label, "monk_skin_tone_label_us", "monk_skin_tone_label_india"),
                "image_modality": "clinical",
                "label_source": "SCIN dermatologist labels" if has_dermatologist_condition else "SCIN user report",
                "ground_truth_method": "dermatologist differential" if has_dermatologist_condition else ("user_reported_related_category" if related else None),
                "self_reported_related_category": related or None,
                "dermatologist_skin_condition_label": dermatologist_label,
                "weighted_skin_condition_label": weighted_label,
                "dermatologist_fitzpatrick_skin_type": _scin_dermatologist_fitzpatrick(label),
                "image_quality_label": _norm(shot_type) or None,
            })
            rows.append(item)
    broad = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    normal = broad.loc[broad.normal_skin.fillna(False)].copy()
    conditions = broad.loc[
        broad.lesion_present.fillna(-1).eq(1)
        | broad.other_skin_condition.fillna(False)
    ].copy()
    unexpected_missing = sorted(set(missing) - SCIN_KNOWN_MISSING_OBJECTS)
    usable_images = len(expected_objects) - len(missing) - len(corrupt)
    status = "metadata_only" if usable_images == 0 else (
        "ready" if labels_path.is_file() and not unexpected_missing and not corrupt else "incomplete"
    )
    report = {
        "dataset": "SCIN", "status": status, "metadata_file": str(cases_path),
        "labels_file": str(labels_path) if labels_path.is_file() else None,
        "total_cases": len(cases), "expected_image_objects": len(expected_objects),
        "valid_images": usable_images, "downloaded_images": usable_images,
        "eligible_clinical_images": len(broad), "weak_normal_images": len(normal),
        "condition_images": len(conditions), "missing_images": sorted(missing),
        "known_missing_images": sorted(set(missing) & SCIN_KNOWN_MISSING_OBJECTS),
        "unexpected_missing_images": unexpected_missing, "corrupt_images": corrupt,
        "invalid_images": corrupt,
        "available_metadata": [
            column for column in (
                "age_group", "sex_at_birth", "fitzpatrick_skin_type", "body_parts_*",
                "related_category", "dermatologist_skin_condition_label",
                "weighted_skin_condition_label", "dermatologist_fitzpatrick_skin_type",
                "monk_skin_tone",
            )
        ],
    }
    return broad, normal, conditions, report

def _clinical_from_metadata(dataset: str, row: pd.Series | None) -> bool:
    """Require authoritative modality metadata for paired MILK10k images.

    PAD-UFES-20 and SCIN are released as ordinary clinical photographs.  The
    Fitzpatrick17k annotation file does not reliably declare modality, so its
    files are retained for provenance but excluded until the user supplies an
    official clinical-only bundle or review.
    """
    if dataset == "SCIN":
        return True
    if dataset == "PAD-UFES-20":
        # The release is clinical-only, but a local file is eligible only when
        # it joins to the official metadata rather than merely sharing a folder.
        return row is not None
    if dataset == "Fitzpatrick17k" or row is None:
        return False
    modality = next((row[c] for c in row.index if str(c).lower() in {"modality", "image_modality", "image type", "image_type"}), None)
    value = _norm(modality).lower()
    return value in {"clinical", "macro", "clinical close-up", "clinical: close-up", "close-up"}


def _declared_modality(row: pd.Series | None) -> str:
    if row is None:
        return "unknown"
    value = next(
        (row[column] for column in row.index if str(column).lower() in {"modality", "image_modality", "image type", "image_type"}),
        None,
    )
    normalized = _norm(value).lower()
    if "dermoscop" in normalized:
        return "dermoscopic"
    if "microscop" in normalized:
        return "microscopic"
    if "patholog" in normalized:
        return "pathology"
    return "clinical" if normalized in {"clinical", "macro", "clinical close-up", "clinical: close-up", "close-up"} else "unknown"


def _readiness_status(metadata_path: Path | None, valid_images: int, eligible_images: int) -> str:
    """Only report ready when usable files exist for a supported task."""
    if valid_images == 0:
        return "metadata_only" if metadata_path is not None else "incomplete"
    return "ready" if eligible_images > 0 else "incomplete"

def _field(row: pd.Series | None, *names):
    if row is None:
        return None
    col = _column(pd.DataFrame([row]), names)
    return None if col is None else (None if not _norm(row[col]) else row[col])


def _pad_symptoms(row: pd.Series | None) -> str | None:
    if row is None:
        return None
    present = [
        name for name in ("itch", "grew", "hurt", "changed", "bleed", "elevation")
        if name in row.index and _bool(row[name])
    ]
    return "; ".join(present) if present else None


def _build_manual_gate_dataset(raw_root: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    """Ingest an approved local ImageQX or Muhaba bundle by explicit labels only.

    The importer intentionally requires a metadata table with a recognized
    image identifier and source-label column.  It never infers labels from
    directory names or image content.
    """
    directory = raw_root / dataset
    manual_status = "manual_access_required"
    if not directory.exists():
        return empty_manifest(), {
            "dataset": dataset, "status": manual_status, "valid_images": 0,
            "eligible_clinical_images": 0, "invalid_images": [],
        }
    metadata_path = _metadata_file(directory)
    images, invalid = _valid_images(directory)
    if metadata_path is None:
        substantive = any(
            path.is_file() and path.name not in {"PROVENANCE.json", "REQUIRES_MANUAL_ACCESS.md"}
            for path in directory.rglob("*")
        )
        return empty_manifest(), {
            "dataset": dataset,
            "status": "local_files_require_schema_review" if substantive else manual_status,
            "valid_images": len(images), "eligible_clinical_images": 0,
            "invalid_images": invalid,
        }

    metadata = pd.read_csv(metadata_path, low_memory=False)
    id_col = _column(
        metadata,
        ("image_id", "image", "image_name", "filename", "file_name", "img_id"),
    )
    label_col = _column(
        metadata,
        ("label", "class", "category", "diagnosis", "diagnostic", "skin_condition"),
    )
    if id_col is None or label_col is None:
        return empty_manifest(), {
            "dataset": dataset, "status": "local_files_require_schema_review",
            "metadata_file": str(metadata_path), "valid_images": len(images),
            "eligible_clinical_images": 0, "invalid_images": invalid,
            "schema_error": "recognized image-id and source-label columns are required",
        }
    metadata_by_id = {
        Path(_norm(row[id_col])).stem: row for _, row in metadata.iterrows()
        if _norm(row[id_col])
    }
    rows = []
    recognized = 0
    for image in images:
        source = metadata_by_id.get(image.stem)
        if source is None:
            continue
        original = _norm(source[label_col])
        normalized = original.lower().replace("_", "-").replace("-", " ")
        item = {column: None for column in MANIFEST_COLUMNS}
        item.update({
            "dataset": dataset, "source_dataset": dataset,
            "image_path": str(image.resolve()), "image_id": image.stem,
            "patient_id": _field(source, "patient_id", "patient", "participant_id", "user_id"),
            "case_id": _field(source, "case_id", "case", "consultation_id"),
            "lesion_id": _field(source, "lesion_id", "lesion"),
            "original_label": original or None,
            "supported_for_diagnosis": False,
            "age": _field(source, "age", "patient_age", "user_age"),
            "sex": _field(source, "sex", "gender", "patient_sex"),
            "anatomical_site": _field(source, "anatomical_site", "site", "body_part", "location"),
            "skin_tone": _field(source, "skin_tone", "fitzpatrick", "fitzpatrick_skin_type"),
            "symptoms": _field(source, "symptoms", "symptom_list", "clinical_features"),
            "image_modality": "clinical",
        })
        if dataset == "ImageQX":
            decision = IMAGEQX_LABELS.get(normalized)
            if decision == "positive":
                item.update({
                    "lesion_present": 1, "normal_skin": False,
                    "other_skin_condition": False,
                    "supported_for_lesion_detection": True,
                    "supported_for_lesion_presence": True,
                    "gate_label_strength": "strong",
                    "gate_label_source": "ImageQX dermatologist plurality label",
                    "gate_mapping_reason": "ImageQX lesion class (ICD-10 lesion labels merged by the study)",
                    "label_source": "up to 12 board-certified dermatologists",
                    "ground_truth_method": "plurality label fusion",
                })
                recognized += 1
            elif decision == "negative":
                item.update({
                    "lesion_present": 0, "normal_skin": True,
                    "normal_label_strength": "moderate",
                    "normal_label_method": "ImageQX dermatologist plurality: no lesion visible",
                    "other_skin_condition": False,
                    "supported_for_lesion_detection": True,
                    "supported_for_lesion_presence": True,
                    "gate_label_strength": "moderate",
                    "gate_label_source": "ImageQX dermatologist plurality label",
                    "gate_mapping_reason": "ImageQX healthy-skin class explicitly means no visible lesion",
                    "label_source": "up to 12 board-certified dermatologists",
                    "ground_truth_method": "plurality label fusion",
                })
                recognized += 1
            elif decision == "unsupported_quality":
                item.update({
                    "image_quality_label": "poor_quality",
                    "supported_for_lesion_detection": False,
                    "supported_for_lesion_presence": False,
                    "gate_mapping_reason": "ImageQX poor-quality class has no gate target",
                    "gate_label_source": "ImageQX dermatologist plurality label",
                })
                recognized += 1
            elif decision == "unsupported_no_skin":
                item.update({
                    "image_modality": "non_skin",
                    "image_quality_label": "no_skin",
                    "supported_for_lesion_detection": False,
                    "supported_for_lesion_presence": False,
                    "gate_mapping_reason": "ImageQX no-skin class is invalid input, not a healthy-skin negative",
                    "gate_label_source": "ImageQX dermatologist plurality label",
                })
                recognized += 1
        else:
            if normalized in MUHABA_HEALTHY_LABELS:
                item.update({
                    "lesion_present": 0, "normal_skin": True,
                    "normal_label_strength": "strong",
                    "normal_label_method": "Muhaba expert-confirmed healthy class",
                    "other_skin_condition": False,
                    "supported_for_lesion_detection": True,
                    "supported_for_lesion_presence": True,
                    "gate_label_strength": "strong",
                    "gate_label_source": "Muhaba expert study label",
                    "gate_mapping_reason": "Muhaba expert-confirmed healthy-skin class",
                    "label_source": "clinical study experts",
                    "ground_truth_method": "expert-confirmed study category",
                })
                recognized += 1
            elif normalized in MUHABA_OTHER_LABELS:
                item.update({
                    "lesion_present": None, "normal_skin": False,
                    "other_skin_condition": True,
                    "supported_for_lesion_detection": False,
                    "supported_for_lesion_presence": False,
                    "gate_label_source": "Muhaba expert study label",
                    "gate_mapping_reason": "Muhaba disease category preserved as OTHER; not a focal-lesion target",
                    "label_source": "clinical study experts",
                    "ground_truth_method": "expert-confirmed study category",
                })
                recognized += 1
        rows.append(item)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    eligible_count = int(frame.supported_for_lesion_presence.fillna(False).sum()) if not frame.empty else 0
    return frame, {
        "dataset": dataset,
        "status": "ready" if recognized and len(images) else "local_files_require_schema_review",
        "metadata_file": str(metadata_path), "valid_images": len(images),
        "matched_metadata_images": len(frame), "recognized_source_labels": recognized,
        "eligible_clinical_images": eligible_count, "invalid_images": invalid,
    }


def _hard_negative_label(dataset: str, image: Path) -> tuple[str | None, str | None]:
    """Derive only documented source labels from MSLD/MCSI path conventions."""
    stem = image.stem.upper()
    parts = [part.upper().replace("-", "_").replace(" ", "_") for part in image.parts]
    if dataset == "MSLD_v2":
        code = stem.split("_")[0]
        labels = {"MKP": "Mpox", "CHP": "Chickenpox", "CWP": "Cowpox", "MSL": "Measles", "HFMD": "HFMD", "HEALTHY": "Healthy"}
        return labels.get(code), code
    labels = {"MPOX": "Mpox", "MONKEYPOX": "Mpox", "CHICKENPOX": "Chickenpox", "ACNE": "Acne", "HEALTHY": "Healthy", "NORMAL": "Healthy"}
    for part in reversed(parts):
        for token, label in labels.items():
            if token in part:
                return label, token
    return None, None


def _build_public_hard_negative_dataset(raw_root: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    """Ingest documented MSLD original files or MCSI, never offline augmentations."""
    directory = raw_root / dataset
    if not directory.exists():
        status = "kaggle_authentication_required" if dataset == "MSLD_v2" else "not_present"
        return empty_manifest(), {"dataset": dataset, "status": status, "valid_images": 0, "eligible_clinical_images": 0, "invalid_images": []}
    images, invalid = _valid_images(directory)
    if dataset == "MSLD_v2":
        images = [
            image for image in images
            if not any("AUG" in part.upper() for part in image.parts)
        ]
    # MSLD files appear in five published folds; retain a single canonical
    # source file per coded original image rather than treating fold copies as
    # independent patients/images.
    unique: dict[str, Path] = {}
    for image in images:
        key = image.stem.upper()
        unique.setdefault(key, image)
    metadata_labels: dict[str, str] = {}
    if dataset == "MCSI":
        metadata_path = directory / "metadata.csv"
        if not metadata_path.is_file():
            return empty_manifest(), {"dataset": dataset, "status": "incomplete", "valid_images": len(images), "eligible_clinical_images": 0, "invalid_images": invalid, "reason": "MCSI metadata.csv is required for source labels"}
        metadata = pd.read_csv(metadata_path)
        for _, row in metadata.iterrows():
            identifier = _norm(row.get("img_id")).removesuffix(Path(_norm(row.get("img_id"))).suffix).upper()
            diagnostic = _norm(row.get("diagnostic")).lower()
            metadata_labels[identifier] = {
                "normal": "Healthy", "healthy": "Healthy", "acne": "Acne",
                "chickenpox": "Chickenpox", "monkeypox": "Mpox", "mpox": "Mpox",
            }.get(diagnostic, "")
    rows = []
    unrecognized = 0
    for image_id, image in sorted(unique.items()):
        label = metadata_labels.get(image_id) if dataset == "MCSI" else None
        label, code = (label, None) if label else _hard_negative_label(dataset, image)
        if label is None:
            unrecognized += 1
            continue
        healthy = label.lower() == "healthy"
        patient_id = None
        if dataset == "MSLD_v2":
            match = re.match(r"^[A-Z]+_(\d+)_", image_id)
            patient_id = f"MSLD:{match.group(1)}" if match else None
        item = {column: None for column in MANIFEST_COLUMNS}
        item.update({
            "dataset": dataset, "source_dataset": dataset,
            "image_path": str(image.resolve()), "image_id": image_id,
            "patient_id": patient_id, "case_id": patient_id,
            "original_label": label, "lesion_present": 0,
            "normal_skin": healthy,
            "normal_label_strength": "moderate" if healthy else None,
            "normal_label_method": "published healthy/no-evident-symptoms class" if healthy else None,
            "gate_negative_subtype": "healthy_no_visible_lesion" if healthy else "other_skin_condition",
            "other_skin_condition": not healthy,
            "gate_mapping_reason": (
                f"{dataset} published Healthy class" if healthy else
                f"{dataset} published {label} class is a non-target hard negative"
            ),
            "gate_label_strength": "moderate",
            "gate_label_source": f"{dataset} published class label",
            "supported_for_lesion_detection": True,
            "supported_for_lesion_presence": True,
            "supported_for_diagnosis": False,
            "image_modality": "clinical",
            "label_source": f"{dataset} release",
            "ground_truth_method": "published dataset label",
            "skin_tone": None,
        })
        rows.append(item)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return frame, {
        "dataset": dataset,
        "status": "ready" if len(frame) else "incomplete",
        "valid_images": len(images), "unique_original_images": len(frame),
        "discarded_fold_or_duplicate_files": len(images) - len(unique),
        "unrecognized_label_files": unrecognized, "eligible_clinical_images": len(frame),
        "invalid_images": invalid,
        "disk_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
    }


def _build_arsenic_skin_image_bd(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    """Ingest only documented *original* ArsenicSkinImageBD photographs.

    The release's folder spelling is retained rather than silently corrected:
    ``not_infacted`` is its published healthy class and ``infacted`` is the
    arsenicosis-affected class.  The latter is preserved as an excluded OTHER
    record because its relationship to a focal-lesion target is not established.
    """
    dataset = "ArsenicSkinImageBD"
    directory = raw_root / dataset
    if not directory.exists():
        return empty_manifest(), {"dataset": dataset, "status": "not_present", "valid_images": 0, "eligible_clinical_images": 0, "invalid_images": []}
    images, invalid = _valid_images(directory)
    original, augmented, unknown = [], [], []
    for image in images:
        parts = {part.lower() for part in image.parts}
        if "augmented" in parts:
            augmented.append(image)
        elif "original" in parts:
            original.append(image)
        else:
            unknown.append(image)
    rows, healthy_count, affected_count = [], 0, 0
    for image in sorted(original):
        classes = {part.lower() for part in image.parts}
        is_healthy = "not_infacted" in classes
        is_affected = "infacted" in classes
        if not (is_healthy or is_affected):
            unknown.append(image)
            continue
        item = {column: None for column in MANIFEST_COLUMNS}
        item.update({
            "dataset": dataset, "source_dataset": dataset,
            "image_path": str(image.resolve()), "image_id": _image_id(image),
            "original_label": "not_infacted" if is_healthy else "infacted",
            "image_modality": "clinical", "label_source": "ArsenicSkinImageBD published folder class",
            "ground_truth_method": "published dataset class", "supported_for_diagnosis": False,
        })
        if is_healthy:
            healthy_count += 1
            item.update({
                "lesion_present": 0, "normal_skin": True, "other_skin_condition": False,
                "normal_label_strength": "moderate",
                "normal_label_method": "ArsenicSkinImageBD original not_infacted class",
                "gate_negative_subtype": "healthy_no_visible_lesion",
                "gate_mapping_reason": "Original ArsenicSkinImageBD not_infacted smartphone-photo class",
                "gate_label_strength": "moderate", "gate_label_source": "ArsenicSkinImageBD published original class",
                "supported_for_lesion_detection": True, "supported_for_lesion_presence": True,
            })
        else:
            affected_count += 1
            item.update({
                "lesion_present": None, "normal_skin": False, "other_skin_condition": True,
                "gate_mapping_reason": "Original arsenicosis-affected class retained as OTHER; not defensibly mapped to focal-lesion gate",
                "gate_label_source": "ArsenicSkinImageBD published original class",
                "supported_for_lesion_detection": False, "supported_for_lesion_presence": False,
            })
        rows.append(item)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return frame, {
        "dataset": dataset, "status": "ready" if len(frame) else "incomplete",
        "valid_images": len(images), "original_images": len(original),
        "original_healthy_images": healthy_count, "original_affected_images_excluded": affected_count,
        "augmented_images_excluded": len(augmented), "unknown_layout_images_excluded": len(unknown),
        "eligible_clinical_images": healthy_count, "invalid_images": invalid,
        "disk_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
    }

def build_dataset_manifest(raw_root: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    """Build one conservative dataset manifest and an ingestion report."""
    if dataset == "SCIN":
        broad, _, _, report = build_scin_manifests(raw_root)
        return broad, report
    if dataset in {"ImageQX", "Muhaba"}:
        return _build_manual_gate_dataset(raw_root, dataset)
    if dataset in {"MSLD_v2", "MCSI"}:
        return _build_public_hard_negative_dataset(raw_root, dataset)
    if dataset == "ArsenicSkinImageBD":
        return _build_arsenic_skin_image_bd(raw_root)
    if dataset == "MonkeyPox":
        return empty_manifest(), {
            "dataset": dataset, "status": "provenance_audit_required",
            "valid_images": 0, "eligible_clinical_images": 0, "invalid_images": [],
            "reason": "Curated/extended aggregate with no reliable patient grouping; not admitted automatically.",
        }
    if dataset == "ENCoDE":
        directory = raw_root / dataset
        status = "manual_access_required" if dataset in {"ImageQX", "Muhaba"} else "credentialed_access_required"
        # A locally supplied approved bundle is intentionally not guessed from
        # arbitrary CSVs.  Add a dataset-specific importer after access terms
        # and the official schema accompany the files.
        substantive_files = directory.exists() and any(
            path.is_file()
            and path.name != "PROVENANCE.json"
            and path.suffix.lower() not in {".md", ".txt"}
            for path in directory.rglob("*")
        )
        return empty_manifest(), {
            "dataset": dataset,
            "status": "local_files_require_schema_review" if substantive_files else status,
            "valid_images": 0,
            "eligible_clinical_images": 0,
            "invalid_images": [],
        }
    directory = raw_root / dataset
    if not directory.exists():
        return empty_manifest(), {"dataset": dataset, "status": "not_present", "valid_images": 0, "eligible_clinical_images": 0, "invalid_images": []}
    metadata_path = _metadata_file(directory)
    metadata = pd.read_csv(metadata_path, low_memory=False) if metadata_path else pd.DataFrame()
    id_col = _column(metadata, ("image_id", "image", "image_name", "isic_id", "filename", "file_name", "img_id"))
    metadata_by_id = {_norm(row[id_col]).removesuffix(Path(_norm(row[id_col])).suffix): row for _, row in metadata.iterrows()} if id_col else {}
    images, invalid = _valid_images(directory)
    rows = []
    excluded_dermoscopy = 0
    for image in images:
        image_id = _image_id(image)
        row = metadata_by_id.get(image_id)
        clinical = _clinical_from_metadata(dataset, row)
        excluded_dermoscopy += int(_declared_modality(row) == "dermoscopic")
        original = _field(row, "diagnosis_3", "diagnostic", "diagnosis", "dx", "label", "three_partition_label")
        item = {column: None for column in MANIFEST_COLUMNS}
        gate_reason = (
            f"{dataset} publisher-defined clinical lesion image"
            if clinical and dataset in {"PAD-UFES-20", "MILK10k"} else None
        )
        item.update({
            "dataset": dataset, "source_dataset": dataset, "image_path": str(image.resolve()), "image_id": image_id,
            "patient_id": _field(row, "patient_id", "patient", "case_id", "case"),
            "case_id": _field(row, "case_id", "case"),
            "lesion_id": _field(row, "lesion_id", "lesion"), "original_label": original,
            "lesion_present": True if clinical else None, "normal_skin": False if clinical else None,
            "normal_label_strength": None, "normal_label_method": None, "other_skin_condition": False if clinical else None,
            "gate_mapping_reason": gate_reason,
            "gate_label_strength": "strong" if gate_reason else None,
            "gate_label_source": f"{dataset} official dataset annotation" if gate_reason else None,
            "supported_for_lesion_detection": clinical, "supported_for_diagnosis": clinical and original is not None,
            "supported_for_lesion_presence": clinical,
            "age": _field(row, "age", "age_approx"), "sex": _field(row, "sex", "gender"),
            "anatomical_site": _field(row, "anatomical_site", "anatom_site_general", "location", "region"),
            "skin_tone": _field(row, "skin_tone", "fitzpatrick", "fitspatrick", "fitzpatrick_skin_type", "mst"),
            "symptoms": _pad_symptoms(row) if dataset == "PAD-UFES-20" else _field(row, "symptoms", "symptom", "clinical_features"),
            "lesion_diameter_mm": _field(row, "lesion_diameter_mm", "diameter", "diameter_1"),
            "image_modality": "clinical" if clinical else "unknown",
            "label_source": f"{dataset} official dataset annotation",
            "ground_truth_method": (
                "biopsy" if dataset == "PAD-UFES-20" and _bool(_field(row, "biopsed"))
                else "dermatologist consensus/clinical diagnosis" if dataset == "PAD-UFES-20" and clinical
                else _field(row, "ground_truth_method", "diagnosis_method", "diagnosis_confirm_type", "benign_malignant")
            ),
        })
        rows.append(item)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not frame.empty:
        pad_mapping = {
            "nev": {"harmonized_diagnosis": "nevus", "binary_target": 0},
            "sek": {"harmonized_diagnosis": "seborrheic keratosis", "binary_target": 0},
            # Actinic keratosis is a premalignant precursor and is therefore
            # kept multiclass-only rather than silently called benign/malignant.
            "ack": {"harmonized_diagnosis": "actinic keratosis", "binary_target": None},
        }
        frame = harmonize_manifest(
            frame, custom_mapping=pad_mapping if dataset == "PAD-UFES-20" else None
        )
        # MILK10k's first diagnosis tier is the publisher's explicit benign /
        # malignant classification.  It is not inferred from image content.
        if dataset == "MILK10k" and "diagnosis_1" in metadata:
            tiers = frame.image_id.map(lambda image_id: _norm(metadata_by_id.get(image_id, {}).get("diagnosis_1", "")).lower())
            explicit_target = tiers.map({"benign": 0, "malignant": 1}).astype("Int64")
            frame.loc[explicit_target.notna(), "binary_target"] = explicit_target[explicit_target.notna()]
            frame.loc[explicit_target.notna(), "binary_mapping_reason"] = "MILK10k publisher benign/malignant tier"
        frame["supported_for_diagnosis"] &= frame.binary_target.notna() | frame.harmonized_diagnosis.notna()
    eligible_images = int(frame.image_modality.eq("clinical").sum()) if not frame.empty else 0
    return frame, {
        "dataset": dataset,
        "status": _readiness_status(metadata_path, len(images), eligible_images),
        "metadata_file": str(metadata_path) if metadata_path else None,
        "valid_images": len(images),
        "eligible_clinical_images": eligible_images,
        "excluded_dermoscopy_images": excluded_dermoscopy,
        "invalid_images": invalid,
        "disk_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
    }

def build_development_manifest(project_root: str | Path = ".", datasets: Iterable[str] | None = None) -> dict:
    """Combine approved sources, make safe splits, and persist reports."""
    root = Path(project_root).resolve(); raw = root / "data" / "raw"; reports = []
    frames = []
    scin_normal, scin_conditions = empty_manifest(), empty_manifest()
    selected_datasets = tuple(datasets or DEVELOPMENT_DATASETS)
    unknown = set(selected_datasets) - set(DEVELOPMENT_DATASETS)
    if unknown:
        raise ValueError(f"Unknown development datasets: {sorted(unknown)}")
    for dataset in selected_datasets:
        if dataset == "SCIN":
            frame, scin_normal, scin_conditions, report = build_scin_manifests(raw)
        else:
            frame, report = build_dataset_manifest(raw, dataset)
        frames.append(frame); reports.append(report)
    manifest = pd.concat(frames, ignore_index=True) if any(not f.empty for f in frames) else empty_manifest()
    eligible = manifest.loc[manifest.image_modality.eq("clinical")].copy()
    if manifest.dataset.fillna("").astype(str).str.upper().eq("DDI").any():
        raise PermissionError("DDI is final external test data and cannot enter development preparation")
    missing_eligible_paths = [
        str(path) for path in eligible.image_path if not Path(str(path)).is_file()
    ]
    if missing_eligible_paths:
        raise FileNotFoundError(
            f"Eligible manifest contains {len(missing_eligible_paths)} missing image paths"
        )
    perceptual_duplicates = pd.DataFrame(
        columns=["left_index", "right_index", "distance", "possible_duplicate"]
    )
    if not eligible.empty:
        eligible = add_exact_hashes(eligible)
        eligible, perceptual_duplicates = add_perceptual_duplicate_groups(eligible)
    if len(eligible) >= 3:
        eligible = make_group_splits(eligible)
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
    # Broad SCIN data and its two explicit views make label reliability visible
    # instead of silently mixing user report with diagnostic labels.
    if not scin_normal.empty or not scin_conditions.empty:
        write_manifest(scin_normal, root / "data" / "processed" / "scin_normal_skin_manifest.csv")
        write_manifest(scin_conditions, root / "data" / "processed" / "scin_lesion_skin_condition_manifest.csv")
    duplicate_path = root / "data" / "processed" / "exact_duplicates.csv"
    (eligible.loc[eligible.exact_duplicate_flag] if "exact_duplicate_flag" in eligible else pd.DataFrame()).to_csv(duplicate_path, index=False)
    perceptual_path = root / "data" / "processed" / "perceptual_duplicates.csv"
    perceptual_duplicates.to_csv(perceptual_path, index=False)
    leakage = leakage_report(eligible) if not eligible.empty else pd.DataFrame()
    if not leakage.empty:
        raise RuntimeError("Patient/case/lesion or exact-duplicate groups cross development splits")

    def counts(frame, column):
        return frame[column].astype("string").fillna("<missing>").value_counts().to_dict() if column in frame else {}

    eligible_normal = eligible.normal_skin.fillna(False).astype(bool) if "normal_skin" in eligible else pd.Series(False, index=eligible.index)
    eligible_lesion = eligible.lesion_present.fillna(False).astype(bool) if "lesion_present" in eligible else pd.Series(False, index=eligible.index)
    duplicate_rows = eligible.loc[eligible.exact_duplicate_flag] if "exact_duplicate_flag" in eligible else pd.DataFrame()
    diagnosis_rows = eligible.loc[
        eligible.supported_for_diagnosis.fillna(False).astype(bool)
        & eligible.lesion_present.fillna(False).astype(bool)
        & eligible.binary_target.notna()
    ]
    lesion_presence_rows = eligible.loc[
        eligible.supported_for_lesion_presence.fillna(False).astype(bool)
        & eligible.lesion_present.notna()
    ].copy()
    lesion_presence_rows["lesion_present"] = (
        lesion_presence_rows.lesion_present.astype(bool).astype(int)
    )
    audit_rows = eligible.copy()
    audit_rows["gate_state"] = "excluded"
    audit_rows.loc[
        audit_rows.other_skin_condition.fillna(False).astype(bool), "gate_state"
    ] = "other"
    audit_rows.loc[
        audit_rows.index.isin(lesion_presence_rows.index)
        & audit_rows.lesion_present.fillna(False).astype(bool),
        "gate_state",
    ] = "positive"
    audit_rows.loc[
        audit_rows.index.isin(lesion_presence_rows.index)
        & audit_rows.lesion_present.fillna(True).astype(bool).eq(False),
        "gate_state",
    ] = "negative"
    positive_rows = lesion_presence_rows.loc[lesion_presence_rows.lesion_present.eq(1)].copy()
    negative_rows = lesion_presence_rows.loc[lesion_presence_rows.lesion_present.eq(0)].copy()
    healthy_negative_rows = negative_rows.loc[
        negative_rows.gate_negative_subtype.eq("healthy_no_visible_lesion")
    ].copy()
    hard_negative_rows = negative_rows.loc[
        negative_rows.gate_negative_subtype.eq("other_skin_condition")
    ].copy()
    positive_rows["source_category"] = positive_rows.original_label.fillna("<missing>")
    scin_positive = positive_rows.source_dataset.eq("SCIN")
    positive_rows.loc[scin_positive, "source_category"] = positive_rows.loc[
        scin_positive, "self_reported_related_category"
    ].fillna("<unstructured>")
    other_rows = audit_rows.loc[audit_rows.gate_state.eq("other")].copy()
    other_rows["source_category"] = other_rows.original_label.fillna(
        other_rows.self_reported_related_category
    ).fillna("<missing>")
    scin_other = other_rows.source_dataset.eq("SCIN")
    other_rows.loc[scin_other, "source_category"] = other_rows.loc[
        scin_other, "self_reported_related_category"
    ].fillna("<unstructured>")
    report = {
        "generated": date.today().isoformat(),
        "datasets": reports,
        "manifest": str(manifest_path),
        "all_rows": len(manifest),
        "eligible_development_rows": len(eligible),
        "eligible_lesion_images": int(eligible_lesion.sum()),
        "eligible_normal_skin_images": int(eligible_normal.sum()),
        "normal_skin_by_strength": eligible.loc[
            eligible_normal, "normal_label_strength"
        ].fillna("<missing>").value_counts().to_dict(),
        "normal_skin_by_source": eligible.loc[
            eligible_normal, "source_dataset"
        ].fillna(eligible.dataset).value_counts().to_dict(),
        "metadata_coverage": {
            key: int(eligible[key].notna().sum())
            for key in ("age", "age_group", "sex", "skin_tone", "monk_skin_tone", "anatomical_site")
        },
        "eligible_other_skin_condition_images": int(eligible.other_skin_condition.fillna(False).astype(bool).sum()),
        "class_distribution": counts(diagnosis_rows, "binary_target"),
        "eligible_binary_target_availability": counts(eligible, "binary_target"),
        "lesion_presence_distribution": {
            "present": int(lesion_presence_rows.lesion_present.fillna(False).astype(bool).sum()),
            "absent": int((~lesion_presence_rows.lesion_present.fillna(False).astype(bool)).sum()),
        },
        "eligible_gate_images": int(len(lesion_presence_rows)),
        "gate_class_by_source": _grouped_count_records(
            audit_rows, ("source_dataset", "gate_state")
        ),
        "negative_by_source_and_strength": _grouped_count_records(
            negative_rows, ("source_dataset", "gate_label_strength")
        ),
        "healthy_negative_by_source": _grouped_count_records(
            healthy_negative_rows, ("source_dataset", "gate_label_strength")
        ),
        "hard_negative_by_source_and_category": _grouped_count_records(
            hard_negative_rows.assign(source_category=hard_negative_rows.original_label.fillna("<missing>")),
            ("source_dataset", "source_category", "gate_label_strength")
        ),
        "positive_by_source_and_category": _grouped_count_records(
            positive_rows, ("source_dataset", "source_category")
        ),
        "other_by_source_and_category": _grouped_count_records(
            other_rows, ("source_dataset", "source_category")
        ),
        "gate_distribution_by_split": _grouped_count_records(
            audit_rows, ("split", "gate_state")
        ),
        "gate_class_ratio_positive_to_negative": (
            float(len(positive_rows) / len(negative_rows)) if len(negative_rows) else None
        ),
        "source_target_correlation": [
            {
                "source_dataset": source,
                "gate_samples": int(len(group)),
                "p_target_lesion_present": float(group.lesion_present.mean()),
            }
            for source, group in lesion_presence_rows.groupby("source_dataset", dropna=False)
        ],
        "split_counts": eligible.split.value_counts().to_dict() if "split" in eligible else {},
        "source_distribution": counts(eligible, "source_dataset"),
        "source_distribution_by_split": _grouped_count_records(
            eligible, ("source_dataset", "split")
        ),
        "class_coverage": class_coverage_report(
            diagnosis_rows, "binary_target", ("train", "validation", "test")
        ).to_dict("records") if not diagnosis_rows.empty else [],
        "lesion_presence_coverage": class_coverage_report(
            lesion_presence_rows, "lesion_present", ("train", "validation", "test")
        ).to_dict("records") if not lesion_presence_rows.empty else [],
        "excluded_dermoscopy_images": sum(
            int(dataset_report.get("excluded_dermoscopy_images", 0)) for dataset_report in reports
        ),
        "exact_duplicate_images": len(duplicate_rows),
        "exact_duplicate_groups": int(duplicate_rows.file_sha256.nunique()) if not duplicate_rows.empty else 0,
        "perceptual_duplicate_pairs": int(len(perceptual_duplicates)),
        "perceptual_duplicate_groups": int(
            eligible.duplicate_group_id.dropna().nunique()
        ) if "duplicate_group_id" in eligible else 0,
        "leakage": leakage.to_dict("records") if not leakage.empty else [],
        "integrity": {
            "eligible_paths_exist": not missing_eligible_paths,
            "ddi_rows": 0,
            "eligible_nonclinical_rows": int((~eligible.image_modality.eq("clinical")).sum()),
            "group_or_duplicate_leakage_rows": len(leakage),
            "predictive_metadata_excludes": ["dataset", "source_dataset"],
        },
    }
    report = _json_safe(report)
    (root / "data" / "processed" / "preparation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report

def write_provenance(
    raw_root: Path,
    dataset: str,
    downloaded_files: list[str] | None = None,
    manual_required: bool = False,
    acquisition_report: dict | None = None,
) -> Path:
    """Write source, license, download, and exclusion provenance metadata."""
    folder = raw_root / dataset; folder.mkdir(parents=True, exist_ok=True)
    source = SOURCES[dataset]
    payload = {
        "dataset": dataset,
        "official_source_url": source["url"],
        "citation_or_doi": source["doi"],
        "license_or_terms": source["terms"],
        "download_date": date.today().isoformat(),
        "downloaded_files": downloaded_files or [],
        "extraction_method": (
            "official dx-scin-public-data GCS objects" if dataset == "SCIN" and acquisition_report
            else "official Mendeley Data archive" if dataset == "PAD-UFES-20" and acquisition_report
            else "not performed by default"
        ),
        "exclusions": "MILK10k dermoscopy excluded; unknown modality is never eligible",
        "known_missing_files": acquisition_report.get("known_missing_images", []) if acquisition_report else [],
        "manual_access_required": manual_required,
    }
    if acquisition_report:
        payload["acquisition_counts"] = {
            key: acquisition_report[key]
            for key in (
                "cases", "expected_image_objects", "reused_valid_images",
                "downloaded_images", "usable_images", "valid_images",
                "downloaded_bytes",
            )
            if key in acquisition_report
        }
    path = folder / "PROVENANCE.json"; path.write_text(json.dumps(payload, indent=2), encoding="utf-8"); return path
