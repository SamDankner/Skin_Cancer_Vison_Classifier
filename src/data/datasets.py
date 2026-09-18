"""Unified manifest and PyTorch dataset support for clinical macro photographs."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from typing import Callable
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

MANIFEST_COLUMNS = ["dataset", "image_path", "image_id", "patient_id", "lesion_id", "original_label", "harmonized_diagnosis", "binary_target", "lesion_present", "normal_skin", "image_quality_label", "localization_available", "bounding_box", "segmentation_mask_path", "supported_for_lesion_detection", "supported_for_diagnosis", "age", "sex", "anatomical_site", "skin_tone", "image_modality", "label_source", "ground_truth_method", "split"]
DATASET_SETUP = {"PAD-UFES-20": "https://data.mendeley.com/datasets/zr7vgbcyr2/1", "MILK10k": "https://doi.org/10.1038/s41597-024-03501-y", "Fitzpatrick17k": "https://github.com/mattgroh/fitzpatrick17k", "SCIN": "https://github.com/google-research-datasets/scin", "DDI": "https://stanfordaimi.github.io/digital-dermatology/"}


def _nullable_boolean(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "y": True, "n": False,
    }
    converted = values.map(
        lambda value: pd.NA if pd.isna(value) or str(value).strip() == "" else mapping.get(
            value if isinstance(value, (bool, int)) else str(value).strip().lower(), "<INVALID>"
        )
    )
    if converted.eq("<INVALID>").any():
        invalid = sorted(values.loc[converted.eq("<INVALID>")].astype(str).unique().tolist())
        raise ValueError(f"Manifest column {name!r} has invalid boolean values: {invalid}")
    return converted.astype("boolean")

def empty_manifest() -> pd.DataFrame:
    """Return an empty manifest with the complete public schema."""
    return pd.DataFrame(columns=MANIFEST_COLUMNS)

def validate_manifest(manifest: pd.DataFrame, allow_final_test: bool = False) -> pd.DataFrame:
    """Validate schema and reject non-clinical or unapproved final-test samples."""
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing: raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    result = manifest.copy()
    for column in ("lesion_present", "normal_skin", "localization_available", "supported_for_lesion_detection", "supported_for_diagnosis"):
        result[column] = _nullable_boolean(result[column], column)
    modality = result.image_modality.fillna("clinical").str.lower()
    if modality.str.contains("dermoscop|micro|patholog", regex=True).any(): raise ValueError("Only ordinary clinical/macro photographs are allowed")
    if result.dataset.fillna("").str.upper().eq("DDI").any() and not allow_final_test: raise PermissionError("DDI is final external test data; pass allow_final_test=True explicitly")
    normal = result.normal_skin.fillna(False)
    if normal.any() and result.loc[normal, "harmonized_diagnosis"].notna().any(): raise ValueError("Normal skin cannot be assigned a lesion diagnosis")
    if normal.any() and result.loc[normal, "binary_target"].notna().any(): raise ValueError("Normal skin cannot be assigned a benign/malignant target")
    return result

def write_manifest(manifest: pd.DataFrame, path: str | Path, allow_final_test: bool = False) -> Path:
    """Validate and save a manifest CSV, returning its path."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    validate_manifest(manifest, allow_final_test).to_csv(path, index=False); return path

def file_sha256(path: str | Path, chunk_size: int = 1_048_576) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""): digest.update(chunk)
    return digest.hexdigest()

def add_exact_hashes(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add exact hashes and flags; never delete questionable cases."""
    result = manifest.copy(); result["file_sha256"] = [file_sha256(p) for p in result.image_path]
    result["exact_duplicate_flag"] = result.file_sha256.duplicated(keep=False); return result

def find_perceptual_duplicates(manifest: pd.DataFrame, hasher: Callable[[str], object], distance: Callable[[object, object], float], max_distance: float = 4) -> pd.DataFrame:
    """Flag pairs using a caller-provided perceptual-hash implementation."""
    values = [(idx, hasher(path)) for idx, path in manifest.image_path.items()]; rows = []
    for pos, (left_idx, left) in enumerate(values):
        for right_idx, right in values[pos + 1:]:
            value = distance(left, right)
            if value <= max_distance: rows.append({"left_index": left_idx, "right_index": right_idx, "distance": value, "possible_duplicate": True})
    return pd.DataFrame(rows, columns=["left_index", "right_index", "distance", "possible_duplicate"])

TASK_TARGETS = {"lesion_presence": "lesion_present", "diagnosis_binary": "binary_target", "diagnosis_multiclass": "harmonized_diagnosis", "image_quality": "image_quality_label"}

def select_task_manifest(manifest: pd.DataFrame, task: str) -> pd.DataFrame:
    """Filter one manifest for a separable learning task without relabeling normal skin."""
    if task not in TASK_TARGETS: raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASK_TARGETS)}")
    result = manifest.copy(); target = TASK_TARGETS[task]
    if task == "lesion_presence":
        result = result.loc[_nullable_boolean(result.supported_for_lesion_detection, "supported_for_lesion_detection").fillna(False)]
    elif task.startswith("diagnosis"):
        supported = _nullable_boolean(result.supported_for_diagnosis, "supported_for_diagnosis").fillna(False)
        lesion = _nullable_boolean(result.lesion_present, "lesion_present").fillna(False)
        result = result.loc[supported & lesion]
    return result.loc[result[target].notna()].copy()

class ManifestImageDataset(Dataset):
    """Load RGB macro photographs for lesion presence, diagnosis, or quality tasks."""
    def __init__(self, manifest: pd.DataFrame, split: str | None = None, transform: Callable | None = None, task: str = "diagnosis_binary", target_column: str | None = None, allow_final_test: bool = False):
        frame = validate_manifest(manifest, allow_final_test)
        frame = select_task_manifest(frame, task)
        self.frame = frame if split is None else frame.loc[frame.split.eq(split)].reset_index(drop=True)
        self.transform, self.target_column = transform, target_column or TASK_TARGETS[task]
    def __len__(self): return len(self.frame)
    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source: image = source.convert("RGB")
        if self.transform: image = self.transform(image)
        target = row[self.target_column]
        target = None if pd.isna(target) else (str(target) if self.target_column == "harmonized_diagnosis" else int(target))
        return {"image": image, "target": target, "metadata": row.to_dict()}
