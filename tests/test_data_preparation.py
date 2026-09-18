"""Tests for conservative clinical-photo manifest preparation."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from PIL import Image

from src.data.preparation import build_dataset_manifest
from src.data.datasets import MANIFEST_COLUMNS, select_task_manifest, validate_manifest


def test_milk_manifest_excludes_dermoscopy_using_metadata(tmp_path):
    directory = tmp_path / "MILK10k"
    directory.mkdir()
    pd.DataFrame([
        {"isic_id": "clinical", "image_type": "clinical", "lesion_id": "L1", "diagnosis_1": "Benign", "diagnosis_3": "Nevus"},
        {"isic_id": "dermoscopy", "image_type": "dermoscopic", "lesion_id": "L1", "diagnosis_1": "Benign", "diagnosis_3": "Nevus"},
    ]).to_csv(directory / "metadata.csv", index=False)
    Image.new("RGB", (4, 4)).save(directory / "clinical.jpg")
    Image.new("RGB", (4, 4)).save(directory / "dermoscopy.jpg")
    manifest, report = build_dataset_manifest(tmp_path, "MILK10k")
    assert report["eligible_clinical_images"] == 1
    assert manifest.loc[manifest.image_id.eq("clinical"), "image_modality"].item() == "clinical"
    assert not manifest.loc[manifest.image_id.eq("dermoscopy"), "supported_for_diagnosis"].item()
    assert manifest.loc[manifest.image_id.eq("clinical"), "binary_target"].item() == 0


def test_corrupt_images_are_reported(tmp_path):
    directory = tmp_path / "PAD-UFES-20"
    directory.mkdir()
    (directory / "bad.jpg").write_bytes(b"not an image")
    manifest, report = build_dataset_manifest(tmp_path, "PAD-UFES-20")
    assert manifest.empty
    assert len(report["invalid_images"]) == 1


def test_scin_healthy_is_weak_and_ungradable_alone_is_not_normal(tmp_path):
    directory = tmp_path / "SCIN"; directory.mkdir()
    Image.new("RGB", (4, 4)).save(directory / "healthy.jpg")
    Image.new("RGB", (4, 4)).save(directory / "ungradable.jpg")
    pd.DataFrame([
        {"case_id": "healthy", "related_category": "LOOKS_HEALTHY", "image_1_path": "healthy.jpg", "age_group": "AGE_30_TO_39"},
        {"case_id": "ungradable", "image_1_path": "ungradable.jpg"},
    ]).to_csv(directory / "scin_cases.csv", index=False)
    pd.DataFrame([{"case_id": "ungradable", "dermatologist_gradable_for_skin_condition_1": False}]).to_csv(directory / "scin_labels.csv", index=False)
    manifest, _ = build_dataset_manifest(tmp_path, "SCIN")
    healthy = manifest.loc[manifest.case_id.eq("healthy")].iloc[0]
    assert healthy.normal_label_strength == "weak" and pd.isna(healthy.age)
    assert healthy.age_group == "AGE_30_TO_39"
    assert not manifest.loc[manifest.case_id.eq("ungradable"), "normal_skin"].item()
    assert select_task_manifest(validate_manifest(manifest), "diagnosis_binary").empty


def test_nullable_boolean_accepts_serialized_binary_numbers_and_rejects_fraction():
    rows = []
    for index, value in enumerate((0.0, 1.0, np.float64(0.0), np.float64(1.0), np.nan)):
        row = {column: None for column in MANIFEST_COLUMNS}
        row.update({"dataset": "SYNTHETIC", "image_path": f"{index}.jpg", "image_id": str(index),
                    "image_modality": "clinical", "lesion_present": value})
        rows.append(row)
    checked = validate_manifest(pd.DataFrame(rows))
    assert checked.lesion_present.iloc[:4].tolist() == [False, True, False, True]
    assert checked.lesion_present.iloc[4] is pd.NA
    invalid = pd.DataFrame(rows[:1]); invalid.loc[0, "lesion_present"] = 0.5
    with pytest.raises(ValueError, match="invalid boolean"):
        validate_manifest(invalid)
