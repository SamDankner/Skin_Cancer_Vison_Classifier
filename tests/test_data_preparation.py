"""Tests for conservative clinical-photo manifest preparation."""
from __future__ import annotations

import pandas as pd
from PIL import Image

from src.data.preparation import build_dataset_manifest


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
