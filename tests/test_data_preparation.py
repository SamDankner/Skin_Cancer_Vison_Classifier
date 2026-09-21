"""Tests for conservative clinical-photo manifest preparation."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote
from zipfile import ZipFile

import pandas as pd
import numpy as np
import pytest
from PIL import Image

from src.data.preparation import _json_safe, build_dataset_manifest, build_development_manifest
from src.data.datasets import MANIFEST_COLUMNS, select_task_manifest, validate_manifest
from src.data.scin_download import SCIN_BUCKET_URL, SCIN_KNOWN_MISSING_OBJECTS, download_scin_dataset


class _DownloadResponse(BytesIO):
    status = 200

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


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


def test_development_report_grouped_counts_are_json_records(tmp_path):
    directory = tmp_path / "data" / "raw" / "PAD-UFES-20"
    directory.mkdir(parents=True)
    metadata = []
    for index in range(10):
        Image.new("RGB", (4, 4), color=(index * 20, 0, 0)).save(directory / f"image-{index}.png")
        metadata.append({
            "img_id": f"image-{index}", "patient_id": f"patient-{index}",
            "lesion_id": f"lesion-{index}", "diagnostic": "NEV",
        })
    pd.DataFrame(metadata).to_csv(directory / "metadata.csv", index=False)

    report = build_development_manifest(tmp_path, datasets=["PAD-UFES-20"])
    report_path = tmp_path / "data" / "processed" / "preparation_report.json"
    with report_path.open(encoding="utf-8") as handle:
        reloaded = json.load(handle)

    assert reloaded == report
    assert reloaded["source_distribution_by_split"]
    assert all(
        set(record) == {"source_dataset", "split", "count"}
        and record["source_dataset"] == "PAD-UFES-20"
        and isinstance(record["count"], int)
        for record in reloaded["source_distribution_by_split"]
    )


def test_report_json_conversion_preserves_types_and_rejects_structural_keys():
    report = {
        "integer": np.int64(7),
        "floating": np.float64(2.5),
        "boolean": np.bool_(True),
        "missing": [pd.NA, np.nan, pd.NaT],
        "path": Path("data/processed"),
        "timestamp": pd.Timestamp("2026-09-18T12:30:00"),
        "set_values": {"weak", "strong"},
        "tuple_values": (1, 2),
    }

    reloaded = json.loads(json.dumps(_json_safe(report), allow_nan=False))

    assert reloaded["integer"] == 7 and isinstance(reloaded["integer"], int)
    assert reloaded["floating"] == 2.5 and isinstance(reloaded["floating"], float)
    assert reloaded["boolean"] is True
    assert reloaded["missing"] == [None, None, None]
    assert reloaded["path"] == str(Path("data/processed"))
    assert reloaded["timestamp"] == "2026-09-18T12:30:00"
    assert reloaded["set_values"] == ["strong", "weak"]
    assert reloaded["tuple_values"] == [1, 2]
    with pytest.raises(TypeError, match="non-string dictionary key"):
        _json_safe({"grouped": {("SCIN", "weak"): np.int64(123)}})


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


def test_zero_image_datasets_are_not_ready(tmp_path):
    pad = tmp_path / "PAD-UFES-20"
    pad.mkdir()
    _, pad_report = build_dataset_manifest(tmp_path, "PAD-UFES-20")
    assert pad_report["valid_images"] == 0
    assert pad_report["status"] != "ready"

    fitzpatrick = tmp_path / "Fitzpatrick17k"
    fitzpatrick.mkdir()
    pd.DataFrame([{"md5hash": "example", "label": "example"}]).to_csv(
        fitzpatrick / "fitzpatrick17k.csv", index=False
    )
    _, fitzpatrick_report = build_dataset_manifest(tmp_path, "Fitzpatrick17k")
    assert fitzpatrick_report["status"] == "metadata_only"

    encode = tmp_path / "ENCoDE"
    encode.mkdir()
    (encode / "REQUIRES_PHYSIONET_ACCESS.md").write_text("credentialed", encoding="utf-8")
    _, encode_report = build_dataset_manifest(tmp_path, "ENCoDE")
    assert encode_report["status"] == "credentialed_access_required"


def test_scin_metadata_only_and_known_missing_image_handling(tmp_path):
    directory = tmp_path / "SCIN"
    images = directory / "dataset" / "images"
    images.mkdir(parents=True)
    known_missing = next(iter(SCIN_KNOWN_MISSING_OBJECTS))
    valid_object = "dataset/images/valid-object"
    Image.new("RGB", (4, 4)).save(directory / valid_object, format="PNG")
    pd.DataFrame([
        {"case_id": "valid", "related_category": "RASH", "image_1_path": valid_object},
        {"case_id": "missing", "related_category": "RASH", "image_1_path": known_missing},
    ]).to_csv(directory / "dataset" / "scin_cases.csv", index=False)
    pd.DataFrame([{"case_id": "valid"}, {"case_id": "missing"}]).to_csv(
        directory / "dataset" / "scin_labels.csv", index=False
    )

    manifest, report = build_dataset_manifest(tmp_path, "SCIN")

    assert report["status"] == "ready"
    assert report["known_missing_images"] == [known_missing]
    assert report["unexpected_missing_images"] == []
    assert len(manifest) == 1 and Path(manifest.image_path.item()).is_file()

    (directory / valid_object).unlink()
    _, metadata_only = build_dataset_manifest(tmp_path, "SCIN")
    assert metadata_only["status"] == "metadata_only"


def test_official_scin_download_is_idempotent_and_tolerates_known_missing(tmp_path):
    known_missing = next(iter(SCIN_KNOWN_MISSING_OBJECTS))
    valid_object = "dataset/images/valid-object"
    cases = pd.DataFrame([
        {"case_id": "valid", "image_1_path": valid_object},
        {"case_id": "missing", "image_1_path": known_missing},
    ]).to_csv(index=False).encode()
    labels = pd.DataFrame([{"case_id": "valid"}, {"case_id": "missing"}]).to_csv(index=False).encode()
    image_buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(image_buffer, format="PNG")
    objects = {
        "dataset/scin_cases.csv": cases,
        "dataset/scin_labels.csv": labels,
        valid_object: image_buffer.getvalue(),
    }
    calls = []

    def opener(request, timeout):
        del timeout
        object_name = unquote(request.full_url.removeprefix(f"{SCIN_BUCKET_URL}/"))
        calls.append(object_name)
        if object_name not in objects:
            raise HTTPError(request.full_url, 404, "not found", {}, None)
        return _DownloadResponse(objects[object_name])

    first = download_scin_dataset(tmp_path, opener=opener, workers=2, progress=None)
    second = download_scin_dataset(tmp_path, opener=opener, workers=2, progress=None)

    assert first["downloaded_images"] == 1
    assert first["known_missing_images"] == [known_missing]
    assert second["reused_valid_images"] == 1
    assert second["downloaded_images"] == 0
    assert calls.count(valid_object) == 1


def test_scin_case_grouping_and_other_conditions_are_excluded_from_core_gate(tmp_path):
    directory = tmp_path / "data" / "raw" / "SCIN"
    image_directory = directory / "dataset" / "images"
    image_directory.mkdir(parents=True)
    cases = []
    labels = []
    for case_index in range(10):
        related = "LOOKS_HEALTHY" if case_index % 2 == 0 else "RASH"
        case = {"case_id": f"case-{case_index}", "related_category": related}
        for image_index in range(2):
            object_name = f"dataset/images/{case_index}-{image_index}"
            Image.new("RGB", (4, 4), color=(case_index * 20, image_index * 80, 0)).save(
                directory / object_name, format="PNG"
            )
            case[f"image_{image_index + 1}_path"] = object_name
        cases.append(case)
        labels.append({"case_id": f"case-{case_index}"})
    pd.DataFrame(cases).to_csv(directory / "dataset" / "scin_cases.csv", index=False)
    pd.DataFrame(labels).to_csv(directory / "dataset" / "scin_labels.csv", index=False)

    report = build_development_manifest(tmp_path, datasets=["SCIN"])
    manifest = pd.read_csv(report["manifest"])

    assert manifest.groupby("case_id").split.nunique().max() == 1
    assert manifest.loc[manifest.self_reported_related_category.eq("LOOKS_HEALTHY"), "lesion_present"].eq(0).all()
    rash = manifest.loc[manifest.self_reported_related_category.eq("RASH")]
    assert rash.lesion_present.eq(0).all()
    assert rash.other_skin_condition.eq(True).all()
    assert rash.supported_for_lesion_presence.eq(True).all()
    assert rash.gate_negative_subtype.eq("other_skin_condition").all()
    assert not manifest.supported_for_diagnosis.fillna(False).any()


def test_development_report_uses_only_eligible_rows_and_covers_test_split(tmp_path):
    directory = tmp_path / "data" / "raw" / "MILK10k"
    directory.mkdir(parents=True)
    metadata = []
    for index in range(10):
        diagnosis_1 = "Benign" if index % 2 == 0 else "Malignant"
        diagnosis_3 = "Nevus" if index % 2 == 0 else "Melanoma"
        for modality in ("clinical", "dermoscopic"):
            image_id = f"{modality}-{index}"
            metadata.append({
                "isic_id": image_id, "image_type": modality, "lesion_id": f"lesion-{index}",
                "diagnosis_1": diagnosis_1, "diagnosis_3": diagnosis_3,
            })
            Image.new("RGB", (4, 4), color=(index * 20, 0 if modality == "clinical" else 100, 0)).save(
                directory / f"{image_id}.png"
            )
    pd.DataFrame(metadata).to_csv(directory / "metadata.csv", index=False)

    report = build_development_manifest(tmp_path, datasets=["MILK10k"])

    assert report["all_rows"] == 20
    assert report["eligible_development_rows"] == 10
    assert sum(report["class_distribution"].values()) == 10
    assert report["lesion_presence_distribution"] == {"present": 10, "absent": 0}
    assert report["excluded_dermoscopy_images"] == 10
    assert {item["split"] for item in report["class_coverage"]} == {"train", "validation", "test"}
    assert {item["split"] for item in report["lesion_presence_coverage"]} == {"train", "validation", "test"}


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


def test_scin_growth_is_eligible_positive_but_acne_is_other(tmp_path):
    directory = tmp_path / "SCIN"
    directory.mkdir()
    for name in ("growth", "acne"):
        Image.new("RGB", (4, 4)).save(directory / f"{name}.jpg")
    pd.DataFrame([
        {"case_id": "growth", "related_category": "GROWTH_OR_MOLE", "image_1_path": "growth.jpg"},
        {"case_id": "acne", "related_category": "ACNE", "image_1_path": "acne.jpg"},
    ]).to_csv(directory / "scin_cases.csv", index=False)
    pd.DataFrame([{"case_id": "growth"}, {"case_id": "acne"}]).to_csv(
        directory / "scin_labels.csv", index=False
    )
    manifest, _ = build_dataset_manifest(tmp_path, "SCIN")
    growth = manifest.loc[manifest.case_id.eq("growth")].iloc[0]
    acne = manifest.loc[manifest.case_id.eq("acne")].iloc[0]
    assert growth.lesion_present == 1 and growth.supported_for_lesion_presence
    assert acne.lesion_present == 0 and acne.other_skin_condition
    assert acne.supported_for_lesion_presence
    assert acne.gate_negative_subtype == "other_skin_condition"


def test_pad_lesion_is_strong_gate_positive_and_preserves_groups(tmp_path):
    directory = tmp_path / "PAD-UFES-20"
    directory.mkdir()
    Image.new("RGB", (4, 4)).save(directory / "img-1.png")
    pd.DataFrame([{
        "img_id": "img-1", "patient_id": "PAT-1", "lesion_id": "LES-1",
        "diagnostic": "NEV", "age": 44, "region": "ARM", "fitspatrick": 3,
        "diameter_1": 7.0,
    }]).to_csv(directory / "metadata.csv", index=False)
    manifest, _ = build_dataset_manifest(tmp_path, "PAD-UFES-20")
    row = manifest.iloc[0]
    assert row.lesion_present == 1 and row.supported_for_lesion_presence
    assert row.gate_label_strength == "strong"
    assert row.patient_id == "PAT-1" and row.lesion_id == "LES-1"
    assert row.lesion_diameter_mm == 7.0


def test_imageqx_explicit_gate_mapping_and_unsupported_classes(tmp_path):
    directory = tmp_path / "ImageQX"
    directory.mkdir()
    labels = ["lesion", "healthy skin", "poor quality", "no skin"]
    for index in range(4):
        Image.new("RGB", (4, 4), color=(index * 30, 0, 0)).save(directory / f"img-{index}.png")
    pd.DataFrame([
        {"image_id": f"img-{index}", "label": label, "user_id": f"user-{index}"}
        for index, label in enumerate(labels)
    ]).to_csv(directory / "metadata.csv", index=False)
    manifest, report = build_dataset_manifest(tmp_path, "ImageQX")
    assert report["status"] == "ready"
    assert manifest.loc[manifest.original_label.eq("lesion"), "lesion_present"].item() == 1
    healthy = manifest.loc[manifest.original_label.eq("healthy skin")].iloc[0]
    assert healthy.lesion_present == 0 and healthy.normal_skin
    unsupported = manifest.loc[manifest.original_label.isin(["poor quality", "no skin"])]
    assert unsupported.lesion_present.isna().all()
    assert not unsupported.supported_for_lesion_presence.any()
    assert manifest.loc[manifest.original_label.eq("no skin"), "image_modality"].item() == "non_skin"


def test_muhaba_healthy_is_negative_and_diseases_remain_other(tmp_path):
    directory = tmp_path / "Muhaba"
    directory.mkdir()
    for index in range(2):
        Image.new("RGB", (4, 4), color=(index * 30, 0, 0)).save(directory / f"img-{index}.png")
    pd.DataFrame([
        {"image_id": "img-0", "diagnosis": "Healthy skin", "patient_id": "P1"},
        {"image_id": "img-1", "diagnosis": "Atopic dermatitis", "patient_id": "P2"},
    ]).to_csv(directory / "metadata.csv", index=False)
    manifest, _ = build_dataset_manifest(tmp_path, "Muhaba")
    healthy, disease = manifest.iloc[0], manifest.iloc[1]
    assert healthy.lesion_present == 0 and healthy.normal_skin
    assert disease.other_skin_condition and pd.isna(disease.lesion_present)
    assert not disease.supported_for_lesion_presence


def test_mcsi_metadata_maps_healthy_and_hard_negatives(tmp_path):
    directory = tmp_path / "MCSI" / "images"
    directory.mkdir(parents=True)
    for image_id in ("normal_1", "acne_1", "monkeypox_1", "chickenpox_1"):
        Image.new("RGB", (4, 4)).save(directory / f"{image_id}.png")
    pd.DataFrame([
        {"img_id": "normal_1.png", "diagnostic": "normal"},
        {"img_id": "acne_1.png", "diagnostic": "acne"},
        {"img_id": "monkeypox_1.png", "diagnostic": "monkeypox"},
        {"img_id": "chickenpox_1.png", "diagnostic": "chickenpox"},
    ]).to_csv(directory.parent / "metadata.csv", index=False)
    manifest, report = build_dataset_manifest(tmp_path, "MCSI")
    assert report["status"] == "ready" and len(manifest) == 4
    healthy = manifest.loc[manifest.original_label.eq("Healthy")].iloc[0]
    assert healthy.lesion_present == 0 and healthy.normal_skin
    assert healthy.gate_negative_subtype == "healthy_no_visible_lesion"
    hard = manifest.loc[manifest.original_label.ne("Healthy")]
    assert hard.lesion_present.eq(0).all() and hard.other_skin_condition.all()
    assert hard.gate_negative_subtype.eq("other_skin_condition").all()


def test_msld_uses_only_original_images_and_preserves_filename_patient_group(tmp_path):
    original = tmp_path / "MSLD_v2" / "Original Images" / "FOLDS" / "fold_1"
    augmented = tmp_path / "MSLD_v2" / "Augmented Images" / "FOLDS_AUG" / "fold_1"
    original.mkdir(parents=True)
    augmented.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(original / "HEALTHY_17_001.jpg")
    Image.new("RGB", (4, 4)).save(original / "MKP_18_001.jpg")
    Image.new("RGB", (4, 4)).save(augmented / "MKP_18_001_AUG.jpg")

    manifest, report = build_dataset_manifest(tmp_path, "MSLD_v2")

    assert report["status"] == "ready"
    assert len(manifest) == 2
    healthy = manifest.loc[manifest.original_label.eq("Healthy")].iloc[0]
    mpox = manifest.loc[manifest.original_label.eq("Mpox")].iloc[0]
    assert healthy.patient_id == "MSLD:17"
    assert healthy.lesion_present == 0
    assert healthy.gate_negative_subtype == "healthy_no_visible_lesion"
    assert mpox.patient_id == "MSLD:18"
    assert mpox.lesion_present == 0
    assert mpox.gate_negative_subtype == "other_skin_condition"


def test_arsenic_uses_original_healthy_only_and_excludes_augmentations(tmp_path):
    original_healthy = tmp_path / "ArsenicSkinImageBD" / "Original" / "not_infacted"
    original_affected = tmp_path / "ArsenicSkinImageBD" / "Original" / "infacted"
    augmented = tmp_path / "ArsenicSkinImageBD" / "Augmented" / "not_infected"
    for directory in (original_healthy, original_affected, augmented):
        directory.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(original_healthy / "healthy.png")
    Image.new("RGB", (4, 4)).save(original_affected / "affected.png")
    Image.new("RGB", (4, 4)).save(augmented / "healthy_augmented_1.png")

    manifest, report = build_dataset_manifest(tmp_path, "ArsenicSkinImageBD")

    assert report["original_healthy_images"] == 1
    assert report["original_affected_images_excluded"] == 1
    assert report["augmented_images_excluded"] == 1
    assert len(manifest) == 2
    healthy = manifest.loc[manifest.original_label.eq("not_infacted")].iloc[0]
    affected = manifest.loc[manifest.original_label.eq("infacted")].iloc[0]
    assert healthy.lesion_present == 0 and healthy.normal_skin
    assert healthy.gate_negative_subtype == "healthy_no_visible_lesion"
    assert healthy.gate_label_strength == "moderate"
    assert pd.isna(affected.lesion_present) and affected.other_skin_condition
    assert not affected.supported_for_lesion_presence


def test_mcvsld_rejects_web_derived_originals_and_offline_copies(tmp_path):
    root = tmp_path / "MCVSLD" / "Skin Lesion Dataset"
    healthy = root / "train" / "Healthy"
    viral = root / "train" / "Monkeypox"
    healthy.mkdir(parents=True)
    viral.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(30, 40, 50)).save(
        healthy / "HEALTHY_01_01_ORIGINAL.jpg"
    )
    Image.new("RGB", (8, 8), color=(35, 45, 55)).save(
        healthy / "HEALTHY_01_01_1.jpg"
    )
    Image.new("RGB", (8, 8), color=(80, 20, 20)).save(
        viral / "MKP_01_01_ORIGINAL.jpg"
    )

    manifest, report = build_dataset_manifest(tmp_path, "MCVSLD")

    assert manifest.empty
    assert report["status"] == "rejected_provenance_and_augmentation"
    assert report["original_images"] == 2
    assert report["original_healthy_images_excluded"] == 1
    assert report["augmented_or_derived_images_excluded"] == 1
    assert report["eligible_clinical_images"] == 0


def test_monkeypox_aggregate_rejects_augments_and_resized_mcsi_copies(tmp_path):
    mcsi = tmp_path / "MCSI" / "images"
    normal = tmp_path / "MonkeyPox" / "Mpox-HSAM" / "normal"
    mcsi.mkdir(parents=True)
    normal.mkdir(parents=True)
    source = Image.new("RGB", (16, 16), color=(90, 100, 110))
    source.save(mcsi / "NORMAL_1.png")
    source.resize((8, 8)).save(normal / "normal_original_000.png")
    source.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(normal / "normal_aug_001.png")

    manifest, report = build_dataset_manifest(tmp_path, "MonkeyPox")

    assert manifest.empty
    assert report["status"] == "rejected_duplicate_aggregate"
    assert report["original_images"] == 1
    assert report["original_normal_images_excluded"] == 1
    assert report["augmented_images_excluded"] == 1
    assert report["candidate_originals_matching_mcsi"] == 1


def test_skin_disease_classification_rejects_dermoscopy_and_split_copies(tmp_path):
    directory = tmp_path / "SkinDiseaseClassification"
    directory.mkdir()
    archive = directory / "archive.zip"
    dermoscopy = BytesIO()
    Image.new("RGB", (8, 8), color=(40, 50, 60)).save(dermoscopy, format="JPEG")
    clinical = BytesIO()
    Image.new("RGB", (8, 8), color=(70, 80, 90)).save(clinical, format="JPEG")
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("Split_smol/train/Melanoma/ISIC_0000001.jpg", dermoscopy.getvalue())
        bundle.writestr("Split_smol/train/Atopic Dermatitis/1_1.jpg", clinical.getvalue())
        bundle.writestr("Split_smol/val/Atopic Dermatitis/1_1.jpg", clinical.getvalue())
        bundle.writestr(
            "Split_smol/train/Tinea Ringworm Candidiasis/aug_0_Screenshot.png",
            clinical.getvalue(),
        )

    manifest, report = build_dataset_manifest(tmp_path, "SkinDiseaseClassification")

    assert manifest.empty
    assert report["status"] == "rejected_provenance_and_modality"
    assert report["reported_healthy_images"] == 0
    assert report["dermoscopy_isic_images_excluded"] == 1
    assert report["offline_augmented_images_detected"] == 1
    assert report["screenshot_files_detected"] == 1
    assert report["exact_duplicate_images"] == 3
    assert report["eligible_clinical_images"] == 0


def test_hard_negatives_are_opt_in_for_lesion_task_selection():
    rows = []
    for index, subtype in enumerate(("healthy_no_visible_lesion", "other_skin_condition")):
        row = {column: None for column in MANIFEST_COLUMNS}
        row.update({
            "dataset": "SYNTHETIC", "image_path": f"{index}.jpg", "image_id": str(index),
            "image_modality": "clinical", "lesion_present": False,
            "normal_skin": subtype == "healthy_no_visible_lesion",
            "other_skin_condition": subtype == "other_skin_condition",
            "supported_for_lesion_presence": True,
            "gate_negative_subtype": subtype,
        })
        rows.append(row)
    manifest = validate_manifest(pd.DataFrame(rows))
    default = select_task_manifest(manifest, "lesion_presence")
    with_hard_negatives = select_task_manifest(
        manifest, "lesion_presence", include_hard_negatives=True
    )
    assert default.gate_negative_subtype.tolist() == ["healthy_no_visible_lesion"]
    assert set(with_hard_negatives.gate_negative_subtype) == {
        "healthy_no_visible_lesion", "other_skin_condition"
    }
