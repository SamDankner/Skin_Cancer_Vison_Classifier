from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.datasets import MANIFEST_COLUMNS, select_task_manifest, validate_manifest
from src.data.harmonize_labels import harmonize_label
from src.data.splits import class_coverage_report, make_group_splits, validate_split_class_coverage
from src.data.transforms import build_transforms


def test_shared_agent_one_modules_import_and_expose_interfaces():
    required = {
        "src.data.datasets": ("MANIFEST_COLUMNS", "validate_manifest", "select_task_manifest"),
        "src.data.splits": ("make_group_splits",),
        "src.data.transforms": ("build_transforms",),
        "src.evaluation.metrics": ("classification_metrics",),
        "src.training.experiment_runner": ("persist_run",),
    }
    for module_name, names in required.items():
        module = importlib.import_module(module_name)
        assert all(hasattr(module, name) for name in names)

    for module_name in (
        "src.data.harmonize_labels", "src.training.checkpointing", "src.training.trainer",
        "src.evaluation.evaluator", "src.utils.device", "src.utils.reporting", "src.utils.seed",
    ):
        importlib.import_module(module_name)


def test_normal_skin_is_not_recast_as_a_benign_diagnosis(manifest_frame):
    normal = manifest_frame.iloc[[0]].copy()
    normal["normal_skin"] = True
    normal["lesion_present"] = 0
    normal["harmonized_diagnosis"] = None
    normal["binary_target"] = None
    normal["supported_for_diagnosis"] = False
    checked = validate_manifest(normal)
    assert select_task_manifest(checked, "diagnosis_binary").empty
    assert select_task_manifest(checked, "lesion_presence")["lesion_present"].tolist() == [0]


def test_unclear_binary_diagnoses_remain_unmapped():
    decision = harmonize_label("clinically atypical pigmented lesion")
    assert decision.binary_target is None
    assert "not mapped" in decision.reason


def test_dermoscopy_and_ddi_are_protected(manifest_frame):
    dermoscopy = manifest_frame.iloc[[0]].copy()
    dermoscopy["image_modality"] = "dermoscopic"
    with pytest.raises(ValueError, match="clinical/macro"):
        validate_manifest(dermoscopy)

    ddi = manifest_frame.iloc[[0]].copy()
    ddi["dataset"] = "DDI"
    with pytest.raises(PermissionError, match="final external"):
        validate_manifest(ddi)
    assert validate_manifest(ddi, allow_final_test=True).iloc[0]["dataset"] == "DDI"


def test_patient_and_lesion_groups_do_not_cross_development_splits(manifest_frame):
    split = make_group_splits(manifest_frame, random_state=3)
    assert split.groupby("patient_id")["split"].nunique().max() == 1
    assert split.groupby("lesion_id")["split"].nunique().max() == 1
    assert class_coverage_report(split, "binary_target")["complete"].all()
    incomplete = split.copy()
    incomplete.loc[incomplete.split.eq("validation") & incomplete.binary_target.eq(1), "binary_target"] = None
    with pytest.raises(ValueError, match="class coverage"):
        validate_split_class_coverage(incomplete, "binary_target")
    with pytest.raises(ValueError, match="enough independent"):
        make_group_splits(manifest_frame.iloc[:6])


def test_transforms_shape_and_evaluation_determinism():
    image = Image.new("RGB", (80, 60), color=(64, 128, 192))
    train = build_transforms(32, training=True)
    evaluation = build_transforms(32, training=False)
    assert train(image).shape == (3, 32, 32)
    assert torch.equal(evaluation(image), evaluation(image))


def test_evaluator_requires_explicit_ddi_override():
    from src.evaluation.evaluator import evaluate_model

    with pytest.raises(PermissionError, match="allow_final_test=True"):
        evaluate_model(torch.nn.Identity(), [], dataset_name="DDI")


def test_evaluator_detects_ddi_from_a_manifest_backed_loader(manifest_frame):
    from src.evaluation.evaluator import evaluate_model

    ddi = manifest_frame.iloc[[0]].copy()
    ddi["dataset"] = "DDI"
    loader = SimpleNamespace(dataset=SimpleNamespace(frame=ddi))
    with pytest.raises(PermissionError, match="allow_final_test=True"):
        evaluate_model(torch.nn.Identity(), loader)


def test_manifest_columns_are_complete_for_synthetic_fixture(manifest_frame):
    assert list(manifest_frame.columns) == MANIFEST_COLUMNS
