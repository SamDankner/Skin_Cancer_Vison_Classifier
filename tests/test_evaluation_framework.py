from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import TinyImageModel
from src.evaluation.calibration import apply_temperature, fit_temperature, select_binary_threshold
from src.evaluation.ensemble import EnsembleMember, equal_weight_ensemble, fit_validation_weights
from src.evaluation.evaluator import (
    PREDICTION_COLUMNS,
    evaluate_model,
    export_predictions,
    freeze_final_configuration,
    load_checkpoint_bundle,
    prediction_frame,
)
from src.evaluation.metrics import classification_metrics


def _member(name, probabilities, task="diagnosis_binary", order=("0", "1")):
    return EnsembleMember(name, task, order, np.asarray(probabilities, dtype=float), ("a", "b", "c", "d"))


def test_equal_weight_ensemble_and_strict_compatibility():
    first = _member("first", [[.9, .1], [.2, .8], [.7, .3], [.1, .9]])
    second = _member("second", [[.7, .3], [.4, .6], [.5, .5], [.3, .7]])
    probabilities, details = equal_weight_ensemble([first, second])
    assert np.allclose(probabilities, (first.probabilities + second.probabilities) / 2)
    assert details["weights"] == [.5, .5]
    with pytest.raises(ValueError, match="class order"):
        equal_weight_ensemble([first, _member("bad", second.probabilities[:, ::-1], order=("1", "0"))])
    with pytest.raises(ValueError, match="incompatible tasks"):
        equal_weight_ensemble([first, _member("bad", second.probabilities, task="lesion_presence")])


def test_validation_weights_prefer_better_predictions_and_reject_test_fit():
    targets = np.array([0, 1, 0, 1])
    good = _member("good", [[.95, .05], [.05, .95], [.9, .1], [.1, .9]])
    poor = _member("poor", [[.1, .9], [.8, .2], [.2, .8], [.7, .3]])
    fitted = fit_validation_weights([good, poor], targets, split="validation", iterations=300)
    assert fitted["weights"][0] > fitted["weights"][1]
    assert np.isclose(sum(fitted["weights"]), 1.0)
    with pytest.raises(ValueError, match="validation"):
        fit_validation_weights([good, poor], targets, split="test")


def test_missing_metadata_member_is_explicitly_renormalized():
    image = _member("image", [[.8, .2], [.2, .8], [.6, .4], [.1, .9]])
    multimodal = _member("metadata", [[.6, .4], [np.nan, np.nan], [.4, .6], [np.nan, np.nan]])
    probabilities, details = equal_weight_ensemble([image, multimodal])
    assert np.allclose(probabilities[1], image.probabilities[1])
    assert details["available_member_count"] == [2, 1, 2, 1]


def test_threshold_selection_is_validation_only_and_supports_constraint():
    targets = [0, 0, 1, 1]
    scores = [.1, .6, .7, .9]
    selected = select_binary_threshold(targets, scores, split="validation", objective="sensitivity_constrained", minimum_sensitivity=1.0)
    assert selected["metrics"]["sensitivity"] == 1.0
    with pytest.raises(ValueError, match="validation"):
        select_binary_threshold(targets, scores, split="test")


def test_temperature_fit_and_apply_are_separate_and_validation_only():
    logits = np.array([[3., 0.], [0., 3.], [2., 0.], [0., 2.]])
    targets = np.array([0, 1, 0, 1])
    calibration, comparison = fit_temperature(logits, targets, class_order=["0", "1"], split="validation", max_iter=20)
    applied = calibration.apply(logits)
    assert applied.shape == logits.shape
    assert np.allclose(applied.sum(axis=1), 1.0)
    assert set(comparison) == {"before", "after", "recommended"}
    with pytest.raises(ValueError, match="validation"):
        fit_temperature(logits, targets, class_order=["0", "1"], split="test")
    assert np.allclose(apply_temperature(applied, 1.0, input_type="probabilities"), applied)


def test_missing_class_is_reported_and_auc_is_not_fabricated():
    metrics = classification_metrics([0, 0], [0, 0], [[.9, .1], [.8, .2]], labels=[0, 1])
    assert metrics["class_coverage_complete"] is False
    assert metrics["missing_classes"] == [1]
    assert metrics["roc_auc"] is None
    assert metrics["sensitivity"] is None


def test_ddi_requires_consent_and_frozen_configuration(manifest_frame, tmp_path):
    ddi = manifest_frame.iloc[[0]].copy(); ddi["dataset"] = "DDI"
    loader = SimpleNamespace(dataset=SimpleNamespace(frame=ddi))
    with pytest.raises(PermissionError, match="allow_final_test=True"):
        evaluate_model(torch.nn.Identity(), loader)
    with pytest.raises(FileNotFoundError, match="frozen"):
        evaluate_model(torch.nn.Identity(), loader, allow_final_test=True, frozen_config_path=tmp_path / "missing.yaml")


def test_freeze_hashes_checkpoints_and_refuses_overwrite(tmp_path):
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "frozen.yaml"
    configuration = {
        "task": "diagnosis_binary", "class_order": ["0", "1"],
        "models": [{"name": "m", "strategy": "efficientnet", "checkpoint": str(checkpoint)}],
        "preprocessing": {"image_resolution": 224, "metadata_fields": []},
        "ensemble": {"method": "equal_weight_probability_average", "weights": [1.0]},
        "threshold": {"threshold": .5, "fit_split": "validation"},
        "calibration": {"enabled": False, "fit_split": "validation"},
        "dataset_mappings": {"development": "persisted_manifest_labels"},
    }
    freeze_final_configuration(configuration, output)
    assert "checkpoint_sha256" in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze_final_configuration(configuration, output)


def test_prediction_export_schema(tmp_path):
    frame = prediction_frame([0], [[.8, .2]], [{"image_id": "x", "patient_id": "p", "dataset": "SYNTHETIC", "split": "validation"}], class_order=["0", "1"], task="diagnosis_binary", strategy="tiny")
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert json.loads(frame.loc[0, "class_probabilities"]) == {"0": .8, "1": .2}
    destination = export_predictions(frame, tmp_path / "predictions.csv")
    assert destination.is_file()


def test_checkpoint_loader_uses_persisted_task_and_class_order(monkeypatch, tmp_path):
    import src.strategies.cnn_common as cnn_common

    model = TinyImageModel(2)
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"strategy_name": "efficientnet", "task": "diagnosis_binary", "backbone": "efficientnet_v2_s", "input_mode": "full_image"}, "strategy": "efficientnet", "task": "diagnosis_binary", "architecture": "efficientnet_v2_s", "class_names": ["0", "1"]}, checkpoint)
    monkeypatch.setattr(cnn_common, "build_diagnostic_input_model", lambda *_args, **_kwargs: TinyImageModel(2))
    bundle = load_checkpoint_bundle(checkpoint, device=torch.device("cpu"))
    assert bundle.task == "diagnosis_binary"
    assert bundle.class_order == ["0", "1"]
    assert bundle.model(torch.randn(1, 3, 16, 16)).shape == (1, 2)
