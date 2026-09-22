"""Regression coverage for the isolated v2 DDI post-hoc runner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import run_ddi_v2_posthoc_evaluation as runner
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble


def _paths(tmp_path: Path):
    v1 = tmp_path / "final_evaluation" / "ddi"; v1.mkdir(parents=True)
    v1_config = tmp_path / "v1.yaml"; v1_config.write_text("version: 1", encoding="utf-8")
    v2_config = tmp_path / "v2.yaml"; v2_config.write_text("version: 2", encoding="utf-8")
    return v1, v1_config, v2_config


def test_v1_output_or_child_is_rejected(tmp_path):
    v1, v1_config, v2_config = _paths(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        runner.guard_paths(output=v1, v1_output=v1, v1_config=v1_config, v2_config=v2_config)
    with pytest.raises(ValueError, match="outside"):
        runner.guard_paths(output=v1 / "v2", v1_output=v1, v1_config=v1_config, v2_config=v2_config)


def test_completed_posthoc_output_is_never_overwritten(tmp_path):
    v1, v1_config, v2_config = _paths(tmp_path)
    output = tmp_path / "posthoc"; output.mkdir()
    (output / "EVALUATION_COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.guard_paths(output=output, v1_output=v1, v1_config=v1_config, v2_config=v2_config)


def test_ddi_no_metadata_excludes_multimodal_and_averages_two_models():
    config = yaml.safe_load(runner.DEFAULT_V2_CONFIG.read_text(encoding="utf-8"))
    decision = AdaptiveEnsemble(config).combine(
        {"convnext": 0.2, "efficientnet": 0.8}, metadata={"age": None, "sex": None, "anatomical_site": None}
    )
    assert decision.active_models == ("convnext", "efficientnet")
    assert decision.inactive_models == ("multimodal",)
    assert decision.malignant_probability == pytest.approx(0.5)


def test_prediction_output_schema_marks_multimodal_inactive():
    reference = {"targets": np.asarray([0, 1]), "metadata": [{"image_id": "a", "image_path": "a.jpg"}, {"image_id": "b", "image_path": "b.jpg"}]}
    individual = {"convnext": {"probabilities": np.asarray([[.8, .2], [.1, .9]])}, "efficientnet": {"probabilities": np.asarray([[.7, .3], [.4, .6]])}}
    result = runner._prediction_csv(reference, individual, np.asarray([.25, .75]), .51)
    assert list(result.columns) == ["ddi_id", "filename", "ground_truth", "convnext_malignant_probability", "efficientnet_malignant_probability", "multimodal_status", "multimodal_malignant_probability", "final_v2_malignant_probability", "predicted_class", "threshold", "correct", "confusion_outcome"]
    assert result.multimodal_status.tolist() == ["inactive", "inactive"]
    assert result.multimodal_malignant_probability.isna().all()


def test_dry_run_does_not_execute_or_create_predictions(monkeypatch, tmp_path, capsys):
    output = tmp_path / "v2"; v1 = tmp_path / "v1"; v1.mkdir()
    plan = {"policy": {"strategy": "equal_probability_average"}, "threshold": .51}
    monkeypatch.setattr(runner, "preflight", lambda **_: plan)
    monkeypatch.setattr(runner, "execute", lambda **_: pytest.fail("dry run must not execute models"))
    monkeypatch.setattr("sys.argv", ["runner", "--dry-run", "--output", str(output), "--v1-output", str(v1)])
    runner.main()
    assert not output.exists()
    assert "No inference performed" in capsys.readouterr().out


def test_missing_confirmation_blocks_execution(monkeypatch, tmp_path):
    output = tmp_path / "v2"; v1 = tmp_path / "v1"; v1.mkdir()
    plan = {"policy": {"strategy": "equal_probability_average"}, "threshold": .51}
    monkeypatch.setattr(runner, "preflight", lambda **_: plan)
    monkeypatch.setattr(runner, "execute", lambda **_: pytest.fail("confirmation is required"))
    monkeypatch.setattr("sys.argv", ["runner", "--output", str(output), "--v1-output", str(v1)])
    with pytest.raises(SystemExit):
        runner.main()
    assert not output.exists()
