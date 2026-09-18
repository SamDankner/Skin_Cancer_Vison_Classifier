from __future__ import annotations

import importlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _references_ddi_in_development_data(config):
    data_fields = {"dataset", "datasets", "data_dir", "data_dirs", "data_path", "data_paths", "manifest", "manifest_path", "source", "sources"}
    def is_data_reference_field(field):
        if field is None:
            return False
        name = field.lower()
        return name in data_fields or "dataset" in name or "manifest" in name or "source" in name or name.endswith(("_path", "_paths", "_dir", "_dirs"))
    def inspect(value, field=None):
        if isinstance(value, dict):
            return any(inspect(nested, key) for key, nested in value.items())
        if isinstance(value, (list, tuple)):
            return any(inspect(item, field) for item in value)
        return is_data_reference_field(field) and str(value).upper() == "DDI"
    return inspect(config)


def test_configs_reference_supported_strategy_names_and_safe_datasets():
    expected = {"efficientnet.yaml": "efficientnet", "efficientnet_crop.yaml": "efficientnet", "efficientnet_full_plus_crop.yaml": "efficientnet", "convnext.yaml": "convnext", "dinov2.yaml": "dinov2", "lesion_presence.yaml": "lesion_presence", "multimodal.yaml": "multimodal"}
    for name, strategy in expected.items():
        config = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert config["strategy_name"] == strategy
        assert config["task"] in {"lesion_presence", "diagnosis_binary", "diagnosis_multiclass", "image_quality"}
        assert isinstance(config.get("backbone"), str)
        assert not _references_ddi_in_development_data(config)
    ensemble = yaml.safe_load((ROOT / "configs" / "ensemble.yaml").read_text(encoding="utf-8"))
    assert ensemble["model"] == "ensemble"
    assert ensemble["prediction_split"] == "validation"
    assert ensemble["weighted_method"]["fit_split"] == "validation"
    assert ensemble["threshold"]["fit_split"] == "validation"
    assert ensemble["calibration"]["fit_split"] == "validation"
    assert not _references_ddi_in_development_data(ensemble)


def test_production_modules_use_shared_interfaces_and_portable_paths():
    production_files = list((ROOT / "src").rglob("*.py"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in production_files)
    assert "from src.data.datasets import ManifestImageDataset" in source
    assert "from src.evaluation.metrics import classification_metrics" in source
    assert "persist_run" in source
    assert "C:\\\\Users\\" not in source
    assert "cuda:0" not in source and ".cuda(0)" not in source
    assert "models" in source and "results" in source


def test_strategy_modules_import_without_starting_a_run_or_loading_weights(monkeypatch):
    import torch

    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("weights must not load at import time")))
    for name in (
        "src.strategies.cnn_common", "src.strategies.efficientnet_strategy",
        "src.strategies.convnext_strategy", "src.strategies.lesion_presence_strategy",
        "src.strategies.dinov2_strategy", "src.strategies.multimodal_strategy",
        "src.strategies.localization",
    ):
        importlib.import_module(name)
