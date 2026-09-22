"""Focused tests for the frozen Streamlit inference service."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest
import torch
import yaml

from src.evaluation.evaluator import CheckpointBundle
from src.inference import PredictionResult, SkinCancerPredictor, validate_uploaded_image

ROOT = Path(__file__).resolve().parents[1]


def test_final_config_preserves_exact_three_frozen_members():
    config = yaml.safe_load((ROOT / "configs" / "final_model.yaml").read_text(encoding="utf-8"))
    assert config["threshold"]["threshold"] == 0.51
    assert config["ensemble"]["member_names"] == ["convnext", "efficientnet", "multimodal"]
    assert config["ensemble"]["weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert "dinov2" not in config["ensemble"]["member_names"]
    assert next(item for item in config["models"] if item["name"] == "multimodal")["strategy"] == "multimodal"


class _FixedModel(torch.nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, *_args):
        return torch.tensor([[0.0, float(torch.logit(torch.tensor(self.probability)).item())]])


def test_predictor_uses_configured_equal_weight_ensemble(monkeypatch):
    import src.inference as inference

    members = (("convnext", "convnext", 0.8), ("efficientnet", "efficientnet", 0.5), ("multimodal", "multimodal", 0.2))
    config = {
        "status": "frozen", "task": "diagnosis_binary", "class_order": ["benign", "malignant"],
        "models": [{"name": name, "strategy": strategy} for name, strategy, _ in members],
        "ensemble": {"member_names": [name for name, _, _ in members], "weights": [1 / 3] * 3},
        "threshold": {"threshold": 0.51},
    }
    bundles = {name: CheckpointBundle(_FixedModel(value), strategy, "diagnosis_binary", ["benign", "malignant"], {"image_size": 16, "input_mode": "full_image"}, f"{name}.pt", {}) for name, strategy, value in members}
    monkeypatch.setattr(inference, "_bundle_input", lambda *_args: ((torch.zeros((1, 3, 1, 1)),), False, []))
    result = SkinCancerPredictor(config, bundles, torch.device("cpu")).predict(Image.new("RGB", (8, 8)))
    assert isinstance(result, PredictionResult)
    assert result.malignant_probability == pytest.approx(0.5)
    assert result.predicted_class == "benign"
    assert set(result.individual_models) == {"convnext", "efficientnet", "multimodal"}
    assert result.metadata_used == {"age": False, "sex": False, "anatomical_site": False}
    assert all(isinstance(value, float) and 0 <= value <= 1 for value in result.individual_models.values())


def test_v2_skips_multimodal_only_when_metadata_is_absent(monkeypatch):
    import src.inference as inference

    members = (("convnext", "convnext", 0.2), ("efficientnet", "efficientnet", 0.4), ("multimodal", "multimodal", 0.8))
    config = {
        "status": "experimental", "task": "diagnosis_binary", "class_order": ["benign", "malignant"],
        "models": [{"name": name, "strategy": strategy} for name, strategy, _ in members],
        "ensemble": {"member_names": [name for name, _, _ in members], "no_metadata_policy": {"strategy": "equal_probability_average"}, "metadata_available_policy": {"strategy": "closest_pair_consensus", "pair_max_distance": .1, "separation_ratio": 3, "outlier_weight_multiplier": .2, "weights": {"convnext": .34, "efficientnet": .33, "multimodal": .33}}},
        "threshold": {"threshold": .51},
    }
    bundles = {name: CheckpointBundle(_FixedModel(value), strategy, "diagnosis_binary", ["benign", "malignant"], {"image_size": 16, "input_mode": "full_image"}, f"{name}.pt", {}) for name, strategy, value in members}
    monkeypatch.setattr(inference, "_bundle_input", lambda *_args: ((torch.zeros((1, 3, 1, 1)),), False, []))
    predictor = SkinCancerPredictor(config, bundles, torch.device("cpu"))
    without_metadata = predictor.predict(Image.new("RGB", (8, 8)))
    with_metadata = predictor.predict(Image.new("RGB", (8, 8)), age=42)
    assert without_metadata.active_models == ("convnext", "efficientnet")
    assert without_metadata.inactive_models == ("multimodal",)
    assert set(with_metadata.active_models) == {"convnext", "efficientnet", "multimodal"}
    assert "multimodal" in with_metadata.individual_models


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_upload_validation_normalizes_supported_color_modes(mode):
    color = 128 if mode == "L" else (100, 120, 140, 200) if mode == "RGBA" else (100, 120, 140)
    image = Image.new(mode, (12, 12), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    assert validate_uploaded_image(buffer.getvalue(), "photo.png").mode == "RGB"


def test_upload_validation_rejects_bad_content_and_extensions():
    with pytest.raises(ValueError, match="decoded"):
        validate_uploaded_image(b"not an image", "photo.jpg")
    with pytest.raises(ValueError, match="Unsupported"):
        validate_uploaded_image(b"x", "photo.gif")
