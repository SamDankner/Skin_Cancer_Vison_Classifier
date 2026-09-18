from __future__ import annotations

import pytest
import torch

from conftest import TinyDinoBackbone, TinyImageModel


def _offline_cnn_builder(calls):
    def build(architecture, num_classes, dropout=.2, pretrained=True):
        calls.append((architecture, num_classes, pretrained))
        return TinyImageModel(num_classes)
    return build


def test_efficientnet_convnext_and_lesion_presence_wire_offline_models(monkeypatch):
    from src.strategies import convnext_strategy, efficientnet_strategy, lesion_presence_strategy

    calls = []
    builder = _offline_cnn_builder(calls)
    monkeypatch.setattr(efficientnet_strategy, "build_cnn_model", builder)
    monkeypatch.setattr(convnext_strategy, "build_cnn_model", builder)
    monkeypatch.setattr(lesion_presence_strategy, "build_cnn_model", builder)
    image = torch.randn(2, 3, 32, 32)

    assert efficientnet_strategy.build_model(3, pretrained=False)(image).shape == (2, 3)
    assert convnext_strategy.build_model(3, pretrained=False)(image).shape == (2, 3)
    assert lesion_presence_strategy.build_model(pretrained=False)(image).shape == (2, 2)
    assert calls == [("efficientnet_v2_s", 3, False), ("convnext_tiny", 3, False), ("efficientnet_v2_s", 2, False)]


def test_native_efficientnet_and_convnext_accept_synthetic_image_batches():
    from src.strategies import convnext_strategy, efficientnet_strategy

    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        efficientnet = efficientnet_strategy.build_model(3, pretrained=False).eval()
        assert efficientnet(image).shape == (2, 3)
        convnext = convnext_strategy.build_model(3, pretrained=False).eval()
        assert convnext(image).shape == (2, 3)


def test_dinov2_classifier_runs_without_hub_loading():
    from src.strategies.dinov2_strategy import DinoV2Classifier

    model = DinoV2Classifier(TinyDinoBackbone(), num_classes=3, dropout=0.0, head_width=8)
    assert model(torch.randn(2, 3, 56, 56)).shape == (2, 3)


def test_multimodal_model_handles_missing_metadata_without_hub_loading():
    import pandas as pd
    from src.strategies.multimodal_strategy import MetadataPreprocessor, build_multimodal_model

    train = pd.DataFrame({"age": [20, None], "sex": ["female", None], "anatomical_site": ["arm", "leg"], "skin_tone": [None, "medium"]})
    processor = MetadataPreprocessor().fit(train)
    metadata = processor.transform(pd.DataFrame({"age": [45, None], "sex": ["female", None], "anatomical_site": ["arm", None], "skin_tone": ["unknown", None]}))
    model = build_multimodal_model({"pretrained": False, "metadata_width": 8, "fusion_width": 12, "dropout": 0.0}, processor, 2, backbone_factory=lambda *_args, **_kwargs: TinyDinoBackbone())
    assert model(torch.randn(2, 3, 56, 56), metadata).shape == (2, 2)


def test_efficientnet_checkpoint_round_trip_is_prediction_stable(monkeypatch, tmp_path):
    from src.strategies import cnn_common, efficientnet_strategy
    from src.training.checkpointing import save_checkpoint

    monkeypatch.setattr(cnn_common, "build_cnn_model", lambda _architecture, num_classes, _dropout, _pretrained: TinyImageModel(num_classes))
    model = TinyImageModel(2).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint = tmp_path / "efficientnet.pt"
    save_checkpoint(model, optimizer, 1, checkpoint, config={"input_mode": "full_image", "dropout": 0.2}, architecture="efficientnet_v2_s", class_names=["0", "1"])
    loaded, state = efficientnet_strategy.load_cnn_checkpoint(checkpoint, device=torch.device("cpu"))
    image = torch.randn(2, 3, 32, 32)
    assert state["architecture"] == "efficientnet_v2_s"
    assert torch.allclose(model(image), loaded(image), atol=1e-6)


def test_training_rejects_pre_split_patient_or_lesion_leakage(manifest_frame):
    from src.strategies.efficientnet_strategy import train

    with pytest.raises(ValueError, match="groups cross"):
        train(manifest_frame, {"task": "diagnosis_binary", "pretrained": False, "epochs": 1})
