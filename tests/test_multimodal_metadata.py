"""Regression tests for multimodal metadata and diagnosis label contracts."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from torch import nn

from src.data.metadata import metadata_audit_report, normalize_metadata_frame, normalize_metadata_value
from src.data.targets import target_encoding_for_manifest
from src.evaluation.metrics import classification_metrics
from conftest import TinyDinoBackbone
from src.strategies.multimodal_strategy import MetadataPreprocessor, _run_epoch, build_multimodal_model


def test_explicit_metadata_normalization_is_case_and_whitespace_insensitive():
    assert normalize_metadata_value("sex", " FEMALE ") == "female"
    assert normalize_metadata_value("sex", "m") == "male"
    assert normalize_metadata_value("anatomical_site", " ARM ") == "upper_extremity"
    assert normalize_metadata_value("anatomical_site", "head/neck") == "head_neck"
    assert normalize_metadata_value("anatomical_site", "THIGH") == "lower_extremity"
    assert normalize_metadata_value("skin_tone", " Type IV ") == "fitzpatrick_4"
    assert normalize_metadata_value("skin_tone", "1.0") == "fitzpatrick_1"
    assert normalize_metadata_value("skin_tone", 1.0) == "fitzpatrick_1"
    assert normalize_metadata_value("anatomical_site", "ambiguous source term") == "unmapped:ambiguous source term"
    assert normalize_metadata_value("sex", None) is None


def test_metadata_preprocessor_fits_only_train_and_keeps_missing_unknown_separate():
    train = pd.DataFrame({"age": [20, 40], "sex": [" FEMALE", "male "], "anatomical_site": ["ARM", "FOREARM"], "skin_tone": ["I", None]})
    held_out = pd.DataFrame({"age": [1000, None], "sex": ["other", None], "anatomical_site": ["MYSTERY", None], "skin_tone": ["VI", None]})
    processor = MetadataPreprocessor().fit(train)
    assert processor.age_mean == 30.0
    assert processor.age_std > 0
    assert processor.vocabularies["sex"] == {"female": 2, "male": 3}
    assert processor.vocabularies["anatomical_site"] == {"upper_extremity": 2}
    tensors = processor.transform(held_out)
    assert tensors["sex"].tolist() == [1, 0]
    assert tensors["anatomical_site"].tolist() == [1, 0]
    assert tensors["skin_tone"].tolist() == [1, 0]
    assert tensors["continuous"][0, 0].item() > 10  # validation/dev-test age never refits mean/std
    assert "other" not in processor.vocabularies["sex"]
    assert "fitzpatrick_6" not in processor.vocabularies["skin_tone"]


def test_normalization_provenance_retains_raw_and_source():
    frame = pd.DataFrame({"source_dataset": ["SOURCE"], "sex": [" FEMALE "], "anatomical_site": ["ARM"], "skin_tone": ["II"], "age": [33]})
    result = normalize_metadata_frame(frame, ("age", "sex", "anatomical_site", "skin_tone"))
    assert result.loc[0, "metadata_source"] == "SOURCE"
    assert result.loc[0, "raw_anatomical_site"] == "ARM"
    assert result.loc[0, "normalized_anatomical_site"] == "upper_extremity"


def test_binary_metric_reports_and_json_use_canonical_diagnostic_names():
    metrics = classification_metrics([0, 1], [0, 1], [[.9, .1], [.1, .9]], labels=[0, 1], class_names=["benign", "malignant"])
    assert metrics["labels"] == [0, 1]
    assert metrics["class_names"] == ["benign", "malignant"]
    assert [row["class_name"] for row in metrics["class_metrics"]] == ["benign", "malignant"]
    assert "benign" in json.dumps(metrics)


def test_multimodal_epoch_uses_loader_target_encoding_names():
    class Model(nn.Module):
        def forward(self, image, metadata):
            return torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    class Loader(list):
        pass
    loader = Loader([{"image": torch.zeros(2, 3, 2, 2), "target": torch.tensor([0, 1]), "metadata": [{}, {}], "metadata_tensors": {"continuous": torch.zeros(2, 0)}}])
    loader.dataset = type("Dataset", (), {"class_to_index": {"benign": 0, "malignant": 1}})()
    loader.class_names = ["benign", "malignant"]
    result = _run_epoch(Model(), loader, None, None, torch.device("cpu"), {"loss": "cross_entropy"}, None, training=False)
    assert result["metrics"]["class_names"] == ["benign", "malignant"]


def test_target_encoding_and_metadata_audit_preserve_binary_and_ddi_protection(manifest_frame):
    encoding = target_encoding_for_manifest(manifest_frame, "diagnosis_binary")
    assert encoding.encode(0) == 0 and encoding.encode(1) == 1
    report = metadata_audit_report(manifest_frame)
    assert report["task"] == "diagnosis_binary"
    ddi = manifest_frame.copy(); ddi["dataset"] = "DDI"
    with pytest.raises(PermissionError, match="DDI"):
        metadata_audit_report(ddi)


@pytest.mark.parametrize("fields", [(), ("age", "sex", "anatomical_site"), ("age", "sex", "anatomical_site", "skin_tone")])
def test_multimodal_field_subsets_and_image_only_path(fields):
    processor = MetadataPreprocessor(fields).fit(pd.DataFrame({
        "age": [20, 40], "sex": ["female", "male"],
        "anatomical_site": ["ARM", "FACE"], "skin_tone": ["1", "6"],
    }))
    model = build_multimodal_model({"backbone": "tiny", "pretrained": False, "metadata_width": 4, "fusion_width": 8, "metadata_embedding_dim": 3, "dropout": 0.0}, processor, 2, backbone_factory=lambda *_args, **_kwargs: TinyDinoBackbone())
    encoded = processor.transform(pd.DataFrame([{}]))
    assert model(torch.zeros(1, 3, 4, 4), encoded).shape == (1, 2)
    assert MetadataPreprocessor.from_config(processor.config()).fields == fields
