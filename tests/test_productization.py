"""Lightweight tests for productized training and inference behavior."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch

from conftest import TinyDinoBackbone, TinyImageModel
from src.data.datasets import MANIFEST_COLUMNS
from src.evaluation.evaluator import CheckpointBundle
from src.training.early_stopping import EarlyStopping
from src.training.experiment_runner import create_run_directory, validate_experiment_config


def test_early_stopping_honors_min_delta_and_restores_best_state():
    stopper = EarlyStopping(patience=2, min_delta=0.05, mode="max", monitor="validation_macro_f1")
    assert stopper.update(0.50, 1, {"weight": torch.tensor([1.0])})
    assert not stopper.update(0.53, 2, {"weight": torch.tensor([2.0])})
    assert not stopper.should_stop
    assert stopper.update(0.56, 3, {"weight": torch.tensor([3.0])})
    assert not stopper.update(0.57, 4, {"weight": torch.tensor([4.0])})
    assert not stopper.update(0.58, 5, {"weight": torch.tensor([5.0])})
    assert stopper.should_stop
    assert stopper.best_epoch == 3
    assert stopper.best_state["weight"].item() == 3.0


def test_missing_metadata_uses_indicators_without_fabricated_values():
    from src.strategies.multimodal_strategy import MetadataPreprocessor

    processor = MetadataPreprocessor().fit(pd.DataFrame({
        "age": [20, 40], "sex": ["female", "male"],
        "anatomical_site": ["arm", "torso"], "skin_tone": ["light", "dark"],
    }))
    transformed = processor.transform(pd.DataFrame([{}]))
    assert transformed["continuous"].tolist() == [[0.0, 1.0]]
    assert transformed["sex"].item() == 0
    assert transformed["anatomical_site"].item() == 0
    assert transformed["skin_tone"].item() == 0
    assert processor.config()["missing_value_policy"]["age"] == "training_mean_with_missing_indicator"


def test_multimodal_checkpoint_preserves_class_order_and_preprocessing(tmp_path):
    from src.strategies.multimodal_strategy import (
        MetadataPreprocessor, build_multimodal_model, load_multimodal_checkpoint,
    )

    processor = MetadataPreprocessor(("age", "sex")).fit(pd.DataFrame({"age": [20, 40], "sex": ["f", "m"]}))
    config = {
        "backbone": "tiny", "pretrained": False, "metadata_embedding_dim": 4,
        "metadata_width": 6, "fusion_width": 8, "dropout": 0.0, "metadata_dropout": 0.0,
        "unfreeze_last_blocks": 0,
    }
    model = build_multimodal_model(config, processor, 2, backbone_factory=lambda *_args, **_kwargs: TinyDinoBackbone())
    checkpoint = tmp_path / "multimodal.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "class_names": ["benign", "malignant"], "metadata_preprocessor": processor.config(),
    }, checkpoint)
    loaded, payload = load_multimodal_checkpoint(
        checkpoint,
        backbone_factory=lambda *_args, **_kwargs: TinyDinoBackbone(),
        map_location="cpu",
    )
    assert payload["class_names"] == ["benign", "malignant"]
    assert loaded.metadata_preprocessor.fields == ("age", "sex")
    assert loaded.metadata_preprocessor.age_mean == 30.0


def test_multimodal_training_runs_full_lightweight_lifecycle(tmp_path):
    from src.strategies.multimodal_strategy import train

    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for index in range(4):
            image_path = tmp_path / f"{split}_{index}.png"
            Image.new("RGB", (20, 20), color=(40 + index * 20, 80, 120)).save(image_path)
            row = {column: None for column in MANIFEST_COLUMNS}
            row.update(
                dataset="SYNTHETIC", image_path=str(image_path), image_id=f"{split}-{index}",
                patient_id=f"patient-{split_index}-{index}", lesion_id=f"lesion-{split_index}-{index}",
                original_label="melanoma" if index % 2 else "nevus",
                harmonized_diagnosis="melanoma" if index % 2 else "nevus",
                binary_target=index % 2, lesion_present=1, normal_skin=False,
                supported_for_lesion_detection=True, supported_for_diagnosis=True,
                age=30 + index, sex="female" if index % 2 else None,
                image_modality="clinical_macro", split=split,
            )
            rows.append(row)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    result = train(
        manifest,
        {
            "backbone": "tiny", "pretrained": False, "image_size": 16, "batch_size": 2, "epochs": 1,
            "metadata_fields": ["age", "sex"], "metadata_embedding_dim": 4,
            "metadata_width": 6, "fusion_width": 8, "dropout": 0.0,
            "metadata_dropout": 0.0, "loss": "cross_entropy",
            "early_stopping_patience": 1, "inference_warmup": 0, "inference_repeats": 1,
            "models_root": str(tmp_path / "models"), "run_directory": str(tmp_path / "run"),
            "summary_path": str(tmp_path / "summary.csv"),
        },
        run_name="tiny_multimodal",
        backbone_factory=lambda *_args, **_kwargs: TinyDinoBackbone(),
    )
    assert Path(result.best_checkpoint).is_file()
    assert result.best_epoch == 1
    assert result.development_test_metrics["sample_count"] == 4
    assert (tmp_path / "run" / "development_test_predictions.csv").is_file()
    assert (tmp_path / "run" / "training_history.png").is_file()


def test_lesion_presence_one_batch_uses_canonical_indices(monkeypatch, tmp_path):
    import src.strategies.cnn_common as cnn_common

    rows = []
    values = (False, True, 0.0, 1.0)
    for split_index, split in enumerate(("train", "validation", "test")):
        for index, target in enumerate(values):
            image_path = tmp_path / f"lesion_{split}_{index}.png"
            Image.new("RGB", (20, 20), color=(40 + index * 30, 80, 120)).save(image_path)
            canonical = int(target)
            row = {column: None for column in MANIFEST_COLUMNS}
            row.update(
                dataset="SYNTHETIC",
                source_dataset="SYNTHETIC",
                image_path=str(image_path),
                image_id=f"{split}-{index}",
                patient_id=f"patient-{split_index}-{index}",
                lesion_id=f"lesion-{split_index}-{index}",
                lesion_present=target,
                normal_skin=not bool(canonical),
                normal_label_strength="weak" if not canonical else None,
                supported_for_lesion_detection=True,
                supported_for_lesion_presence=True,
                supported_for_diagnosis=False,
                image_modality="clinical",
                split=split,
            )
            rows.append(row)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    monkeypatch.setattr(
        cnn_common,
        "build_diagnostic_input_model",
        lambda _input_mode, _architecture, num_classes, _dropout, _pretrained: TinyImageModel(num_classes),
    )

    result = cnn_common.train_cnn_strategy(
        manifest,
        {
            "task": "lesion_presence",
            "pretrained": False,
            "image_size": 16,
            "batch_size": 4,
            "epochs": 1,
            "head_epochs": 1,
            "weighted_sampling": True,
            "loss": "cross_entropy",
            "early_stopping_patience": 1,
            "inference_warmup": 0,
            "inference_repeats": 1,
            "models_root": str(tmp_path / "models"),
            "run_directory": str(tmp_path / "run"),
            "summary_path": str(tmp_path / "summary.csv"),
        },
        strategy_name="lesion_presence",
        architecture="efficientnet_v2_s",
        run_name="tiny_lesion_presence",
    )

    assert result.config["class_names"] == ["no_lesion", "lesion_present"]
    assert result.best_epoch == 1
    assert result.development_test_metrics["sample_count"] == 4


def test_photo_only_inference_returns_probabilities_without_metadata(monkeypatch, tmp_path):
    import src.inference as inference

    model = TinyImageModel(2).eval()
    bundle = CheckpointBundle(
        model, "efficientnet", "diagnosis_binary", ["0", "1"],
        {"image_size": 16, "input_mode": "full_image"}, "tiny.pt", {},
    )
    monkeypatch.setattr(inference, "load_checkpoint_bundle", lambda *_args, **_kwargs: bundle)
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (24, 24), color=(80, 120, 160)).save(image_path)
    result = inference.predict_image(image_path, checkpoint="tiny.pt", warmup=0, repeats=2)
    assert result["metadata_used"] is False
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["model"] == "efficientnet"
    assert model.training is False


def test_top_level_config_validation_and_run_directory(tmp_path):
    strategy = tmp_path / "strategy.yaml"
    strategy.write_text("strategy_name: efficientnet\ntask: diagnosis_binary\nbackbone: efficientnet_v2_s\n", encoding="utf-8")
    config = {
        "manifest_path": "data/processed/development_manifest.csv",
        "strategies": {"efficientnet": {"enabled": True, "config": strategy.name}},
    }
    validated = validate_experiment_config(config, root=tmp_path)
    assert validated["enabled_strategies"] == ["efficientnet"]
    created = create_run_directory("run", tmp_path / "runs")
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory("run", tmp_path / "runs")
    config["manifest_path"] = "data/final_external_test/DDI/manifest.csv"
    with pytest.raises(PermissionError, match="DDI"):
        validate_experiment_config(config, root=tmp_path)
