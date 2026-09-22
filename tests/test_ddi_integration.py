from __future__ import annotations

import pandas as pd
from PIL import Image

import pytest
import torch
from src.data.ddi import build_ddi_manifest, validate_ddi_evaluation_manifest
from src.evaluation.evaluator import CheckpointBundle, build_evaluation_loader


def test_official_ddi_flag_is_the_only_binary_mapping(tmp_path):
    root = tmp_path / "ddi"; images = root / "images"; images.mkdir(parents=True)
    Image.new("RGB", (12, 12)).save(images / "000001.png")
    Image.new("RGB", (12, 12)).save(images / "000002.png")
    pd.DataFrame([{"DDI_file": "000001.png", "malignancy(malig=1)": 1, "disease": "unreviewed wording", "skin_tone": 56}, {"DDI_file": "000002.png", "malignancy(malig=1)": 0, "disease": "other", "skin_tone": 12}]).to_csv(root / "ddi_metadata.csv", index=False)
    manifest, report = build_ddi_manifest(root)
    assert manifest.binary_target.tolist() == [1, 0]
    assert manifest.lesion_present.tolist() == [True, True]
    assert manifest.original_label.tolist()[0] == "unreviewed wording"
    assert report["eligible_binary_count"] == 2
    assert validate_ddi_evaluation_manifest(manifest).binary_target.tolist() == [1, 0]


def test_ddi_empty_or_single_class_targets_fail_clearly(manifest_frame):
    empty = manifest_frame.iloc[:0].copy()
    with pytest.raises(ValueError, match="zero eligible"):
        validate_ddi_evaluation_manifest(empty)
    one = manifest_frame.iloc[[0]].copy(); one["binary_target"] = 0
    with pytest.raises(ValueError, match="both canonical"):
        validate_ddi_evaluation_manifest(one)


def test_external_loader_uses_frozen_binary_contract_not_train_split(tmp_path):
    root = tmp_path / "ddi"; images = root / "images"; images.mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (12, 12)).save(images / name)
    pd.DataFrame([{"DDI_file": "a.png", "malignant": False}, {"DDI_file": "b.png", "malignant": True}]).to_csv(root / "ddi_metadata.csv", index=False)
    manifest, _ = build_ddi_manifest(root)
    frozen = tmp_path / "frozen.yaml"; frozen.write_text("status: frozen\n", encoding="utf-8")
    bundle = CheckpointBundle(torch.nn.Identity(), "efficientnet", "diagnosis_binary", ["benign", "malignant"], {"image_size": 16}, "unused", {})
    loader = build_evaluation_loader(manifest, bundle, split=None, allow_final_test=True, frozen_config_path=frozen)
    assert loader.target_encoding.class_values == (0, 1)
    assert next(iter(loader))["target"] == [0, 1]
