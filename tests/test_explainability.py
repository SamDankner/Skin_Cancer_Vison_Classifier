"""Focused tests for on-demand CNN and DINOv2 attribution helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.explainability import generate_attribution, overlay_heatmap
from src.explainability.gradcam import gradcam
from src.explainability.transformer_attribution import _patch_grid


class TinyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 2)
        with torch.no_grad():
            self.classifier.weight[1].fill_(1.0)

    def forward(self, image):
        return self.classifier(self.pool(self.features(image)).flatten(1))


class TokenBlock(nn.Module):
    def forward(self, tokens):
        cls = tokens[:, :1] + tokens[:, 1:].mean(dim=1, keepdim=True)
        return torch.cat((cls, tokens[:, 1:]), dim=1)


class TinyDinoBackbone(nn.Module):
    class PatchEmbed:
        patch_size = 2

    def __init__(self):
        super().__init__()
        self.patch_embed = self.PatchEmbed()
        self.num_tokens = 1
        self.num_register_tokens = 0
        self.blocks = nn.ModuleList([TokenBlock()])

    def forward(self, image):
        patches = image.mean(dim=1).unfold(1, 2, 2).unfold(2, 2, 2).mean(dim=(-1, -2)).reshape(image.shape[0], -1, 1)
        tokens = torch.cat((torch.zeros_like(patches[:, :1]), patches), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return {"x_norm_clstoken": tokens[:, 0]}


class TinyMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyDinoBackbone()
        self.classifier = nn.Linear(1, 2)
        with torch.no_grad():
            self.classifier.weight[1].fill_(1.0)

    def forward(self, image, metadata):
        return self.classifier(self.backbone(image)["x_norm_clstoken"] + metadata["continuous"][:, :1] * 0)


def test_gradcam_captures_spatial_gradients_normalizes_output_and_removes_hook():
    model = TinyCnn()
    image = torch.ones((1, 3, 8, 8), requires_grad=True)
    heatmap = gradcam(model, (image,), model.features[-1], 1)
    assert heatmap.shape == (8, 8)
    assert torch.isfinite(heatmap).all()
    assert 0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1
    assert not model.features[-1]._forward_hooks


def test_cnn_service_targets_malignant_logit_and_reports_gradcam():
    result = generate_attribution(TinyCnn(), (torch.ones((1, 3, 8, 8)),), model_name="convnext", malignant_index=1)
    assert result.method == "Grad-CAM"
    assert result.heatmap.shape == (8, 8)
    assert np.isfinite(result.heatmap).all()
    assert 0 <= result.heatmap.min() <= result.heatmap.max() <= 1


def test_transformer_service_uses_final_block_input_tokens_and_returns_non_degenerate_patch_map():
    model = TinyMultimodal()
    metadata = {"continuous": torch.zeros((1, 2))}
    image = torch.arange(192, dtype=torch.float32).reshape(1, 3, 8, 8)
    result = generate_attribution(model, (image, metadata), model_name="multimodal", malignant_index=1)
    assert result.method == "Transformer Attribution"
    assert result.method != "Grad-CAM"
    assert result.heatmap.shape == (8, 8)
    assert np.isfinite(result.heatmap).all()
    assert 0 <= result.heatmap.min() <= result.heatmap.max() <= 1
    assert result.heatmap.std() > 0
    assert not model.backbone.blocks[-1]._forward_pre_hooks


def test_patch_grid_removes_only_declared_cls_and_register_tokens():
    backbone = TinyDinoBackbone()
    image = torch.zeros((1, 3, 8, 8))
    assert _patch_grid(backbone, image, token_count=17) == (4, 4, 1)
    with pytest.raises(ValueError, match="token count"):
        _patch_grid(backbone, image, token_count=18)


def test_overlay_keeps_original_display_dimensions():
    from PIL import Image

    overlay = overlay_heatmap(Image.new("RGB", (13, 7), "white"), np.ones((4, 4)), .45)
    assert overlay.mode == "RGB"
    assert overlay.size == (13, 7)
