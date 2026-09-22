"""Dispatch model-specific attribution without duplicate checkpoint loading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gradcam import gradcam
from .transformer_attribution import transformer_token_attribution


@dataclass(frozen=True)
class AttributionResult:
    method: str
    heatmap: np.ndarray


def _cnn_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError(f"{model_name} does not expose CNN feature stages for Grad-CAM.")
    # torchvision ConvNeXt-Tiny's final stage is features[7]; EfficientNetV2-S
    # terminates in features[7].  ``features[-1]`` is robust to compatible
    # torchvision layouts and is the final spatial feature map before pooling.
    layer = features[-1]
    return layer


def generate_attribution(model: torch.nn.Module, inputs: tuple, *, model_name: str, malignant_index: int) -> AttributionResult:
    """Generate one on-demand image attribution map from an already-loaded model."""
    image = inputs[0].detach().clone().requires_grad_(True)
    attribution_inputs = (image, *inputs[1:])
    with torch.enable_grad():
        if model_name in {"convnext", "efficientnet"}:
            heatmap = gradcam(model, attribution_inputs, _cnn_target_layer(model, model_name), malignant_index)
            method = "Grad-CAM"
        elif model_name == "multimodal":
            heatmap = transformer_token_attribution(model, attribution_inputs, malignant_index)
            method = "Transformer Attribution"
        else:
            raise ValueError(f"Attribution is not supported for {model_name!r}.")
    return AttributionResult(method=method, heatmap=heatmap.numpy())
