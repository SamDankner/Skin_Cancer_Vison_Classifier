"""Separate lesion-present versus normal-skin CNN strategy for macro photographs."""
from __future__ import annotations
from src.data.datasets import build_lesion_presence_manifest
from .cnn_common import build_cnn_model, load_cnn_checkpoint, train_cnn_strategy

def build_model(num_classes: int = 2, dropout: float = .2, pretrained: bool = True, architecture: str = "efficientnet_v2_s"):
    """Build a lesion-present versus normal-skin classifier."""
    return build_cnn_model(architecture, num_classes, dropout, pretrained)

def train(manifest, config: dict, run_name: str | None = None):
    """Train lesion presence only when genuine normal-skin negatives exist."""
    config = dict(config)
    task = config.setdefault("task", "lesion_presence")
    if task != "lesion_presence":
        raise ValueError("lesion_presence_strategy only supports the lesion_presence task")
    strengths = tuple(config.get("normal_label_strengths", ("strong", "moderate", "weak")))
    manifest = build_lesion_presence_manifest(manifest, strengths)
    architecture = config.get("backbone", "efficientnet_v2_s")
    return train_cnn_strategy(manifest, config, "lesion_presence", architecture, run_name)

__all__ = ["build_model", "load_cnn_checkpoint", "train"]
