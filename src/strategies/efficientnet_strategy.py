"""EfficientNetV2-S transfer-learning strategy for clinical macro photographs."""
from __future__ import annotations
from .cnn_common import build_cnn_model, load_cnn_checkpoint, train_cnn_strategy

ARCHITECTURE = "efficientnet_v2_s"

def build_model(num_classes: int, dropout: float = .2, pretrained: bool = True):
    """Build the EfficientNetV2-S classification model."""
    return build_cnn_model(ARCHITECTURE, num_classes, dropout, pretrained)

def train(manifest, config: dict, run_name: str | None = None, box_provider=None):
    """Run the complete EfficientNet development lifecycle."""
    return train_cnn_strategy(manifest, config, "efficientnet", ARCHITECTURE, run_name, box_provider)

__all__ = ["ARCHITECTURE", "build_model", "load_cnn_checkpoint", "train"]
