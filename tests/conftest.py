"""Shared offline fixtures and tiny neural networks for the test suite."""
from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from src.data.datasets import MANIFEST_COLUMNS


class TinyImageModel(nn.Module):
    """Offline stand-in used to exercise strategy wiring, not model quality."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))

    def forward(self, image):
        return self.classifier(image)


class TinyDinoBackbone(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(3, self.embed_dim)

    def forward(self, image):
        return self.projection(self.pool(image).flatten(1))


@pytest.fixture
def manifest_frame():
    rows = []
    for index in range(60):
        row = {column: None for column in MANIFEST_COLUMNS}
        row.update(
            dataset="SYNTHETIC",
            image_path=f"synthetic_{index}.jpg",
            image_id=f"image-{index}",
            patient_id=f"patient-{index // 2}",
            lesion_id=f"lesion-{index // 2}",
            original_label="melanoma" if index % 2 else "nevus",
            harmonized_diagnosis="melanoma" if index % 2 else "nevus",
            binary_target=index % 2,
            lesion_present=1,
            normal_skin=False,
            image_quality_label="usable",
            supported_for_lesion_detection=True,
            supported_for_diagnosis=True,
            image_modality="clinical_macro",
            split=("train", "validation", "test")[index % 3],
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
