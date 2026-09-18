"""Late-fusion DINOv2 and pre-diagnosis metadata classifiers for macro photographs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.strategies.dinov2_strategy import DinoV2Classifier, backbone_embedding_dim, load_dinov2_backbone, set_backbone_trainability

SAFE_METADATA_FIELDS = ("age", "sex", "anatomical_site", "skin_tone")
LEAKAGE_PRONE_FIELDS = ("pathology", "diagnosis", "ground_truth", "biopsy", "treatment", "assessment", "target", "label")


@dataclass
class MetadataPreprocessor:
    """Training-split-only normalization and categorical vocabularies with missing/unknown IDs."""
    fields: tuple[str, ...] = SAFE_METADATA_FIELDS
    age_mean: float = 0.0
    age_std: float = 1.0
    vocabularies: dict | None = None
    def _validate_fields(self):
        invalid = set(self.fields) - set(SAFE_METADATA_FIELDS)
        if invalid: raise ValueError(f"Metadata fields are not allow-listed: {sorted(invalid)}")
    def fit(self, train_frame: pd.DataFrame):
        self._validate_fields(); self.vocabularies = {}
        if "age" in self.fields:
            ages = pd.to_numeric(train_frame.age, errors="coerce"); self.age_mean = float(ages.mean()) if ages.notna().any() else 0.; self.age_std = float(ages.std()) if ages.notna().sum() > 1 else 1.
            if not np.isfinite(self.age_std) or self.age_std == 0: self.age_std = 1.
        for field in set(self.fields) - {"age"}:
            values = train_frame[field].fillna("<MISSING>").astype(str).str.strip().replace("", "<MISSING>")
            self.vocabularies[field] = {value:index + 2 for index, value in enumerate(sorted(values.unique()))}
        return self
    def transform(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        if self.vocabularies is None: raise RuntimeError("Call fit on the training split before transform")
        result = {}
        if "age" in self.fields:
            age = pd.to_numeric(frame.age, errors="coerce"); result["continuous"] = torch.tensor(np.column_stack([(age.fillna(self.age_mean) - self.age_mean) / self.age_std, age.isna().astype(float)]), dtype=torch.float32)
        else: result["continuous"] = torch.empty((len(frame), 0), dtype=torch.float32)
        for field, vocabulary in self.vocabularies.items():
            values = frame[field].fillna("<MISSING>").astype(str).str.strip().replace("", "<MISSING>")
            result[field] = torch.tensor([0 if value == "<MISSING>" else vocabulary.get(value, 1) for value in values], dtype=torch.long)
        return result
    def config(self): return {"fields":list(self.fields), "age_mean":self.age_mean, "age_std":self.age_std, "vocabularies":self.vocabularies}


class MetadataEncoder(nn.Module):
    def __init__(self, preprocessor, embedding_dim=16, width=64):
        super().__init__(); self.fields = tuple(preprocessor.fields); self.embeddings = nn.ModuleDict({field:nn.Embedding(len(vocab) + 2, embedding_dim) for field, vocab in (preprocessor.vocabularies or {}).items()}); input_dim = (2 if "age" in self.fields else 0) + embedding_dim * len(self.embeddings); self.network = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width))
    def forward(self, metadata): return self.network(torch.cat([metadata["continuous"]] + [self.embeddings[field](metadata[field]) for field in self.embeddings], dim=1))


class MultimodalClassifier(nn.Module):
    """Late fusion: clinical-image embedding concatenated with a metadata MLP embedding."""
    def __init__(self, backbone, preprocessor, num_classes, metadata_embedding_dim=16, metadata_width=64, fusion_width=256, dropout=.2):
        super().__init__(); self.backbone = backbone; self.image_dim = backbone_embedding_dim(backbone); self.metadata_encoder = MetadataEncoder(preprocessor, metadata_embedding_dim, metadata_width); self.classifier = nn.Sequential(nn.LayerNorm(self.image_dim + metadata_width), nn.Linear(self.image_dim + metadata_width, fusion_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(fusion_width, num_classes))
    def forward(self, image, metadata):
        embedding = self.backbone(image)
        if isinstance(embedding, dict): embedding = embedding["x_norm_clstoken"] if "x_norm_clstoken" in embedding else next(iter(embedding.values()))
        return self.classifier(torch.cat([embedding, self.metadata_encoder(metadata)], dim=1))


def make_metadata_tensors(manifest: pd.DataFrame, fields: Iterable[str] = SAFE_METADATA_FIELDS):
    """Fit metadata processing on only the preexisting training split, then transform rows."""
    processor = MetadataPreprocessor(tuple(fields)).fit(manifest.loc[manifest.split.eq("train")]); return processor, processor.transform(manifest)


def image_only_baseline(backbone, num_classes, dropout=.2):
    """Create the controlled image-only counterpart using the same image encoder."""
    return DinoV2Classifier(backbone, num_classes, dropout)


def build_multimodal_model(config: dict, preprocessor: MetadataPreprocessor, num_classes: int, backbone_factory=load_dinov2_backbone):
    """Construct a late-fusion model; callers use existing project loaders/splits for training."""
    cfg = {"backbone":"dinov2_vits14", "pretrained":True, "unfreeze_last_blocks":0, "metadata_embedding_dim":16, "metadata_width":64, "fusion_width":256, "dropout":.2, **config}
    backbone = backbone_factory(cfg["backbone"], pretrained=cfg["pretrained"]); set_backbone_trainability(backbone, cfg["unfreeze_last_blocks"])
    return MultimodalClassifier(backbone, preprocessor, num_classes, cfg["metadata_embedding_dim"], cfg["metadata_width"], cfg["fusion_width"], cfg["dropout"])


def ablation_config(base_config: dict, fields: Iterable[str] | None = None, image_only: bool = False) -> dict:
    """Create one controlled ablation configuration without launching a combinatorial grid."""
    config = dict(base_config); config["metadata_fields"] = [] if image_only else list(fields or SAFE_METADATA_FIELDS); config["ablation"] = "image_only" if image_only else "image_plus_metadata"; return config
