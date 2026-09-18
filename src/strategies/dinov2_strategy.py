"""DINOv2 image-only classifiers for clinical macro-photo diagnosis tasks."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.datasets import ManifestImageDataset, TASK_TARGETS, select_task_manifest, validate_manifest
from src.data.transforms import build_transforms
from src.evaluation.metrics import classification_metrics
from src.training.checkpointing import checkpoint_path, save_checkpoint
from src.training.experiment_runner import persist_run
from src.training.trainer import TrainingResult


def load_dinov2_backbone(name: str = "dinov2_vits14", pretrained: bool = True) -> nn.Module:
    """Load a configurable DINOv2 hub backbone only when a run is started."""
    try:
        return torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
    except Exception as exc:
        raise RuntimeError("Could not load DINOv2. Cache the requested weights or provide backbone_factory for an offline run.") from exc


def backbone_embedding_dim(backbone: nn.Module) -> int:
    for attribute in ("embed_dim", "num_features", "hidden_dim"):
        value = getattr(backbone, attribute, None)
        if isinstance(value, int): return value
    with torch.no_grad(): output = backbone(torch.zeros(1, 3, 224, 224))
    if isinstance(output, dict): output = output["x_norm_clstoken"] if "x_norm_clstoken" in output else next(iter(output.values()))
    return int(output.shape[-1])


def set_backbone_trainability(backbone: nn.Module, unfreeze_last_blocks: int = 0) -> None:
    """Freeze DINOv2, optionally unfreezing its final transformer blocks and norm."""
    for parameter in backbone.parameters(): parameter.requires_grad = False
    if unfreeze_last_blocks <= 0: return
    blocks = list(getattr(backbone, "blocks", []))
    if not blocks: raise ValueError("Selected backbone has no exposed transformer blocks")
    for block in blocks[-unfreeze_last_blocks:]:
        for parameter in block.parameters(): parameter.requires_grad = True
    for name in ("norm", "fc_norm"):
        layer = getattr(backbone, name, None)
        if layer:
            for parameter in layer.parameters(): parameter.requires_grad = True


class DinoV2Classifier(nn.Module):
    """DINOv2 encoder with a compact configurable classification head."""
    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = .2, head_width: int | None = None):
        super().__init__(); self.backbone = backbone; self.embedding_dim = backbone_embedding_dim(backbone)
        width = head_width or self.embedding_dim
        self.head = nn.Sequential(nn.LayerNorm(self.embedding_dim), nn.Dropout(dropout), nn.Linear(self.embedding_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, num_classes))
    def encode_image(self, image):
        output = self.backbone(image)
        if not isinstance(output, dict): return output
        return output["x_norm_clstoken"] if "x_norm_clstoken" in output else next(iter(output.values()))
    def forward(self, image): return self.head(self.encode_image(image))


class _EncodedManifestDataset(Dataset):
    def __init__(self, manifest, transform, task, class_to_index): self.base = ManifestImageDataset(manifest, transform=transform, task=task); self.class_to_index = class_to_index
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        item = self.base[index]; return {**item, "target": self.class_to_index[item["target"]]}


def make_image_loaders(manifest, task, image_size, batch_size, augmentation=None, weighted_sampling=False, num_workers=0):
    """Build loaders from existing manifest splits; this function never creates splits."""
    frame = select_task_manifest(validate_manifest(manifest), task)
    if not set(frame.split.dropna()).issuperset({"train", "validation"}): raise ValueError("Manifest must have train/validation splits from make_group_splits")
    labels = sorted(frame.loc[frame.split.eq("train"), TASK_TARGETS[task]].unique().tolist()); class_to_index = {label: index for index, label in enumerate(labels)}
    if len(class_to_index) < 2: raise ValueError("Training split needs at least two classes")
    train = _EncodedManifestDataset(frame.loc[frame.split.eq("train")], build_transforms(image_size, True, augmentation), task, class_to_index)
    validation = _EncodedManifestDataset(frame.loc[frame.split.eq("validation")], build_transforms(image_size, False), task, class_to_index)
    sampler = None
    if weighted_sampling:
        targets = [train[index]["target"] for index in range(len(train))]; counts = np.bincount(targets, minlength=len(labels)); sampler = WeightedRandomSampler([1. / counts[target] for target in targets], len(targets), replacement=True)
    return {"train": DataLoader(train, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=num_workers), "validation": DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers)}, class_to_index


def _loss(logits, target, name, weights):
    if name == "weighted_cross_entropy": return nn.functional.cross_entropy(logits, target, weight=weights)
    if name == "focal":
        base = nn.functional.cross_entropy(logits, target, weight=weights, reduction="none"); return ((1 - torch.exp(-base)).pow(2) * base).mean()
    if name == "cross_entropy": return nn.functional.cross_entropy(logits, target)
    raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal")


def train_dinov2(manifest, config: dict, backbone_factory: Callable[..., nn.Module] = load_dinov2_backbone) -> TrainingResult:
    """Train DINOv2, preserving the best validation checkpoint and reproducible run data."""
    cfg = {"strategy_name":"dinov2", "task":"diagnosis_binary", "backbone":"dinov2_vits14", "pretrained":True, "image_size":224, "batch_size":8, "epochs":10, "head_learning_rate":3e-4, "backbone_learning_rate":1e-5, "weight_decay":1e-4, "dropout":.2, "unfreeze_last_blocks":0, "loss":"weighted_cross_entropy", "patience":4, **config}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_to_index = make_image_loaders(manifest, cfg["task"], cfg["image_size"], cfg["batch_size"], cfg.get("augmentation"), cfg.get("weighted_sampling", False), cfg.get("num_workers", 0))
    backbone = backbone_factory(cfg["backbone"], pretrained=cfg["pretrained"]); set_backbone_trainability(backbone, cfg["unfreeze_last_blocks"]); model = DinoV2Classifier(backbone, len(class_to_index), cfg["dropout"], cfg.get("head_width")).to(device)
    optimizer = torch.optim.AdamW([{"params":model.head.parameters(), "lr":cfg["head_learning_rate"]}, {"params":[p for p in model.backbone.parameters() if p.requires_grad], "lr":cfg["backbone_learning_rate"]}], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = np.bincount([item["target"] for item in loaders["train"].dataset], minlength=len(class_to_index)); weights = torch.tensor(len(counts) / (len(counts) * counts), dtype=torch.float32, device=device) if cfg["loss"] in {"weighted_cross_entropy", "focal"} else None
    best_state, best_metrics, history, stale, start = None, {}, [], 0, time.monotonic()
    for epoch in range(cfg["epochs"]):
        row = {"epoch":epoch + 1}
        for phase in ("train", "validation"):
            model.train(phase == "train"); total, targets, predictions, probabilities = 0., [], [], []
            for batch in loaders[phase]:
                image, target = batch["image"].to(device), batch["target"].to(device); optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(image); loss = _loss(logits, target, cfg["loss"], weights)
                if phase == "train": scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total += loss.item() * len(target); probability = logits.detach().softmax(1); targets.extend(target.cpu().tolist()); predictions.extend(probability.argmax(1).cpu().tolist()); probabilities.extend(probability.cpu().tolist())
            metrics = classification_metrics(targets, predictions, probabilities); prefix = "train" if phase == "train" else "validation"; row.update({f"{prefix}_loss":total / len(loaders[phase].dataset), **{f"{prefix}_{key}":value for key, value in metrics.items() if isinstance(value, (int, float))}})
            if phase == "validation": validation_metrics = metrics
        row["learning_rate"] = optimizer.param_groups[0]["lr"]; history.append(row)
        if validation_metrics["macro_f1"] > best_metrics.get("macro_f1", -1): best_state, best_metrics, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_metrics, epoch + 1, 0
        else: stale += 1
        if stale >= cfg["patience"]: break
    run_name = cfg.get("run_name", f"{cfg['task']}_{cfg['backbone']}"); path = checkpoint_path("dinov2", cfg["backbone"], run_name); model.load_state_dict(best_state); save_checkpoint(model, optimizer, best_epoch, path, config=cfg, class_to_index=class_to_index, metrics=best_metrics)
    result = TrainingResult(history=history, best_epoch=best_epoch, best_checkpoint=str(path), config=cfg, timing={"training_seconds":time.monotonic() - start}, metrics=best_metrics); persist_run(result.as_dict(), Path("results/runs") / run_name); return result


def load_dinov2_checkpoint(path: str | Path, backbone_factory=load_dinov2_backbone, map_location="cpu"):
    """Recreate a saved DINOv2 classifier for later external evaluation."""
    payload = torch.load(path, map_location=map_location, weights_only=False); cfg = payload["config"]; model = DinoV2Classifier(backbone_factory(cfg["backbone"], pretrained=False), len(payload["class_to_index"]), cfg["dropout"], cfg.get("head_width")); model.load_state_dict(payload["model_state_dict"]); return model, payload
