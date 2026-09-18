"""Reusable transfer-learning support for the project's CNN strategies."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image

from src.data.datasets import ManifestImageDataset, select_task_manifest, validate_manifest
from src.data.splits import leakage_report, validate_split_class_coverage
from src.data.transforms import build_transforms
from src.evaluation.metrics import classification_metrics, lesion_presence_metrics
from src.training.checkpointing import checkpoint_path, save_checkpoint
from src.training.experiment_runner import create_run_directory, persist_run
from src.training.trainer import TrainingResult
from src.utils.device import get_device
from .localization import lesion_crop, parse_bounding_box


class FullImageCropFusion(nn.Module):
    """Fuse features from a full macro photograph and an intentional lesion crop."""
    def __init__(self, architecture: str, num_classes: int, dropout: float = .2, pretrained: bool = True):
        super().__init__()
        full = build_cnn_model(architecture, num_classes, dropout, pretrained)
        crop = build_cnn_model(architecture, num_classes, dropout, pretrained)
        self.full_features, self.crop_features = full.features, crop.features
        self.full_pool, self.crop_pool = full.avgpool, crop.avgpool
        dimension = full.classifier[-1].in_features if architecture == "efficientnet_v2_s" else full.classifier[-1][-1].in_features
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimension * 2, num_classes))

    def _features(self, features, pool, image):
        return torch.flatten(pool(features(image)), 1)

    def forward(self, full_image, lesion_crop):
        return self.classifier(torch.cat((self._features(self.full_features, self.full_pool, full_image), self._features(self.crop_features, self.crop_pool, lesion_crop)), dim=1))


def build_diagnostic_input_model(input_mode: str, architecture: str, num_classes: int, dropout: float = .2, pretrained: bool = True) -> nn.Module:
    """Build separate full-image, crop-only, or original-plus-crop experiments."""
    if input_mode in {"full_image", "lesion_crop"}:
        return build_cnn_model(architecture, num_classes, dropout, pretrained)
    if input_mode == "full_plus_crop":
        return FullImageCropFusion(architecture, num_classes, dropout, pretrained)
    raise ValueError("input_mode must be full_image, lesion_crop, or full_plus_crop")


def collate_manifest_batch(batch: list[dict]) -> dict:
    """Collate manifest samples while retaining string diagnosis labels."""
    output = {"target": [sample["target"] for sample in batch], "metadata": [sample["metadata"] for sample in batch]}
    for key in ("image", "full_image", "lesion_crop"):
        if key in batch[0]:
            output[key] = torch.stack([sample[key] for sample in batch])
    return output


class CropAssistedManifestDataset(ManifestImageDataset):
    """Return full photographs and crops from an explicit localization provider.

    ``box_provider`` receives a manifest row dictionary and returns a predicted
    xyxy box. Passing it is required for automatic-crop evaluation; callers may
    explicitly use ``ground_truth_box`` only for oracle ablations.
    """
    def __init__(self, *args, crop_transform, box_provider=None, ground_truth_box: bool = False, crop_margin: float = .15, output_mode: str = "full_plus_crop", **kwargs):
        super().__init__(*args, **kwargs)
        if box_provider is None and not ground_truth_box:
            raise ValueError("Crop-assisted input requires an automatic box_provider; use ground_truth_box=True only for an explicit oracle ablation")
        self.crop_transform, self.box_provider = crop_transform, box_provider
        self.ground_truth_box, self.crop_margin, self.output_mode = ground_truth_box, crop_margin, output_mode

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        metadata = row.to_dict()
        box = parse_bounding_box(row.bounding_box) if self.ground_truth_box else parse_bounding_box(self.box_provider(metadata))
        if box is None:
            raise ValueError(f"No localization box available for image_id={row.image_id}")
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
            crop = lesion_crop(image, box, self.crop_margin)
        target = row[self.target_column]
        target = str(target) if self.target_column == "harmonized_diagnosis" else int(target)
        if self.output_mode == "lesion_crop":
            return {"image": self.crop_transform(crop), "target": target, "metadata": metadata}
        return {"full_image": self.transform(image) if self.transform else image, "lesion_crop": self.crop_transform(crop), "target": target, "metadata": metadata}


def class_names_for_manifest(manifest, task: str) -> list[str]:
    target = {"diagnosis_binary": "binary_target", "diagnosis_multiclass": "harmonized_diagnosis", "lesion_presence": "lesion_present"}[task]
    values = select_task_manifest(manifest, task)[target].dropna().unique().tolist()
    return [str(value) for value in sorted(values, key=lambda value: str(value))]


def _targets(values: Iterable[Any], class_names: list[str]) -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(class_names)}
    return torch.tensor([lookup[str(value)] for value in values], dtype=torch.long)


def build_cnn_model(architecture: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    """Build a pretrained EfficientNetV2 or ConvNeXt with a replacement head."""
    from torchvision.models import (
        ConvNeXt_Tiny_Weights, EfficientNet_V2_S_Weights, convnext_tiny,
        efficientnet_v2_s,
    )
    if architecture == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    elif architecture == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
    else:
        raise ValueError("Supported CNN architectures are efficientnet_v2_s and convnext_tiny")
    return model


def _head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier.parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze only the pretrained feature extractor."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in _head_parameters(model):
        parameter.requires_grad = True


def _loss(logits, targets, method: str, class_weights: torch.Tensor | None, focal_gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    if method == "focal":
        return ((1 - torch.exp(-ce)).pow(focal_gamma) * ce).mean()
    if method not in {"cross_entropy", "weighted_cross_entropy"}:
        raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal")
    return ce.mean()


def _epoch(model, loader, optimizer, scaler, device, loss_method, weights, focal_gamma, training: bool) -> tuple[float, list[int], list[int], list[list[float]]]:
    model.train(training)
    total_loss, count, true, predicted, scores = 0.0, 0, [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch.get("image")
            if images is not None:
                images = images.to(device, non_blocking=True)
            targets = _targets(batch["target"], loader.class_names).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images) if images is not None else model(batch["full_image"].to(device, non_blocking=True), batch["lesion_crop"].to(device, non_blocking=True))
                loss = _loss(logits, targets, loss_method, weights, focal_gamma)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            probability = torch.softmax(logits.detach(), 1)
            total_loss += loss.detach().item() * len(targets); count += len(targets)
            true.extend(targets.cpu().tolist()); predicted.extend(probability.argmax(1).cpu().tolist()); scores.extend(probability.cpu().tolist())
    return total_loss / max(count, 1), true, predicted, scores


def _metrics(task: str, true, predicted, scores) -> dict:
    result = lesion_presence_metrics(true, predicted, scores) if task == "lesion_presence" else classification_metrics(true, predicted, scores)
    return result


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int, class_names: list[str], weighted_sampling: bool = False) -> DataLoader:
    sampler = None
    if weighted_sampling:
        encoded = _targets(dataset.frame[dataset.target_column].tolist(), class_names).tolist()
        counts = Counter(encoded)
        sampler = WeightedRandomSampler([1.0 / counts[target] for target in encoded], len(encoded), replacement=True)
        shuffle = False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), collate_fn=collate_manifest_batch)
    loader.class_names = class_names
    return loader


def train_cnn_strategy(manifest, config: dict, strategy_name: str, architecture: str, run_name: str | None = None, box_provider=None) -> TrainingResult:
    """Train a CNN on train/validation development splits; DDI is rejected."""
    manifest = validate_manifest(manifest)
    task = config.get("task", "diagnosis_binary")
    if task not in {"diagnosis_binary", "diagnosis_multiclass", "lesion_presence"}:
        raise ValueError("CNN strategies support diagnosis_binary, diagnosis_multiclass, and lesion_presence")
    selected = select_task_manifest(manifest, task)
    if selected.dataset.fillna("").str.upper().eq("DDI").any():
        raise PermissionError("DDI cannot be used by development CNN training")
    if not {"train", "validation"}.issubset(set(selected.split.dropna())):
        raise ValueError("Manifest needs independent train and validation splits before training")
    if not leakage_report(selected).empty:
        raise ValueError("Patient/lesion/image groups cross development splits")
    validate_split_class_coverage(selected, {"diagnosis_binary": "binary_target", "diagnosis_multiclass": "harmonized_diagnosis", "lesion_presence": "lesion_present"}[task])
    class_names = class_names_for_manifest(selected, task)
    if len(class_names) < 2:
        suffix = " Add true normal-skin examples before lesion-presence training." if task == "lesion_presence" else ""
        raise ValueError("Training needs at least two target classes." + suffix)
    image_size, batch_size = int(config.get("image_size", 224)), int(config.get("batch_size", 16))
    train_transform, evaluation_transform = build_transforms(image_size, True, config.get("augmentation")), build_transforms(image_size, False)
    input_mode = config.get("input_mode", "full_image")
    if input_mode == "full_image":
        train_ds = ManifestImageDataset(selected, "train", train_transform, task)
        val_ds = ManifestImageDataset(selected, "validation", evaluation_transform, task)
    else:
        oracle = bool(config.get("ground_truth_crop_ablation", False))
        train_ds = CropAssistedManifestDataset(selected, "train", train_transform, task, crop_transform=train_transform, box_provider=box_provider, ground_truth_box=oracle, crop_margin=float(config.get("crop_margin", .15)), output_mode=input_mode)
        val_ds = CropAssistedManifestDataset(selected, "validation", evaluation_transform, task, crop_transform=evaluation_transform, box_provider=box_provider, ground_truth_box=oracle, crop_margin=float(config.get("crop_margin", .15)), output_mode=input_mode)
    workers = int(config.get("num_workers", 0))
    method = config.get("loss", "cross_entropy")
    train_loader = _loader(train_ds, batch_size, True, workers, class_names, method == "weighted_sampling")
    val_loader = _loader(val_ds, batch_size, False, workers, class_names)
    device = get_device(); model = build_diagnostic_input_model(input_mode, architecture, len(class_names), float(config.get("dropout", .2)), bool(config.get("pretrained", True))).to(device)
    set_backbone_trainable(model, False)
    encoded_train = _targets(train_ds.frame[train_ds.target_column].tolist(), class_names)
    weights = None
    if method in {"weighted_cross_entropy", "focal"}:
        counts = torch.bincount(encoded_train, minlength=len(class_names)).float()
        weights = (counts.sum() / (len(class_names) * counts.clamp_min(1))).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config.get("learning_rate", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=float(config.get("scheduler_factor", .5)), patience=int(config.get("scheduler_patience", 2)))
    epochs, head_epochs = int(config.get("epochs", 20)), int(config.get("head_epochs", 3))
    patience, stale = int(config.get("early_stopping_patience", 5)), 0
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_name = run_name or config.get("run_name") or f"{architecture}_{task}"
    checkpoint = checkpoint_path(strategy_name, architecture, run_name)
    history, best_epoch, best_score, best_state = [], None, float("-inf"), None
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        if epoch == head_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("fine_tuning_learning_rate", 3e-5)), weight_decay=float(config.get("weight_decay", 1e-5)))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=float(config.get("scheduler_factor", .5)), patience=int(config.get("scheduler_patience", 2)))
        train_loss, train_true, train_pred, _ = _epoch(model, train_loader, optimizer, scaler, device, method, weights, float(config.get("focal_gamma", 2)), True)
        val_loss, val_true, val_pred, val_scores = _epoch(model, val_loader, optimizer, scaler, device, method, weights, float(config.get("focal_gamma", 2)), False)
        train_metrics, val_metrics = _metrics(task, train_true, train_pred, None), _metrics(task, val_true, val_pred, val_scores)
        score = val_metrics[config.get("selection_metric", "macro_f1")]
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss, "train_accuracy": train_metrics["accuracy"], "validation_accuracy": val_metrics["accuracy"], "balanced_accuracy": val_metrics["balanced_accuracy"], "macro_precision": val_metrics["macro_precision"], "macro_recall": val_metrics["macro_recall"], "macro_f1": val_metrics["macro_f1"], "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(score)
        if score > best_score:
            best_score, best_epoch, stale, best_state = score, epoch, 0, deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch, checkpoint, config=config, class_names=class_names, task=task, strategy=strategy_name, architecture=architecture)
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    _, true, predicted, scores = _epoch(model, val_loader, optimizer, scaler, device, method, weights, float(config.get("focal_gamma", 2)), False)
    final_metrics = _metrics(task, true, predicted, scores)
    payload = {"run_name": run_name, "config": {**config, "strategy_name": strategy_name, "backbone": architecture, "class_names": class_names, "class_weights": weights.cpu().tolist() if weights is not None else None, "checkpoint_path": str(checkpoint), "device": str(device)}, "history": history, "best_epoch": best_epoch, "best_checkpoint": str(checkpoint), "metrics": final_metrics, "timing": {"training_seconds": perf_counter() - start}, "leakage_free": True, "status": "complete"}
    run_dir = create_run_directory(run_name)
    persist_run(payload, run_dir)
    return TrainingResult(history=history, validation_history=history, best_epoch=best_epoch, best_checkpoint=str(checkpoint), config=payload["config"], timing=payload["timing"], metrics=final_metrics)


def load_cnn_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    """Load a CNN checkpoint saved by this module for later development evaluation."""
    device = device or get_device()
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_diagnostic_input_model(state["config"].get("input_mode", "full_image"), state["architecture"], len(state["class_names"]), state["config"].get("dropout", .2), pretrained=False).to(device)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    return model, state
