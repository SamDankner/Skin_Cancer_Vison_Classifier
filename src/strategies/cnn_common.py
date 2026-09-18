"""Reusable training, evaluation, and persistence for CNN strategies."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.datasets import ManifestImageDataset, TASK_TARGETS, select_task_manifest, validate_manifest
from src.data.splits import leakage_report, validate_split_class_coverage
from src.data.transforms import build_transforms
from src.evaluation.evaluator import export_predictions, prediction_frame
from src.evaluation.metrics import classification_metrics, lesion_presence_metrics
from src.evaluation.reporting import plot_binary_curves, plot_confusion_matrix, plot_training_history
from src.training.checkpointing import checkpoint_path, save_checkpoint, update_checkpoint_metadata
from src.training.early_stopping import EarlyStopping
from src.training.experiment_runner import persist_run
from src.training.optimization import build_optimizer, build_scheduler, step_scheduler
from src.training.trainer import TrainingResult
from src.utils.device import get_device
from src.utils.seed import seed_everything
from src.utils.timing import benchmark_callable, environment_summary
from .localization import lesion_crop, parse_bounding_box


class FullImageCropFusion(nn.Module):
    """Fuse features from a full macro photograph and an intentional lesion crop."""

    def __init__(self, architecture: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        full = build_cnn_model(architecture, num_classes, dropout, pretrained)
        crop = build_cnn_model(architecture, num_classes, dropout, pretrained)
        self.full_features, self.crop_features = full.features, crop.features
        self.full_pool, self.crop_pool = full.avgpool, crop.avgpool
        if architecture == "efficientnet_v2_s":
            dimension = full.classifier[-1].in_features
        else:
            dimension = full.classifier[-1][-1].in_features
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimension * 2, num_classes))

    def _features(self, features, pool, image):
        return torch.flatten(pool(features(image)), 1)

    def forward(self, full_image, lesion_crop_image):
        """Return logits from paired full-image and crop tensors."""
        combined = torch.cat((
            self._features(self.full_features, self.full_pool, full_image),
            self._features(self.crop_features, self.crop_pool, lesion_crop_image),
        ), dim=1)
        return self.classifier(combined)


def build_diagnostic_input_model(
    input_mode: str,
    architecture: str,
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> nn.Module:
    """Build a full-image, crop-only, or original-plus-crop classifier."""
    if input_mode in {"full_image", "lesion_crop"}:
        return build_cnn_model(architecture, num_classes, dropout, pretrained)
    if input_mode == "full_plus_crop":
        return FullImageCropFusion(architecture, num_classes, dropout, pretrained)
    raise ValueError("input_mode must be full_image, lesion_crop, or full_plus_crop")


def collate_manifest_batch(batch: list[dict]) -> dict:
    """Collate tensors while retaining raw labels and row metadata."""
    output = {
        "target": [sample["target"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }
    for key in ("image", "full_image", "lesion_crop"):
        if key in batch[0]:
            output[key] = torch.stack([sample[key] for sample in batch])
    return output


class CropAssistedManifestDataset(ManifestImageDataset):
    """Return full photographs and crops from an explicit localization provider."""

    def __init__(
        self,
        *args,
        crop_transform,
        box_provider=None,
        ground_truth_box: bool = False,
        crop_margin: float = 0.15,
        output_mode: str = "full_plus_crop",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if box_provider is None and not ground_truth_box:
            raise ValueError(
                "Crop-assisted input requires an automatic box_provider; use ground_truth_box=True "
                "only for an explicit oracle ablation"
            )
        self.crop_transform = crop_transform
        self.box_provider = box_provider
        self.ground_truth_box = ground_truth_box
        self.crop_margin = crop_margin
        self.output_mode = output_mode

    def __getitem__(self, index: int) -> dict:
        """Load one source image and the explicitly sourced lesion crop."""
        row = self.frame.iloc[index]
        metadata = row.to_dict()
        raw_box = row.bounding_box if self.ground_truth_box else self.box_provider(metadata)
        box = parse_bounding_box(raw_box)
        if box is None:
            raise ValueError(f"No localization box available for image_id={row.image_id}")
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
            crop = lesion_crop(image, box, self.crop_margin)
            full_value = self.transform(image) if self.transform else image.copy()
        target = row[self.target_column]
        target = str(target) if self.target_column == "harmonized_diagnosis" else int(target)
        if self.output_mode == "lesion_crop":
            return {"image": self.crop_transform(crop), "target": target, "metadata": metadata}
        return {
            "full_image": full_value,
            "lesion_crop": self.crop_transform(crop),
            "target": target,
            "metadata": metadata,
        }


def class_names_for_manifest(manifest, task: str) -> list[str]:
    """Return deterministic class order from the task's training rows."""
    target = TASK_TARGETS[task]
    frame = select_task_manifest(manifest, task)
    values = frame.loc[frame.split.eq("train"), target].dropna().unique().tolist()
    return [str(value) for value in sorted(values, key=str)]


def _targets(values: Iterable[Any], class_names: list[str]) -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(class_names)}
    return torch.tensor([lookup[str(value)] for value in values], dtype=torch.long)


def build_cnn_model(
    architecture: str,
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> nn.Module:
    """Build EfficientNetV2-S or ConvNeXt-Tiny with a replacement head."""
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        EfficientNet_V2_S_Weights,
        convnext_tiny,
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
    """Freeze or unfreeze the feature extractor while keeping the head trainable."""
    # The replacement classifier must continue learning during the head-only
    # phase, even while pretrained image features are frozen.
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in _head_parameters(model):
        parameter.requires_grad = True


def _loss(logits, targets, method: str, class_weights, focal_gamma: float) -> torch.Tensor:
    base = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    if method == "focal":
        return ((1 - torch.exp(-base)).pow(focal_gamma) * base).mean()
    if method not in {"cross_entropy", "weighted_cross_entropy"}:
        raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal")
    return base.mean()


def _forward(model, batch, device):
    if "image" in batch:
        return model(batch["image"].to(device, non_blocking=True))
    return model(
        batch["full_image"].to(device, non_blocking=True),
        batch["lesion_crop"].to(device, non_blocking=True),
    )


def _epoch(model, loader, optimizer, scaler, device, config, weights, *, training: bool) -> dict:
    model.train(training)
    started = perf_counter()
    total_loss, count, true, scores, metadata = 0.0, 0, [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            targets = _targets(batch["target"], loader.class_names).to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            # CUDA autocast reduces activation precision where safe; GradScaler
            # below protects small gradients during the backward pass.
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = _forward(model, batch, device)
                loss = _loss(logits, targets, config["loss"], weights, float(config.get("focal_gamma", 2.0)))
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # Metrics consume normalized probabilities, never raw logits.
            probability = torch.softmax(logits.detach().float(), dim=1)
            total_loss += float(loss.detach()) * len(targets)
            count += len(targets)
            true.extend(targets.cpu().tolist())
            scores.extend(probability.cpu().tolist())
            metadata.extend(batch["metadata"])
    probabilities = np.asarray(scores, dtype=float)
    predictions = probabilities.argmax(axis=1)
    metric_function = lesion_presence_metrics if config["task"] == "lesion_presence" else classification_metrics
    metrics = metric_function(
        true,
        predictions,
        probabilities,
        labels=list(range(len(loader.class_names))),
        class_names=loader.class_names,
    )
    metrics["loss"] = total_loss / max(count, 1)
    return {
        "loss": total_loss / max(count, 1),
        "targets": np.asarray(true),
        "probabilities": probabilities,
        "metadata": metadata,
        "metrics": metrics,
        "seconds": perf_counter() - started,
    }


def _loader(dataset, config: dict, class_names: list[str], *, training: bool) -> DataLoader:
    sampler = None
    if training and config.get("weighted_sampling", False):
        encoded = _targets(dataset.frame[dataset.target_column].tolist(), class_names).tolist()
        counts = Counter(encoded)
        sampler = WeightedRandomSampler(
            [1.0 / counts[target] for target in encoded], len(encoded), replacement=True
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_manifest_batch,
    )
    loader.class_names = class_names
    return loader


def _datasets(selected, config, box_provider):
    train_transform = build_transforms(int(config["image_size"]), True, config.get("augmentation"))
    evaluation_transform = build_transforms(int(config["image_size"]), False)
    input_mode = config.get("input_mode", "full_image")
    datasets = {}
    for split in ("train", "validation", "test"):
        transform = train_transform if split == "train" else evaluation_transform
        if input_mode == "full_image":
            datasets[split] = ManifestImageDataset(selected, split, transform, config["task"])
        else:
            datasets[split] = CropAssistedManifestDataset(
                selected,
                split,
                transform,
                config["task"],
                crop_transform=transform,
                box_provider=box_provider,
                ground_truth_box=bool(config.get("ground_truth_crop_ablation", False)),
                crop_margin=float(config.get("crop_margin", 0.15)),
                output_mode=input_mode,
            )
    return datasets


def train_cnn_strategy(
    manifest,
    config: dict,
    strategy_name: str,
    architecture: str,
    run_name: str | None = None,
    box_provider=None,
) -> TrainingResult:
    """Train, select on validation, restore, and test a CNN development run."""
    experiment_started = perf_counter()
    cfg = {
        "strategy_name": strategy_name,
        "task": "diagnosis_binary",
        "backbone": architecture,
        "pretrained": True,
        "image_size": 224,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "fine_tuning_learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "epochs": 20,
        "head_epochs": 3,
        "dropout": 0.2,
        "loss": "cross_entropy",
        "selection_metric": "macro_f1",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "input_mode": "full_image",
        **config,
    }
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    task = cfg["task"]
    if task not in {"diagnosis_binary", "diagnosis_multiclass", "lesion_presence"}:
        raise ValueError("CNN strategies support diagnosis and lesion-presence tasks")
    selected = select_task_manifest(validate_manifest(manifest), task)
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(set(selected.split.dropna())):
        raise ValueError("Training requires independent train, validation, and development-test splits")
    if not leakage_report(selected).empty:
        raise ValueError("Patient/lesion/image groups cross development splits")
    validate_split_class_coverage(selected, TASK_TARGETS[task], ("train", "validation", "test"))
    class_names = class_names_for_manifest(selected, task)
    if len(class_names) < 2:
        suffix = " Add true normal-skin examples." if task == "lesion_presence" else ""
        raise ValueError("Training needs at least two target classes." + suffix)

    datasets = _datasets(selected, cfg, box_provider)
    loaders = {
        split: _loader(dataset, cfg, class_names, training=split == "train")
        for split, dataset in datasets.items()
    }
    device = get_device()
    model = build_diagnostic_input_model(
        cfg["input_mode"], architecture, len(class_names), float(cfg["dropout"]), bool(cfg["pretrained"])
    ).to(device)

    # Begin by adapting only the replacement head to the project data.
    set_backbone_trainable(model, False)
    encoded_train = _targets(datasets["train"].frame[datasets["train"].target_column].tolist(), class_names)
    weights = None
    if cfg["loss"] in {"weighted_cross_entropy", "focal"}:
        counts = torch.bincount(encoded_train, minlength=len(class_names)).float()
        weights = (counts.sum() / (len(class_names) * counts.clamp_min(1))).to(device)
    optimizer = build_optimizer(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        cfg,
        default_lr=float(cfg["learning_rate"]),
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    monitor = str(cfg["selection_metric"])
    selection_mode = str(cfg.get("selection_mode", "min" if "loss" in monitor else "max"))
    stopper = EarlyStopping(
        int(cfg["early_stopping_patience"]),
        float(cfg["early_stopping_min_delta"]),
        selection_mode,
        f"validation_{monitor}",
    )
    run_name = run_name or cfg.get("run_name") or f"{architecture}_{task}"
    checkpoint = checkpoint_path(strategy_name, architecture, run_name, cfg.get("models_root", "models"))
    run_directory = Path(cfg.get("run_directory") or Path("results/runs") / run_name)
    history, epoch_seconds, validation_seconds = [], [], 0.0
    training_started = perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if epoch == int(cfg["head_epochs"]) + 1:
            # Fine-tuning starts with a new optimizer so the lower configured
            # learning rate applies to the now-trainable pretrained backbone.
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(
                model.parameters(), cfg, default_lr=float(cfg["fine_tuning_learning_rate"])
            )
            scheduler = build_scheduler(optimizer, cfg)
        epoch_started = perf_counter()
        training = _epoch(model, loaders["train"], optimizer, scaler, device, cfg, weights, training=True)
        validation = _epoch(model, loaders["validation"], optimizer, scaler, device, cfg, weights, training=False)
        validation_seconds += validation["seconds"]
        score = validation["loss"] if "loss" in monitor else validation["metrics"].get(monitor)
        if score is None:
            raise ValueError(f"Validation metric {monitor!r} is undefined")
        step_scheduler(scheduler, cfg, score)
        epoch_seconds.append(perf_counter() - epoch_started)
        history.append({
            "epoch": epoch,
            "train_loss": training["loss"],
            "validation_loss": validation["loss"],
            "train_accuracy": training["metrics"]["accuracy"],
            "validation_accuracy": validation["metrics"]["accuracy"],
            "train_macro_f1": training["metrics"]["macro_f1"],
            "validation_macro_f1": validation["metrics"]["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds[-1],
        })
        if stopper.update(score, epoch, model.state_dict()):
            save_checkpoint(
                model, optimizer, epoch, checkpoint,
                config={**cfg, "class_names": class_names},
                class_names=class_names,
                task=task,
                strategy=strategy_name,
                architecture=architecture,
                threshold=cfg.get("threshold"),
                calibration=cfg.get("calibration"),
                preprocessing={"image_size": int(cfg["image_size"]), "transform": "imagenet_normalization"},
                checkpoint_identifier=checkpoint.stem,
            )
        if stopper.should_stop:
            break
    training_seconds = perf_counter() - training_started
    if stopper.best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")

    # Development-test metrics must use the validation-selected state, not the
    # final epoch, which may have already begun to overfit.
    model.load_state_dict(stopper.best_state)
    validation = _epoch(model, loaders["validation"], optimizer, scaler, device, cfg, weights, training=False)
    evaluation_started = perf_counter()
    development_test = _epoch(model, loaders["test"], optimizer, scaler, device, cfg, weights, training=False)
    evaluation_seconds = perf_counter() - evaluation_started
    stop_summary = stopper.summary(len(history), int(cfg["epochs"]))
    update_checkpoint_metadata(checkpoint, early_stopping=stop_summary)

    sample = collate_manifest_batch([datasets["test"][0]])
    for key in ("image", "full_image", "lesion_crop"):
        if key in sample:
            sample[key] = sample[key].to(device)
    inference = benchmark_callable(
        lambda: _forward(model, sample, device),
        device=device,
        warmup=int(cfg.get("inference_warmup", 3)),
        repeats=int(cfg.get("inference_repeats", 20)),
    )
    timing = {
        "training_seconds": training_seconds,
        "epoch_seconds": epoch_seconds,
        "validation_seconds": validation_seconds,
        "evaluation_seconds": evaluation_seconds,
        "inference_model_only_batch_size_1": inference,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    threshold = cfg.get("threshold")
    threshold_value = threshold.get("threshold") if isinstance(threshold, dict) else threshold
    predictions = prediction_frame(
        development_test["targets"], development_test["probabilities"], development_test["metadata"],
        class_order=class_names, task=task, strategy=strategy_name,
        checkpoint_id=checkpoint.stem, threshold=threshold_value, calibrated=False,
    )
    export_predictions(predictions, run_directory / "development_test_predictions.csv")
    plot_training_history(pd.DataFrame(history), run_directory / "training_history.png")
    plot_confusion_matrix(development_test["metrics"]["confusion_matrix"], class_names, run_directory / "confusion_matrix.png")
    if len(class_names) == 2:
        plot_binary_curves(development_test["targets"], development_test["probabilities"][:, 1], run_directory / "roc_pr_calibration.png")
    (run_directory / "development_test_metrics.json").write_text(
        json.dumps(development_test["metrics"], indent=2), encoding="utf-8"
    )
    split_summary = selected.groupby(["split", TASK_TARGETS[task]], dropna=False).size().rename("count").reset_index().to_dict("records")
    timing["total_experiment_seconds"] = perf_counter() - experiment_started
    payload = {
        "run_name": run_name,
        "config": {
            **cfg,
            "class_names": class_names,
            "class_weights": weights.cpu().tolist() if weights is not None else None,
            "checkpoint_path": str(checkpoint),
            "device": str(device),
        },
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_checkpoint": str(checkpoint),
        "metrics": validation["metrics"],
        "development_test_metrics": development_test["metrics"],
        "early_stopping": stop_summary,
        "timing": timing,
        "seed": int(cfg.get("seed", 42)),
        "dataset_split_summary": split_summary,
        "environment": environment_summary(),
        "leakage_free": True,
        "status": "complete",
    }
    persist_run(payload, run_directory, cfg.get("summary_path", "results/experiment_summary.csv"))
    return TrainingResult(
        history=history,
        validation_history=history,
        best_epoch=stopper.best_epoch,
        best_checkpoint=str(checkpoint),
        config=payload["config"],
        timing=timing,
        metrics=validation["metrics"],
        development_test_metrics=development_test["metrics"],
        early_stopping=stop_summary,
        run_directory=str(run_directory),
    )


def load_cnn_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    """Reload a CNN and its persisted inference/training contract."""
    device = device or get_device()
    state = torch.load(path, map_location=device, weights_only=False)
    config = state.get("config", {})
    model = build_diagnostic_input_model(
        config.get("input_mode", "full_image"),
        state.get("architecture") or config["backbone"],
        len(state["class_names"]),
        float(config.get("dropout", 0.2)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state
