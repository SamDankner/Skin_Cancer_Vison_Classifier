"""Train, evaluate, persist, and reload DINOv2 image classifiers."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.datasets import ManifestImageDataset, TASK_TARGETS, select_task_manifest, validate_manifest
from src.data.splits import leakage_report, validate_split_class_coverage
from src.data.transforms import build_transforms
from src.evaluation.evaluator import export_predictions, prediction_frame
from src.evaluation.metrics import classification_metrics
from src.evaluation.reporting import plot_binary_curves, plot_confusion_matrix, plot_training_history
from src.training.checkpointing import checkpoint_path, save_checkpoint, update_checkpoint_metadata
from src.training.early_stopping import EarlyStopping
from src.training.experiment_runner import persist_run
from src.training.optimization import build_optimizer, build_scheduler, step_scheduler
from src.training.trainer import TrainingResult
from src.utils.device import get_device
from src.utils.seed import seed_everything
from src.utils.timing import benchmark_callable, environment_summary


def load_dinov2_backbone(name: str = "dinov2_vits14", pretrained: bool = True) -> nn.Module:
    """Load a DINOv2 hub backbone only when a run or reload is requested."""
    try:
        return torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
    except Exception as exc:
        raise RuntimeError(
            "Could not load DINOv2. Cache the weights or provide an offline backbone_factory."
        ) from exc


def backbone_embedding_dim(backbone: nn.Module) -> int:
    """Resolve the image embedding width exposed by a backbone."""
    for attribute in ("embed_dim", "num_features", "hidden_dim"):
        value = getattr(backbone, attribute, None)
        if isinstance(value, int):
            return value
    with torch.no_grad():
        output = backbone(torch.zeros(1, 3, 224, 224))
    if isinstance(output, dict):
        output = output["x_norm_clstoken"] if "x_norm_clstoken" in output else next(iter(output.values()))
    return int(output.shape[-1])


def set_backbone_trainability(backbone: nn.Module, unfreeze_last_blocks: int = 0) -> None:
    """Freeze DINOv2, optionally unfreezing its final blocks and norm."""
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if unfreeze_last_blocks <= 0:
        return
    blocks = list(getattr(backbone, "blocks", []))
    if not blocks:
        raise ValueError("Selected backbone has no exposed transformer blocks")
    for block in blocks[-unfreeze_last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for name in ("norm", "fc_norm"):
        layer = getattr(backbone, name, None)
        if layer:
            for parameter in layer.parameters():
                parameter.requires_grad = True


class DinoV2Classifier(nn.Module):
    """DINOv2 encoder with a compact configurable classification head."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.2, head_width: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = backbone_embedding_dim(backbone)
        width = head_width or self.embedding_dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def encode_image(self, image):
        """Return the backbone image-level representation."""
        output = self.backbone(image)
        if not isinstance(output, dict):
            return output
        return output["x_norm_clstoken"] if "x_norm_clstoken" in output else next(iter(output.values()))

    def forward(self, image):
        """Return class logits for an image batch."""
        return self.head(self.encode_image(image))


class _EncodedManifestDataset(Dataset):
    """Wrap manifest images with persisted integer target encoding."""

    def __init__(self, manifest, transform, task, class_to_index):
        self.base = ManifestImageDataset(manifest, transform=transform, task=task)
        self.frame = self.base.frame
        self.class_to_index = {str(key): value for key, value in class_to_index.items()}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        return {**item, "target": self.class_to_index[str(item["target"])]}


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "metadata": [item["metadata"] for item in batch],
    }


def make_image_loaders(
    manifest,
    task,
    image_size,
    batch_size,
    augmentation=None,
    weighted_sampling=False,
    num_workers=0,
):
    """Build train/validation/test loaders from existing group-safe splits."""
    frame = select_task_manifest(validate_manifest(manifest), task)
    if not {"train", "validation", "test"}.issubset(set(frame.split.dropna())):
        raise ValueError("Manifest must have train, validation, and development-test splits")
    if not leakage_report(frame).empty:
        raise ValueError("Patient/lesion/image groups cross development splits")
    validate_split_class_coverage(frame, TASK_TARGETS[task], ("train", "validation", "test"))
    labels = sorted(frame.loc[frame.split.eq("train"), TASK_TARGETS[task]].unique().tolist(), key=str)
    class_to_index = {str(label): index for index, label in enumerate(labels)}
    if len(class_to_index) < 2:
        raise ValueError("Training split needs at least two classes")
    datasets = {
        split: _EncodedManifestDataset(
            frame.loc[frame.split.eq(split)],
            build_transforms(image_size, split == "train", augmentation if split == "train" else None),
            task,
            class_to_index,
        )
        for split in ("train", "validation", "test")
    }
    sampler = None
    if weighted_sampling:
        targets = [datasets["train"][index]["target"] for index in range(len(datasets["train"]))]
        counts = np.bincount(targets, minlength=len(labels))
        sampler = WeightedRandomSampler([1.0 / counts[target] for target in targets], len(targets), replacement=True)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler if split == "train" else None,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
        )
        for split, dataset in datasets.items()
    }
    return loaders, class_to_index


def _loss(logits, target, name, weights, focal_gamma):
    base = nn.functional.cross_entropy(logits, target, weight=weights, reduction="none")
    if name == "focal":
        return ((1 - torch.exp(-base)).pow(focal_gamma) * base).mean()
    if name not in {"cross_entropy", "weighted_cross_entropy"}:
        raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal")
    return base.mean()


def _phase(model, loader, optimizer, scaler, device, config, weights, *, training: bool) -> dict:
    model.train(training)
    started = perf_counter()
    total, targets, probabilities, metadata = 0.0, [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(image)
                loss = _loss(logits, target, config["loss"], weights, float(config.get("focal_gamma", 2.0)))
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            probability = logits.detach().softmax(1)
            total += float(loss.detach()) * len(target)
            targets.extend(target.cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            metadata.extend(batch["metadata"])
    probability_array = np.asarray(probabilities, dtype=float)
    metrics = classification_metrics(
        targets,
        probability_array.argmax(axis=1),
        probability_array,
        labels=list(range(probability_array.shape[1])),
        class_names=list(loader.dataset.class_to_index),
    )
    metrics["loss"] = total / max(len(targets), 1)
    return {
        "loss": total / max(len(targets), 1),
        "targets": np.asarray(targets),
        "probabilities": probability_array,
        "metadata": metadata,
        "metrics": metrics,
        "seconds": perf_counter() - started,
    }


def train_dinov2(
    manifest,
    config: dict,
    backbone_factory: Callable[..., nn.Module] = load_dinov2_backbone,
) -> TrainingResult:
    """Run the complete validation-selected DINOv2 development lifecycle."""
    experiment_started = perf_counter()
    cfg = {
        "strategy_name": "dinov2",
        "task": "diagnosis_binary",
        "backbone": "dinov2_vits14",
        "pretrained": True,
        "image_size": 224,
        "batch_size": 8,
        "epochs": 12,
        "head_learning_rate": 3e-4,
        "backbone_learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "unfreeze_last_blocks": 0,
        "loss": "weighted_cross_entropy",
        "selection_metric": "macro_f1",
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.0,
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        **config,
    }
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    if cfg["task"] not in {"diagnosis_binary", "diagnosis_multiclass"}:
        raise ValueError("DINOv2 strategy supports diagnosis tasks only")
    if str(cfg["backbone"]).startswith("dinov2") and int(cfg["image_size"]) % 14:
        raise ValueError("DINOv2 image_size must be divisible by its 14-pixel patch size")
    device = get_device()
    loaders, class_to_index = make_image_loaders(
        manifest,
        cfg["task"],
        int(cfg["image_size"]),
        int(cfg["batch_size"]),
        cfg.get("augmentation"),
        bool(cfg.get("weighted_sampling", False)),
        int(cfg.get("num_workers", 0)),
    )
    class_names = list(class_to_index)
    backbone = backbone_factory(cfg["backbone"], pretrained=cfg["pretrained"])
    set_backbone_trainability(backbone, int(cfg["unfreeze_last_blocks"]))
    model = DinoV2Classifier(backbone, len(class_names), float(cfg["dropout"]), cfg.get("head_width")).to(device)
    parameter_groups = [{"params": model.head.parameters(), "lr": float(cfg["head_learning_rate"])}]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": float(cfg["backbone_learning_rate"])})
    optimizer = build_optimizer(parameter_groups, cfg, default_lr=float(cfg["head_learning_rate"]))
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = np.bincount(
        [loaders["train"].dataset[index]["target"] for index in range(len(loaders["train"].dataset))],
        minlength=len(class_names),
    )
    weights = None
    if cfg["loss"] in {"weighted_cross_entropy", "focal"}:
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    monitor = str(cfg["selection_metric"])
    selection_mode = str(cfg.get("selection_mode", "min" if "loss" in monitor else "max"))
    stopper = EarlyStopping(
        int(cfg["early_stopping_patience"]), float(cfg["early_stopping_min_delta"]),
        selection_mode, f"validation_{monitor}",
    )
    run_name = cfg.get("run_name", f"{cfg['task']}_{cfg['backbone']}")
    path = checkpoint_path("dinov2", cfg["backbone"], run_name, cfg.get("models_root", "models"))
    run_directory = Path(cfg.get("run_directory") or Path("results/runs") / run_name)
    history, epoch_seconds, validation_seconds = [], [], 0.0
    training_started = perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_started = perf_counter()
        training = _phase(model, loaders["train"], optimizer, scaler, device, cfg, weights, training=True)
        validation = _phase(model, loaders["validation"], optimizer, scaler, device, cfg, weights, training=False)
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
                model, optimizer, epoch, path,
                config={**cfg, "class_names": class_names},
                class_names=class_names, class_to_index=class_to_index,
                task=cfg["task"], strategy="dinov2", architecture=cfg["backbone"],
                threshold=cfg.get("threshold"), calibration=cfg.get("calibration"),
                preprocessing={"image_size": int(cfg["image_size"]), "transform": "imagenet_normalization"},
                checkpoint_identifier=path.stem,
            )
        if stopper.should_stop:
            break
    training_seconds = perf_counter() - training_started
    if stopper.best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")
    model.load_state_dict(stopper.best_state)
    validation = _phase(model, loaders["validation"], optimizer, scaler, device, cfg, weights, training=False)
    evaluation_started = perf_counter()
    development_test = _phase(model, loaders["test"], optimizer, scaler, device, cfg, weights, training=False)
    evaluation_seconds = perf_counter() - evaluation_started
    stop_summary = stopper.summary(len(history), int(cfg["epochs"]))
    update_checkpoint_metadata(path, early_stopping=stop_summary)

    sample = _collate([loaders["test"].dataset[0]])
    image = sample["image"].to(device)
    inference = benchmark_callable(
        lambda: model(image), device=device,
        warmup=int(cfg.get("inference_warmup", 3)), repeats=int(cfg.get("inference_repeats", 20)),
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
        class_order=class_names, task=cfg["task"], strategy="dinov2",
        checkpoint_id=path.stem, threshold=threshold_value, calibrated=False,
    )
    export_predictions(predictions, run_directory / "development_test_predictions.csv")
    plot_training_history(pd.DataFrame(history), run_directory / "training_history.png")
    plot_confusion_matrix(development_test["metrics"]["confusion_matrix"], class_names, run_directory / "confusion_matrix.png")
    if len(class_names) == 2:
        plot_binary_curves(development_test["targets"], development_test["probabilities"][:, 1], run_directory / "roc_pr_calibration.png")
    (run_directory / "development_test_metrics.json").write_text(
        json.dumps(development_test["metrics"], indent=2), encoding="utf-8"
    )
    frame = select_task_manifest(validate_manifest(manifest), cfg["task"])
    split_summary = frame.groupby(["split", TASK_TARGETS[cfg["task"]]], dropna=False).size().rename("count").reset_index().to_dict("records")
    timing["total_experiment_seconds"] = perf_counter() - experiment_started
    payload = {
        "run_name": run_name,
        "config": {**cfg, "class_names": class_names, "checkpoint_path": str(path), "device": str(device)},
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_checkpoint": str(path),
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
        history=history, validation_history=history, best_epoch=stopper.best_epoch,
        best_checkpoint=str(path), config=payload["config"], timing=timing,
        metrics=validation["metrics"], development_test_metrics=development_test["metrics"],
        early_stopping=stop_summary, run_directory=str(run_directory),
    )


def load_dinov2_checkpoint(
    path: str | Path,
    backbone_factory=load_dinov2_backbone,
    map_location="cpu",
):
    """Reload a saved DINOv2 classifier and its persisted contract."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = payload["config"]
    model = DinoV2Classifier(
        backbone_factory(config["backbone"], pretrained=False),
        len(payload.get("class_names") or payload["class_to_index"]),
        float(config.get("dropout", 0.2)),
        config.get("head_width"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location).eval()
    return model, payload


__all__ = [
    "DinoV2Classifier", "backbone_embedding_dim", "load_dinov2_backbone",
    "load_dinov2_checkpoint", "make_image_loaders", "set_backbone_trainability",
    "train_dinov2",
]
