"""Train, persist, reload, and infer with image-plus-metadata classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

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
from src.strategies.dinov2_strategy import (
    DinoV2Classifier,
    backbone_embedding_dim,
    load_dinov2_backbone,
    set_backbone_trainability,
)
from src.training.checkpointing import checkpoint_path, save_checkpoint, update_checkpoint_metadata
from src.training.early_stopping import EarlyStopping
from src.training.experiment_runner import persist_run
from src.training.optimization import build_optimizer, build_scheduler, step_scheduler
from src.training.trainer import TrainingResult
from src.utils.device import get_device
from src.utils.timing import benchmark_callable, environment_summary


SAFE_METADATA_FIELDS = ("age", "sex", "anatomical_site", "skin_tone")
LEAKAGE_PRONE_FIELDS = (
    "pathology", "diagnosis", "ground_truth", "biopsy", "treatment",
    "assessment", "target", "label",
)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a column or an equally sized missing-value series."""
    if name in frame:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


@dataclass
class MetadataPreprocessor:
    """Fit safe encodings on training rows and retain explicit missing indicators."""

    fields: tuple[str, ...] = SAFE_METADATA_FIELDS
    age_mean: float = 0.0
    age_std: float = 1.0
    vocabularies: dict[str, dict[str, int]] | None = None

    def _validate_fields(self) -> None:
        invalid = set(self.fields) - set(SAFE_METADATA_FIELDS)
        if invalid:
            raise ValueError(f"Metadata fields are not allow-listed: {sorted(invalid)}")
        suspicious = [field for field in self.fields if any(token in field.lower() for token in LEAKAGE_PRONE_FIELDS)]
        if suspicious:
            raise ValueError(f"Leakage-prone metadata fields are prohibited: {suspicious}")

    def fit(self, train_frame: pd.DataFrame) -> "MetadataPreprocessor":
        """Fit normalization and vocabularies using training rows only."""
        self._validate_fields()
        self.vocabularies = {}
        if "age" in self.fields:
            ages = pd.to_numeric(_column(train_frame, "age"), errors="coerce")
            self.age_mean = float(ages.mean()) if ages.notna().any() else 0.0
            self.age_std = float(ages.std()) if ages.notna().sum() > 1 else 1.0
            if not np.isfinite(self.age_std) or self.age_std == 0:
                self.age_std = 1.0
        for field in self.fields:
            if field == "age":
                continue
            values = _column(train_frame, field).fillna("<MISSING>").astype(str).str.strip().replace("", "<MISSING>")
            observed = sorted(value for value in values.unique() if value != "<MISSING>")
            self.vocabularies[field] = {value: index + 2 for index, value in enumerate(observed)}
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Transform rows without inventing medically meaningful metadata."""
        if self.vocabularies is None:
            raise RuntimeError("Call fit on the training split before transform")
        result: dict[str, torch.Tensor] = {}
        if "age" in self.fields:
            age = pd.to_numeric(_column(frame, "age"), errors="coerce")
            normalized = (age.fillna(self.age_mean) - self.age_mean) / self.age_std
            result["continuous"] = torch.tensor(
                np.column_stack((normalized, age.isna().astype(float))), dtype=torch.float32
            )
        else:
            result["continuous"] = torch.empty((len(frame), 0), dtype=torch.float32)
        for field, vocabulary in self.vocabularies.items():
            values = _column(frame, field).fillna("<MISSING>").astype(str).str.strip().replace("", "<MISSING>")
            result[field] = torch.tensor(
                [0 if value == "<MISSING>" else vocabulary.get(value, 1) for value in values],
                dtype=torch.long,
            )
        return result

    def config(self) -> dict:
        """Return the complete serializable preprocessing contract."""
        return {
            "fields": list(self.fields),
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "vocabularies": self.vocabularies,
            "missing_value_policy": {
                "age": "training_mean_with_missing_indicator",
                "categorical": "dedicated_missing_id_0",
                "unknown_category": "dedicated_unknown_id_1",
            },
        }

    @classmethod
    def from_config(cls, config: dict) -> "MetadataPreprocessor":
        """Recreate a fitted preprocessor from checkpoint metadata."""
        return cls(
            tuple(config["fields"]),
            float(config.get("age_mean", 0.0)),
            float(config.get("age_std", 1.0)),
            config.get("vocabularies") or {},
        )


class MetadataEncoder(nn.Module):
    """Encode normalized continuous values and categorical metadata IDs."""

    def __init__(self, preprocessor: MetadataPreprocessor, embedding_dim: int = 16, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fields = tuple(preprocessor.fields)
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(len(vocabulary) + 2, embedding_dim)
            for field, vocabulary in (preprocessor.vocabularies or {}).items()
        })
        input_dim = (2 if "age" in self.fields else 0) + embedding_dim * len(self.embeddings)
        if input_dim == 0:
            raise ValueError("A multimodal model needs at least one metadata field")
        self.network = nn.Sequential(
            nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(width)
        )

    def forward(self, metadata: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one metadata embedding per input row."""
        values = [metadata["continuous"]]
        values.extend(self.embeddings[field](metadata[field]) for field in self.embeddings)
        return self.network(torch.cat(values, dim=1))


class MultimodalClassifier(nn.Module):
    """Fuse a clinical-photo embedding with an allow-listed metadata embedding."""

    def __init__(
        self,
        backbone: nn.Module,
        preprocessor: MetadataPreprocessor,
        num_classes: int,
        metadata_embedding_dim: int = 16,
        metadata_width: int = 64,
        fusion_width: int = 256,
        dropout: float = 0.2,
        metadata_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.image_dim = backbone_embedding_dim(backbone)
        self.metadata_encoder = MetadataEncoder(
            preprocessor, metadata_embedding_dim, metadata_width, metadata_dropout
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.image_dim + metadata_width),
            nn.Linear(self.image_dim + metadata_width, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, num_classes),
        )
        self.metadata_preprocessor = preprocessor

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return the backbone's image-level feature representation."""
        embedding = self.backbone(image)
        if isinstance(embedding, dict):
            embedding = embedding.get("x_norm_clstoken", next(iter(embedding.values())))
        return embedding

    def forward(self, image: torch.Tensor, metadata: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return class logits for image and metadata batches."""
        fused = torch.cat((self.encode_image(image), self.metadata_encoder(metadata)), dim=1)
        return self.classifier(fused)


class _MultimodalDataset(Dataset):
    """Attach encoded targets while retaining raw metadata for export."""

    def __init__(self, frame: pd.DataFrame, transform, task: str, class_to_index: dict):
        self.base = ManifestImageDataset(frame, transform=transform, task=task)
        self.frame = self.base.frame
        self.class_to_index = {str(key): value for key, value in class_to_index.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        return {**item, "target": self.class_to_index[str(item["target"])]}


def _collate_multimodal(batch: list[dict], processor: MetadataPreprocessor) -> dict:
    raw_metadata = [item["metadata"] for item in batch]
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "metadata": raw_metadata,
        "metadata_tensors": processor.transform(pd.DataFrame(raw_metadata)),
    }


def _make_loaders(frame: pd.DataFrame, config: dict, processor: MetadataPreprocessor, class_to_index: dict) -> dict[str, DataLoader]:
    datasets = {
        split: _MultimodalDataset(
            frame.loc[frame.split.eq(split)],
            build_transforms(
                int(config["image_size"]),
                training=split == "train",
                augmentation=config.get("augmentation") if split == "train" else None,
            ),
            config["task"],
            class_to_index,
        )
        for split in ("train", "validation", "test")
    }
    sampler = None
    if config.get("weighted_sampling", False):
        targets = [datasets["train"][index]["target"] for index in range(len(datasets["train"]))]
        counts = np.bincount(targets, minlength=len(class_to_index))
        sampler = WeightedRandomSampler([1.0 / counts[target] for target in targets], len(targets), replacement=True)
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train" and sampler is None,
            sampler=sampler if split == "train" else None,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=partial(_collate_multimodal, processor=processor),
        )
    return loaders


def _loss(logits, targets, name: str, weights, focal_gamma: float) -> torch.Tensor:
    base = nn.functional.cross_entropy(logits, targets, weight=weights, reduction="none")
    if name == "focal":
        return ((1 - torch.exp(-base)).pow(focal_gamma) * base).mean()
    if name not in {"cross_entropy", "weighted_cross_entropy"}:
        raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal")
    return base.mean()


def _run_epoch(model, loader, optimizer, scaler, device, config, weights, *, training: bool) -> dict:
    model.train(training)
    started = perf_counter()
    total_loss, targets, probabilities, metadata = 0.0, [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            encoded_metadata = {
                key: value.to(device, non_blocking=True) for key, value in batch["metadata_tensors"].items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(image, encoded_metadata)
                loss = _loss(logits, target, config["loss"], weights, float(config.get("focal_gamma", 2.0)))
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            probability = logits.detach().float().softmax(1)
            total_loss += float(loss.detach()) * len(target)
            targets.extend(target.cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            metadata.extend(batch["metadata"])
    probability_array = np.asarray(probabilities, dtype=float)
    prediction_array = probability_array.argmax(axis=1)
    metrics = classification_metrics(
        targets,
        prediction_array,
        probability_array,
        labels=list(range(len(loader.dataset.class_to_index))),
    )
    metrics["loss"] = total_loss / max(len(targets), 1)
    return {
        "loss": total_loss / max(len(targets), 1),
        "targets": np.asarray(targets),
        "probabilities": probability_array,
        "metadata": metadata,
        "metrics": metrics,
        "seconds": perf_counter() - started,
    }


def make_metadata_tensors(manifest: pd.DataFrame, fields: Iterable[str] = SAFE_METADATA_FIELDS):
    """Fit safe processing on the training split and transform supplied rows."""
    processor = MetadataPreprocessor(tuple(fields)).fit(manifest.loc[manifest.split.eq("train")])
    return processor, processor.transform(manifest)


def image_only_baseline(backbone, num_classes: int, dropout: float = 0.2):
    """Create a controlled image-only counterpart using the same encoder."""
    return DinoV2Classifier(backbone, num_classes, dropout)


def build_multimodal_model(
    config: dict,
    preprocessor: MetadataPreprocessor,
    num_classes: int,
    backbone_factory: Callable[..., nn.Module] | None = None,
) -> MultimodalClassifier:
    """Construct a late-fusion classifier from a persisted preprocessing contract."""
    cfg = {
        "backbone": "dinov2_vits14",
        "pretrained": True,
        "unfreeze_last_blocks": 0,
        "metadata_embedding_dim": 16,
        "metadata_width": 64,
        "fusion_width": 256,
        "dropout": 0.2,
        "metadata_dropout": 0.1,
        **config,
    }
    factory = backbone_factory or load_dinov2_backbone
    backbone = factory(cfg["backbone"], pretrained=cfg["pretrained"])
    set_backbone_trainability(backbone, int(cfg["unfreeze_last_blocks"]))
    return MultimodalClassifier(
        backbone,
        preprocessor,
        num_classes,
        int(cfg["metadata_embedding_dim"]),
        int(cfg["metadata_width"]),
        int(cfg["fusion_width"]),
        float(cfg["dropout"]),
        float(cfg["metadata_dropout"]),
    )


def train(
    manifest: pd.DataFrame,
    config: dict,
    run_name: str | None = None,
    backbone_factory: Callable[..., nn.Module] | None = None,
) -> TrainingResult:
    """Run the complete multimodal train/validate/test/checkpoint lifecycle."""
    experiment_started = perf_counter()
    cfg = {
        "strategy_name": "multimodal",
        "task": "diagnosis_binary",
        "backbone": "dinov2_vits14",
        "pretrained": True,
        "image_size": 224,
        "batch_size": 8,
        "epochs": 20,
        "head_learning_rate": 3e-4,
        "backbone_learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "metadata_fields": list(SAFE_METADATA_FIELDS),
        "metadata_embedding_dim": 16,
        "metadata_width": 64,
        "fusion_width": 256,
        "dropout": 0.2,
        "unfreeze_last_blocks": 0,
        "loss": "weighted_cross_entropy",
        "selection_metric": "macro_f1",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        **config,
    }
    if cfg["strategy_name"] != "multimodal":
        raise ValueError("Multimodal training requires strategy_name='multimodal'")
    if cfg["task"] not in {"diagnosis_binary", "diagnosis_multiclass"}:
        raise ValueError("Multimodal training supports diagnosis tasks only")
    if str(cfg["backbone"]).startswith("dinov2") and int(cfg["image_size"]) % 14:
        raise ValueError("DINOv2 image_size must be divisible by its 14-pixel patch size")
    fields = tuple(cfg.get("metadata_fields") or ())
    if not fields:
        raise ValueError("Use the DINOv2 strategy for an image-only ablation")
    checked = select_task_manifest(validate_manifest(manifest), cfg["task"])
    if not {"train", "validation", "test"}.issubset(set(checked.split.dropna())):
        raise ValueError("Multimodal training requires independent train, validation, and development-test splits")
    if not leakage_report(checked).empty:
        raise ValueError("Patient/lesion/image groups cross development splits")
    target_column = TASK_TARGETS[cfg["task"]]
    validate_split_class_coverage(checked, target_column, ("train", "validation", "test"))
    train_labels = sorted(checked.loc[checked.split.eq("train"), target_column].unique().tolist(), key=str)
    class_names = [str(label) for label in train_labels]
    class_to_index = {label: index for index, label in enumerate(class_names)}
    processor = MetadataPreprocessor(fields).fit(checked.loc[checked.split.eq("train")])
    loaders = _make_loaders(checked, cfg, processor, class_to_index)

    device = get_device()
    model = build_multimodal_model(cfg, processor, len(class_names), backbone_factory).to(device)
    head_parameters = list(model.metadata_encoder.parameters()) + list(model.classifier.parameters())
    groups = [{"params": head_parameters, "lr": float(cfg["head_learning_rate"])}]
    trainable_backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if trainable_backbone:
        groups.append({"params": trainable_backbone, "lr": float(cfg["backbone_learning_rate"])})
    optimizer = build_optimizer(groups, cfg, default_lr=float(cfg["head_learning_rate"]))
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
        int(cfg["early_stopping_patience"]),
        float(cfg["early_stopping_min_delta"]),
        selection_mode,
        f"validation_{monitor}",
    )
    run_name = run_name or cfg.get("run_name") or f"multimodal_{cfg['task']}"
    checkpoint = checkpoint_path("multimodal", cfg["backbone"], run_name, cfg.get("models_root", "models"))
    run_directory = Path(cfg.get("run_directory") or Path("results/runs") / run_name)
    history, validation_seconds, epoch_seconds = [], 0.0, []
    training_started = perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_started = perf_counter()
        training = _run_epoch(model, loaders["train"], optimizer, scaler, device, cfg, weights, training=True)
        validation = _run_epoch(model, loaders["validation"], optimizer, scaler, device, cfg, weights, training=False)
        validation_seconds += validation["seconds"]
        score = validation["loss"] if "loss" in monitor else validation["metrics"].get(monitor)
        if score is None:
            raise ValueError(f"Validation selection metric {monitor!r} is undefined")
        step_scheduler(scheduler, cfg, score)
        epoch_seconds.append(perf_counter() - epoch_started)
        history.append({
            "epoch": epoch,
            "train_loss": training["loss"],
            "validation_loss": validation["loss"],
            "train_macro_f1": training["metrics"]["macro_f1"],
            "validation_macro_f1": validation["metrics"]["macro_f1"],
            "train_accuracy": training["metrics"]["accuracy"],
            "validation_accuracy": validation["metrics"]["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds[-1],
        })
        if stopper.update(score, epoch, model.state_dict()):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint,
                config={**cfg, "class_names": class_names, "image_size": int(cfg["image_size"])},
                class_names=class_names,
                class_to_index=class_to_index,
                task=cfg["task"],
                strategy="multimodal",
                architecture=cfg["backbone"],
                metadata_preprocessor=processor.config(),
                metadata_fields=list(fields),
                threshold=cfg.get("threshold"),
                calibration=cfg.get("calibration"),
                checkpoint_identifier=checkpoint.stem,
            )
        if stopper.should_stop:
            break

    if stopper.best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")
    model.load_state_dict(stopper.best_state)
    validation = _run_epoch(model, loaders["validation"], optimizer, scaler, device, cfg, weights, training=False)
    evaluation_started = perf_counter()
    development_test = _run_epoch(model, loaders["test"], optimizer, scaler, device, cfg, weights, training=False)
    evaluation_seconds = perf_counter() - evaluation_started
    stop_summary = stopper.summary(len(history), int(cfg["epochs"]))
    update_checkpoint_metadata(checkpoint, early_stopping=stop_summary)

    sample = loaders["test"].dataset[0]
    sample_metadata = processor.transform(pd.DataFrame([sample["metadata"]]))
    image = sample["image"].unsqueeze(0).to(device)
    encoded = {key: value.to(device) for key, value in sample_metadata.items()}
    inference = benchmark_callable(
        lambda: model(image, encoded),
        device=device,
        warmup=int(cfg.get("inference_warmup", 3)),
        repeats=int(cfg.get("inference_repeats", 20)),
    )
    timing = {
        "training_seconds": perf_counter() - training_started - evaluation_seconds,
        "epoch_seconds": epoch_seconds,
        "validation_seconds": validation_seconds,
        "evaluation_seconds": evaluation_seconds,
        "inference_model_only_batch_size_1": inference,
    }
    threshold = cfg.get("threshold")
    threshold_value = threshold.get("threshold") if isinstance(threshold, dict) else threshold
    predictions = prediction_frame(
        development_test["targets"],
        development_test["probabilities"],
        development_test["metadata"],
        class_order=class_names,
        task=cfg["task"],
        strategy="multimodal",
        checkpoint_id=checkpoint.stem,
        threshold=threshold_value,
        calibrated=False,
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    export_predictions(predictions, run_directory / "development_test_predictions.csv")
    plot_training_history(pd.DataFrame(history), run_directory / "training_history.png")
    plot_confusion_matrix(development_test["metrics"]["confusion_matrix"], class_names, run_directory / "confusion_matrix.png")
    if len(class_names) == 2:
        plot_binary_curves(
            development_test["targets"],
            development_test["probabilities"][:, 1],
            run_directory / "roc_pr_calibration.png",
        )
    (run_directory / "development_test_metrics.json").write_text(
        json.dumps(development_test["metrics"], indent=2), encoding="utf-8"
    )
    split_summary = checked.groupby(["split", target_column], dropna=False).size().rename("count").reset_index().to_dict("records")
    timing["total_experiment_seconds"] = perf_counter() - experiment_started
    payload = {
        "run_name": run_name,
        "config": {
            **cfg,
            "class_names": class_names,
            "metadata_preprocessor": processor.config(),
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
        "seed": cfg.get("seed", 42),
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


def load_multimodal_checkpoint(
    path: str | Path,
    *,
    backbone_factory: Callable[..., nn.Module] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[MultimodalClassifier, dict]:
    """Reload a multimodal model and its fitted metadata processing contract."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    processor_config = payload.get("metadata_preprocessor")
    if not processor_config:
        raise ValueError("Checkpoint has no fitted metadata preprocessing contract")
    processor = MetadataPreprocessor.from_config(processor_config)
    config = {**payload["config"], "pretrained": False}
    model = build_multimodal_model(config, processor, len(payload["class_names"]), backbone_factory)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location).eval()
    model.metadata_preprocessor = processor
    return model, payload


def ablation_config(base_config: dict, fields: Iterable[str] | None = None, image_only: bool = False) -> dict:
    """Create a controlled image-only or selected-metadata ablation configuration."""
    config = dict(base_config)
    config["metadata_fields"] = [] if image_only else list(fields or SAFE_METADATA_FIELDS)
    config["ablation"] = "image_only" if image_only else "image_plus_metadata"
    return config


__all__ = [
    "MetadataPreprocessor", "MultimodalClassifier", "SAFE_METADATA_FIELDS",
    "ablation_config", "build_multimodal_model", "image_only_baseline",
    "load_multimodal_checkpoint", "make_metadata_tensors", "train",
]
