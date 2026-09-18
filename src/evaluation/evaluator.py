"""Universal checkpoint evaluation, prediction export, and final-test protection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.datasets import ManifestImageDataset, TASK_TARGETS, select_task_manifest, validate_manifest
from src.data.transforms import build_transforms
from .metrics import bootstrap_confidence_intervals, classification_metrics, lesion_presence_metrics


PREDICTION_COLUMNS = [
    "sample_id", "image_id", "patient_id", "lesion_id", "source_dataset", "split", "task",
    "true_label", "predicted_label", "class_probabilities", "threshold", "calibrated",
    "strategy", "checkpoint_id", "ensemble_id",
]


@dataclass
class CheckpointBundle:
    """Keep a loaded model together with its persisted inference contract."""
    model: torch.nn.Module
    strategy: str
    task: str
    class_order: list[str]
    config: dict
    checkpoint_path: str
    payload: dict


def build_evaluation_loader(
    manifest: pd.DataFrame,
    bundle: CheckpointBundle,
    *,
    split: str | None,
    batch_size: int | None = None,
    box_provider=None,
    allow_oracle_crops: bool = False,
    allow_final_test: bool = False,
    frozen_config_path: str | Path = "results/final_model/frozen_config.yaml",
) -> DataLoader:
    """Build a deterministic loader for any supported checkpoint input contract."""
    is_ddi = manifest.dataset.fillna("").str.upper().eq("DDI").any()
    checked = validate_external_manifest(manifest, allow_final_test=allow_final_test, frozen_config_path=frozen_config_path) if is_ddi else validate_manifest(manifest)
    transform = build_transforms(int(bundle.config.get("image_size", 224)), training=False)
    input_mode = bundle.config.get("input_mode", "full_image")
    if input_mode == "full_image":
        dataset = ManifestImageDataset(checked, split, transform, bundle.task, allow_final_test=is_ddi)
    else:
        from src.strategies.cnn_common import CropAssistedManifestDataset
        oracle = bool(bundle.config.get("ground_truth_crop_ablation", False))
        if oracle and not allow_oracle_crops:
            raise PermissionError("Ground-truth crop evaluation is an oracle ablation; pass allow_oracle_crops=True explicitly")
        dataset = CropAssistedManifestDataset(
            checked, split, transform, bundle.task, crop_transform=transform,
            box_provider=box_provider, ground_truth_box=oracle,
            crop_margin=float(bundle.config.get("crop_margin", .15)), output_mode=input_mode,
            allow_final_test=is_ddi,
        )

    def collate(batch):
        from src.strategies.cnn_common import collate_manifest_batch
        output = collate_manifest_batch(batch)
        if bundle.strategy == "multimodal":
            processor = getattr(bundle.model, "metadata_preprocessor", None)
            if processor is None:
                raise ValueError("Loaded multimodal model has no persisted metadata preprocessor")
            output["metadata_tensors"] = processor.transform(pd.DataFrame(output["metadata"]))
        return output

    loader = DataLoader(dataset, batch_size=batch_size or int(bundle.config.get("batch_size", 16)), shuffle=False, num_workers=0, collate_fn=collate)
    loader.class_names = bundle.class_order
    return loader


def _loader_contains_ddi(loader) -> bool:
    dataset = getattr(loader, "dataset", None)
    while dataset is not None:
        frame = getattr(dataset, "frame", None)
        if frame is not None and "dataset" in frame and frame["dataset"].fillna("").str.upper().eq("DDI").any():
            return True
        dataset = getattr(dataset, "base", None)
    return False


def _loader_split(loader) -> str | None:
    dataset = getattr(loader, "dataset", None)
    while dataset is not None:
        frame = getattr(dataset, "frame", None)
        if frame is not None and "split" in frame:
            values = frame["split"].dropna().unique().tolist()
            return str(values[0]) if len(values) == 1 else None
        dataset = getattr(dataset, "base", None)
    return None


def _guard_final_test(loader, dataset_name: str | None, allow_final_test: bool, frozen_config_path: str | Path) -> bool:
    is_ddi = (dataset_name or "").upper() == "DDI" or _loader_contains_ddi(loader)
    if not is_ddi:
        return False
    if not allow_final_test:
        raise PermissionError("DDI requires allow_final_test=True")
    if not Path(frozen_config_path).is_file():
        raise FileNotFoundError("DDI evaluation requires an existing frozen final configuration")
    return True


def _class_order(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("class_names"):
        return list(map(str, payload["class_names"]))
    mapping = payload.get("class_to_index")
    if mapping:
        return [str(label) for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    config = payload.get("config", {})
    if config.get("class_names"):
        return list(map(str, config["class_names"]))
    raise ValueError("Checkpoint does not persist class_names or class_to_index")


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    *,
    strategy: str | None = None,
    config_path: str | Path | None = None,
    device=None,
    dinov2_backbone_factory=None,
) -> CheckpointBundle:
    """Load any supported strategy checkpoint using its persisted configuration."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    persisted = dict(payload.get("config", {}))
    supplied = {}
    if config_path is not None:
        with Path(config_path).open(encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle) or {}
        for key in ("strategy_name", "task", "backbone", "input_mode", "image_size"):
            if key in supplied and key in persisted and supplied[key] != persisted[key]:
                raise ValueError(f"Associated config disagrees with checkpoint for {key!r}")
    config = {**supplied, **persisted}
    checkpoint_strategy = payload.get("strategy") or config.get("strategy_name")
    if strategy and checkpoint_strategy and strategy != checkpoint_strategy:
        raise ValueError("Requested strategy disagrees with the checkpoint")
    chosen = strategy or checkpoint_strategy
    if chosen not in {"efficientnet", "convnext", "lesion_presence", "dinov2", "multimodal"}:
        raise ValueError(f"Unsupported or missing strategy: {chosen!r}")
    task = payload.get("task") or config.get("task")
    if task not in TASK_TARGETS:
        raise ValueError(f"Checkpoint has unsupported or missing task: {task!r}")
    class_order = _class_order(payload)

    if chosen in {"efficientnet", "convnext", "lesion_presence"}:
        from src.strategies.cnn_common import build_diagnostic_input_model
        architecture = payload.get("architecture") or config.get("backbone")
        model = build_diagnostic_input_model(config.get("input_mode", "full_image"), architecture, len(class_order), float(config.get("dropout", .2)), pretrained=False)
    elif chosen == "dinov2":
        from src.strategies.dinov2_strategy import DinoV2Classifier, load_dinov2_backbone
        factory = dinov2_backbone_factory or load_dinov2_backbone
        backbone = factory(config["backbone"], pretrained=False)
        model = DinoV2Classifier(backbone, len(class_order), float(config.get("dropout", .2)), config.get("head_width"))
    else:
        from src.strategies.multimodal_strategy import MetadataPreprocessor, build_multimodal_model
        processor_config = payload.get("metadata_preprocessor")
        if not processor_config:
            raise ValueError("Multimodal checkpoint must persist metadata_preprocessor fitted on training data")
        processor = MetadataPreprocessor(tuple(processor_config["fields"]), processor_config["age_mean"], processor_config["age_std"], processor_config["vocabularies"])
        model = build_multimodal_model({**config, "pretrained": False}, processor, len(class_order), backbone_factory=dinov2_backbone_factory) if dinov2_backbone_factory else build_multimodal_model({**config, "pretrained": False}, processor, len(class_order))
        model.metadata_preprocessor = processor
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return CheckpointBundle(model, chosen, task, class_order, config, str(path), payload)


def _forward_model(model, batch: Mapping[str, Any], device):
    if "metadata_tensors" in batch:
        metadata = {key: value.to(device) for key, value in batch["metadata_tensors"].items()}
        return model(batch["image"].to(device), metadata)
    if "image" in batch:
        return model(batch["image"].to(device))
    if "full_image" in batch and "lesion_crop" in batch:
        return model(batch["full_image"].to(device), batch["lesion_crop"].to(device))
    raise KeyError("Batch must contain image, full_image+lesion_crop, or image+metadata_tensors")


def collect_predictions(model, loader, *, device=None, class_order: Sequence[str] | None = None) -> dict:
    """Collect logits, probabilities, encoded targets, and non-private row metadata."""
    device = device or next(model.parameters(), torch.empty(0)).device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    class_order = list(class_order or getattr(loader, "class_names", []))
    if not class_order:
        raise ValueError("class_order is required for universal evaluation")
    lookup = {str(label): index for index, label in enumerate(class_order)}
    targets, logits, metadata = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            raw_targets = batch["target"]
            if torch.is_tensor(raw_targets):
                raw_targets = raw_targets.cpu().tolist()
            valid_indices = [index for index, value in enumerate(raw_targets) if value is not None]
            if not valid_indices:
                continue
            output = _forward_model(model, batch, device)
            logits.append(output[valid_indices].detach().float().cpu().numpy())
            targets.extend(lookup[str(raw_targets[index])] for index in valid_indices)
            rows = batch.get("metadata", [{} for _ in raw_targets])
            metadata.extend(rows[index] for index in valid_indices)
    if not targets:
        raise ValueError("Evaluation loader produced no labelled samples")
    logit_array = np.concatenate(logits)
    shifted = logit_array - logit_array.max(axis=1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(axis=1, keepdims=True)
    return {"targets": np.asarray(targets), "logits": logit_array, "probabilities": probability, "metadata": metadata, "class_order": class_order}


def evaluate_model(
    model,
    loader,
    device=None,
    allow_final_test: bool = False,
    dataset_name: str | None = None,
    task: str = "diagnosis_binary",
    *,
    class_order: Sequence[str] | None = None,
    threshold: float | None = None,
    calibration=None,
    evaluation_split: str | None = None,
    frozen_config_path: str | Path = "results/final_model/frozen_config.yaml",
) -> dict:
    """Evaluate a compatible model; DDI needs consent and a pre-existing freeze."""
    is_ddi = _guard_final_test(loader, dataset_name, allow_final_test, frozen_config_path)
    collected = collect_predictions(model, loader, device=device, class_order=class_order)
    probabilities = collected["probabilities"] if calibration is None else calibration.apply(collected["logits"])
    targets = collected["targets"]
    predictions = (probabilities[:, 1] >= threshold).astype(int) if threshold is not None and probabilities.shape[1] == 2 else probabilities.argmax(axis=1)
    metric_function = lesion_presence_metrics if task == "lesion_presence" else classification_metrics
    cross_entropy = -float(np.mean(np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))))
    result = metric_function(targets, predictions, probabilities, labels=list(range(len(collected["class_order"]))), class_names=collected["class_order"], loss=cross_entropy)
    split = evaluation_split or _loader_split(loader)
    scope = "external_final_test" if is_ddi else {"validation": "internal_validation", "test": "internal_development_test"}.get(split, f"development_{split or 'unspecified'}")
    result.update({"loss_name": "cross_entropy", "task": task, "threshold": threshold, "calibrated": calibration is not None, "evaluation_split": split, "evaluation_scope": scope})
    return result


def prediction_frame(
    targets,
    probabilities,
    metadata: Sequence[Mapping[str, Any]],
    *,
    class_order: Sequence[str],
    task: str,
    strategy: str,
    checkpoint_id: str | None = None,
    ensemble_id: str | None = None,
    threshold: float | None = None,
    calibrated: bool = False,
) -> pd.DataFrame:
    """Build the privacy-limited standard prediction export schema."""
    target, probability = np.asarray(targets), np.asarray(probabilities, dtype=float)
    prediction = (probability[:, 1] >= threshold).astype(int) if threshold is not None and probability.shape[1] == 2 else probability.argmax(axis=1)
    rows = []
    for index, values in enumerate(metadata):
        image_id = values.get("image_id")
        rows.append({
            "sample_id": image_id or f"sample-{index}", "image_id": image_id,
            "patient_id": values.get("patient_id"), "lesion_id": values.get("lesion_id"),
            "source_dataset": values.get("dataset"), "split": values.get("split"), "task": task,
            "true_label": class_order[int(target[index])], "predicted_label": class_order[int(prediction[index])],
            "class_probabilities": json.dumps({name: float(probability[index, column]) for column, name in enumerate(class_order)}, sort_keys=True),
            "threshold": threshold, "calibrated": calibrated, "strategy": strategy,
            "checkpoint_id": checkpoint_id, "ensemble_id": ensemble_id,
        })
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def export_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and save standard row-aligned prediction exports."""
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction export is missing columns: {sorted(missing)}")
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(frame[PREDICTION_COLUMNS].to_json(orient="records", indent=2), encoding="utf-8")
    elif destination.suffix.lower() == ".csv":
        frame[PREDICTION_COLUMNS].to_csv(destination, index=False)
    else:
        raise ValueError("Prediction export path must end in .csv or .json")
    return destination


def save_evaluation_report(report: Mapping[str, Any], output_directory: str | Path, *, name: str = "evaluation") -> dict[str, Path]:
    """Persist a full JSON report and a flat one-row metric CSV."""
    destination = Path(output_directory); destination.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = destination / f"{name}.json", destination / f"{name}.csv"
    json_path.write_text(json.dumps(dict(report), indent=2, default=str), encoding="utf-8")
    scalar = {key: value for key, value in report.items() if value is None or isinstance(value, (str, int, float, bool))}
    pd.DataFrame([scalar]).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def stratified_metrics(
    frame: pd.DataFrame,
    targets,
    predictions,
    probabilities,
    *,
    class_order: Sequence[str],
    columns: Sequence[str] = ("dataset", "skin_tone", "sex", "age_group", "anatomical_site"),
    minimum_support: int = 20,
) -> dict:
    """Report every observed dataset/subgroup and flag low-support estimates."""
    target, predicted, probability = np.asarray(targets), np.asarray(predictions), np.asarray(probabilities)
    if len(frame) != len(target):
        raise ValueError("Frame and prediction arrays must be row-aligned")
    frame = frame.copy()
    if "age_group" not in frame and "age" in frame:
        frame["age_group"] = pd.cut(pd.to_numeric(frame.age, errors="coerce"), bins=[-np.inf, 17, 39, 59, np.inf], labels=["0-17", "18-39", "40-59", "60+"])
    output = {}
    for column in columns:
        if column not in frame:
            continue
        reports = []
        for value, indices in frame.groupby(column, dropna=False).indices.items():
            index = np.asarray(indices)
            metrics = classification_metrics(target[index], predicted[index], probability[index], labels=list(range(len(class_order))), class_names=class_order)
            reports.append({"group": None if pd.isna(value) else value, "support": len(index), "low_support": len(index) < minimum_support, "metrics": metrics})
        output[column] = reports
    return output


def lesion_presence_readiness(manifest: pd.DataFrame) -> dict:
    """State whether a real lesion-versus-normal evaluation is possible."""
    checked = validate_manifest(manifest)
    selected = select_task_manifest(checked, "lesion_presence")
    counts = selected.lesion_present.value_counts().to_dict()
    ready = 0 in counts and 1 in counts and bool(selected.loc[selected.lesion_present.eq(0), "normal_skin"].fillna(False).all())
    return {
        "ready": ready,
        "support": {"normal_skin": int(counts.get(0, 0)), "lesion_present": int(counts.get(1, 0))},
        "reason": None if ready else "Real lesion-presence evaluation requires both lesion photographs and true normal-skin negatives; none are fabricated.",
    }


def major_result_report(targets, predictions, probabilities, *, class_order, groups=None, n_resamples: int = 1000, seed: int = 42) -> dict:
    """Return metrics plus grouped or sample-level bootstrap intervals."""
    target, probability = np.asarray(targets), np.asarray(probabilities)
    cross_entropy = -float(np.mean(np.log(np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0))))
    metrics = classification_metrics(target, predictions, probability, labels=list(range(len(class_order))), class_names=class_order, loss=cross_entropy)
    metrics["loss_name"] = "cross_entropy"
    metrics["confidence_intervals"] = bootstrap_confidence_intervals(targets, predictions, probabilities, labels=list(range(len(class_order))), groups=groups, n_resamples=n_resamples, seed=seed)
    metrics["bootstrap_method"] = "percentile bootstrap resampling groups with replacement" if groups is not None else "percentile bootstrap resampling samples with replacement"
    return metrics


def evaluate_end_to_end(
    malignant_targets,
    *,
    usable_input,
    lesion_detected,
    diagnosis_positive_probability,
    diagnosis_threshold: float,
) -> dict:
    """Evaluate the full gate sequence; upstream malignant misses remain misses."""
    targets = np.asarray(malignant_targets, dtype=int)
    usable, detected = np.asarray(usable_input, dtype=bool), np.asarray(lesion_detected, dtype=bool)
    diagnosis = np.asarray(diagnosis_positive_probability, dtype=float)
    if not (targets.shape == usable.shape == detected.shape == diagnosis.shape):
        raise ValueError("All end-to-end arrays must have equal shape")
    reached_diagnosis = usable & detected & np.isfinite(diagnosis)
    predictions = np.zeros(len(targets), dtype=int)
    predictions[reached_diagnosis] = diagnosis[reached_diagnosis] >= diagnosis_threshold
    end_to_end_scores = np.zeros(len(targets), dtype=float)
    end_to_end_scores[reached_diagnosis] = diagnosis[reached_diagnosis]
    probabilities = np.column_stack((1 - end_to_end_scores, end_to_end_scores))
    metrics = classification_metrics(targets, predictions, probabilities, labels=[0, 1], class_names=["benign", "malignant"])
    metrics["pipeline_failures"] = {
        "unusable_input": int((~usable).sum()),
        "lesion_not_detected": int((usable & ~detected).sum()),
        "diagnosis_unavailable": int((usable & detected & ~np.isfinite(diagnosis)).sum()),
        "malignant_upstream_misses": int(((targets == 1) & ~reached_diagnosis).sum()),
        "reached_diagnosis": int(reached_diagnosis.sum()),
    }
    return metrics


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
        return {"commit": commit, "dirty": bool(status.strip()), "worktree_diff_sha256": sha256(diff).hexdigest() if diff else None}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "worktree_diff_sha256": None}


def freeze_final_configuration(configuration: Mapping[str, Any], output_path: str | Path = "results/final_model/frozen_config.yaml") -> Path:
    """Persist an immutable, development-selected system before external testing."""
    required = {"task", "class_order", "models", "preprocessing", "ensemble", "threshold", "calibration", "dataset_mappings"}
    missing = required - set(configuration)
    if missing:
        raise ValueError(f"Frozen configuration is missing: {sorted(missing)}")
    if not configuration["models"] or len(configuration["class_order"]) < 2:
        raise ValueError("Frozen configuration needs at least one model and two ordered classes")
    preprocessing = configuration["preprocessing"]
    if "image_resolution" not in preprocessing or "metadata_fields" not in preprocessing:
        raise ValueError("Frozen preprocessing must include image_resolution and metadata_fields")
    threshold = configuration["threshold"]
    if configuration["task"] in {"diagnosis_binary", "lesion_presence"} and (not threshold or threshold.get("fit_split") != "validation"):
        raise ValueError("Binary frozen systems require a validation-fitted threshold")
    calibration = configuration["calibration"]
    if calibration.get("enabled") and calibration.get("fit_split") != "validation":
        raise ValueError("Enabled calibration must have been fit on validation")
    ensemble = configuration["ensemble"]
    if ensemble.get("weights") and ensemble.get("method", "").startswith("validation") and ensemble.get("fit_split") != "validation":
        raise ValueError("Validation-derived ensemble weights must record fit_split=validation")
    development_sources = [str(value).upper() for value in configuration.get("development_datasets", [])]
    if "DDI" in development_sources:
        raise ValueError("DDI cannot appear in final-model development datasets")
    models = []
    for model in configuration["models"]:
        item = dict(model)
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        item["checkpoint_sha256"] = _file_sha256(checkpoint)
        if item.get("config"):
            config_path = Path(item["config"])
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            item["config_sha256"] = _file_sha256(config_path)
        models.append(item)
    payload = dict(configuration)
    payload["models"] = models
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["code_identifier"] = payload.get("code_identifier") or _git_state()
    payload["external_test_consumed"] = False
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError("Frozen configuration already exists; do not overwrite a pre-DDI artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def save_external_test_report(report: Mapping[str, Any], *, frozen_config_path: str | Path, output_directory: str | Path = "results/final_external_test") -> Path:
    """Save DDI separately and mark the one-way consumption in a copied record."""
    frozen_path = Path(frozen_config_path)
    if not frozen_path.is_file():
        raise FileNotFoundError(frozen_path)
    destination = Path(output_directory); destination.mkdir(parents=True, exist_ok=True)
    payload = {"evaluation_scope": "external_final_test", "dataset": "DDI", "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), **dict(report)}
    result_path = destination / "ddi_report.json"
    if result_path.exists():
        raise FileExistsError("DDI external-test report already exists; automatic reruns are disabled")
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (destination / "frozen_config_used.yaml").write_text(frozen_path.read_text(encoding="utf-8"), encoding="utf-8")
    (destination / "external_test_consumption.json").write_text(json.dumps({"dataset": "DDI", "consumed": True, "consumed_at_utc": payload["evaluated_at_utc"], "frozen_config": str(frozen_path)}, indent=2), encoding="utf-8")
    return result_path


def evaluate_frozen_external_test(
    manifest: pd.DataFrame,
    *,
    allow_final_test: bool = False,
    frozen_config_path: str | Path = "results/final_model/frozen_config.yaml",
    output_directory: str | Path = "results/final_external_test",
    dinov2_backbone_factory=None,
) -> dict:
    """Run the already-frozen system once on compatible DDI labels."""
    checked = validate_external_manifest(manifest, allow_final_test=allow_final_test, frozen_config_path=frozen_config_path)
    frozen_path = Path(frozen_config_path)
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    task, class_order = frozen["task"], list(map(str, frozen["class_order"]))
    if task != "diagnosis_binary":
        raise ValueError("DDI compatibility is limited to the explicitly mapped binary diagnosis task in this project")
    if not (frozen.get("dataset_mappings") or {}).get("DDI"):
        raise ValueError("The pre-DDI freeze must contain an explicit compatible DDI binary label mapping")
    from .calibration import apply_temperature
    from .ensemble import EnsembleMember, average_probabilities

    members, reference = [], None
    for definition in frozen["models"]:
        checkpoint = Path(definition["checkpoint"])
        if _file_sha256(checkpoint) != definition["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after freeze: {checkpoint}")
        if definition.get("config") and _file_sha256(Path(definition["config"])) != definition.get("config_sha256"):
            raise ValueError(f"Model configuration changed after freeze: {definition['config']}")
        bundle = load_checkpoint_bundle(checkpoint, strategy=definition.get("strategy"), config_path=definition.get("config"), dinov2_backbone_factory=dinov2_backbone_factory)
        if bundle.task != task or bundle.class_order != class_order:
            raise ValueError("Frozen member task/class order does not match the frozen system")
        loader = build_evaluation_loader(checked, bundle, split=None, allow_final_test=True, frozen_config_path=frozen_path)
        collected = collect_predictions(bundle.model, loader, class_order=class_order)
        if reference is None:
            reference = collected
        elif not np.array_equal(reference["targets"], collected["targets"]):
            raise ValueError("Frozen members did not evaluate identically ordered DDI samples")
        sample_ids = tuple(str(row.get("image_id", index)) for index, row in enumerate(collected["metadata"]))
        members.append(EnsembleMember(definition.get("name", checkpoint.stem), task, tuple(class_order), collected["probabilities"], sample_ids, bundle.strategy == "multimodal"))
    ensemble = frozen.get("ensemble") or {}
    if ensemble.get("member_names") and ensemble["member_names"] != [member.name for member in members]:
        raise ValueError("Frozen ensemble member names/order do not match frozen model definitions")
    weights = ensemble.get("weights") or [1.0] * len(members)
    probabilities, ensemble_details = average_probabilities(members, weights, missing_member_policy=ensemble.get("missing_member_policy", "renormalize_available"))
    calibration = frozen.get("calibration") or {}
    calibrated = bool(calibration.get("enabled", False))
    if calibrated:
        probabilities = apply_temperature(probabilities, float(calibration["temperature"]), input_type="probabilities")
    threshold = (frozen.get("threshold") or {}).get("threshold")
    predictions = (probabilities[:, 1] >= threshold).astype(int) if threshold is not None else probabilities.argmax(axis=1)
    metrics = major_result_report(reference["targets"], predictions, probabilities, class_order=class_order, groups=[row.get("patient_id") or row.get("image_id") for row in reference["metadata"]], n_resamples=int((frozen.get("bootstrap") or {}).get("resamples", 1000)), seed=int((frozen.get("bootstrap") or {}).get("seed", 42)))
    export = prediction_frame(reference["targets"], probabilities, reference["metadata"], class_order=class_order, task=task, strategy="frozen_ensemble", ensemble_id=ensemble.get("identifier", "final"), threshold=threshold, calibrated=calibrated)
    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Final external-test output directory is not empty; automatic reruns are disabled")
    export_predictions(export, destination / "ddi_predictions.csv")
    report = {"metrics": metrics, "ensemble": ensemble_details, "threshold": threshold, "calibrated": calibrated}
    report_path = save_external_test_report(report, frozen_config_path=frozen_path, output_directory=destination)
    return {"report": report, "report_path": report_path, "predictions_path": destination / "ddi_predictions.csv"}


def validate_external_manifest(manifest: pd.DataFrame, *, allow_final_test: bool, frozen_config_path: str | Path) -> pd.DataFrame:
    """Expose the intentional DDI access gate before a loader opens image files."""
    if not allow_final_test:
        raise PermissionError("DDI requires allow_final_test=True")
    if not Path(frozen_config_path).is_file():
        raise FileNotFoundError("DDI evaluation requires an existing frozen final configuration")
    checked = validate_manifest(manifest, allow_final_test=True)
    if not checked.dataset.fillna("").str.upper().eq("DDI").all():
        raise ValueError("External final-test manifests must contain only DDI rows")
    return checked
