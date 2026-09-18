"""Unified, photo-first inference for persisted single models and ensembles."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
import yaml

from src.data.transforms import build_transforms
from src.data.datasets import file_sha256
from src.evaluation.calibration import apply_temperature
from src.evaluation.ensemble import EnsembleMember, average_probabilities
from src.evaluation.evaluator import CheckpointBundle, load_checkpoint_bundle
from src.utils.device import get_device
from src.utils.timing import benchmark_callable


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    """Open a supported image and return a detached RGB copy with clear errors."""
    if isinstance(image, Image.Image):
        source, should_close = image, False
    else:
        path = Path(image)
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type {path.suffix!r}; use {sorted(SUPPORTED_IMAGE_SUFFIXES)}")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            source, should_close = Image.open(path), True
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Image is unreadable or corrupt: {path}") from exc
    try:
        source.load()
        bands = source.getbands()
        if len(bands) not in {3, 4}:
            raise ValueError(f"Expected a three-channel photo (optional alpha); got mode {source.mode!r}")
        return source.convert("RGB").copy()
    finally:
        if should_close:
            source.close()


def _metadata_frame(metadata: Mapping[str, Any] | None) -> pd.DataFrame:
    """Create one metadata row while retaining true missing values."""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    return pd.DataFrame([dict(metadata or {})])


def _bundle_input(
    bundle: CheckpointBundle,
    photo: Image.Image,
    metadata,
    device,
) -> tuple[tuple, bool, list[str]]:
    """Prepare the persisted input contract for one full-image checkpoint."""
    input_mode = bundle.config.get("input_mode", "full_image")
    if input_mode != "full_image":
        raise ValueError(
            f"Checkpoint requires input_mode={input_mode!r}. Single-photo inference needs a serialized "
            "automatic localization provider, which this repository does not currently contain."
        )
    transform = build_transforms(int(bundle.config.get("image_size", 224)), training=False)
    image_tensor = transform(photo).unsqueeze(0).to(device)
    warnings: list[str] = []
    if bundle.strategy != "multimodal":
        if metadata:
            warnings.append("Metadata was supplied but this image-only model does not use it.")
        return (image_tensor,), False, warnings
    processor = getattr(bundle.model, "metadata_preprocessor", None)
    if processor is None:
        raise ValueError("Multimodal checkpoint has no fitted metadata preprocessor")
    # The fitted processor carries training-only statistics and explicit
    # missing-value indicators, so inference must use it unchanged.
    encoded = {
        key: value.to(device)
        for key, value in processor.transform(_metadata_frame(metadata)).items()
    }
    supplied = dict(metadata or {})
    provided_fields = {
        field for field in processor.fields
        if field in supplied and supplied[field] is not None and str(supplied[field]).strip()
    }
    ignored_fields = sorted(set(supplied) - set(processor.fields))
    if ignored_fields:
        warnings.append(f"Ignored metadata fields not used by this checkpoint: {ignored_fields}")
    if not provided_fields:
        warnings.append(
            "No metadata was supplied; the checkpoint's explicit missing-value indicators were used. "
            "No age, sex, site, or skin tone was invented."
        )
    return (image_tensor, encoded), bool(provided_fields), warnings


def _interpret(
    task: str,
    class_order: list[str],
    probabilities: np.ndarray,
    threshold: float | None,
) -> dict:
    """Create task-specific output without conflating clinical tasks."""
    predicted_index = int(probabilities.argmax())
    if threshold is not None and len(class_order) == 2:
        predicted_index = int(probabilities[1] >= threshold)
    result: dict[str, Any] = {
        "task": task,
        "predicted_class": class_order[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(class_order)
        },
        "threshold": threshold,
    }
    if task == "lesion_presence":
        result["lesion_presence_result"] = (
            "lesion_present" if predicted_index == 1 else "normal_skin"
        ) if class_order == ["0", "1"] else class_order[predicted_index]
        if len(class_order) == 2:
            result["lesion_probability"] = float(probabilities[1])
    elif task == "diagnosis_binary":
        result["predicted_diagnostic_class"] = (
            "malignant" if predicted_index == 1 else "benign"
        ) if class_order == ["0", "1"] else class_order[predicted_index]
        if len(class_order) == 2:
            result["malignant_probability"] = float(probabilities[1])
    elif task == "diagnosis_multiclass":
        result["predicted_diagnostic_class"] = class_order[predicted_index]
    return result


def _predict_bundle(
    bundle: CheckpointBundle,
    photo: Image.Image,
    metadata,
    device,
    warmup: int,
    repeats: int,
) -> dict:
    preprocessing_started = perf_counter()
    args, metadata_used, warnings = _bundle_input(bundle, photo, metadata, device)
    preprocessing_ms = (perf_counter() - preprocessing_started) * 1000.0
    # Evaluation mode and inference_mode prevent training-time behavior and
    # gradient allocation from affecting a production prediction.
    bundle.model.to(device).eval()
    timing = benchmark_callable(
        lambda: bundle.model(*args), device=device, warmup=warmup, repeats=repeats
    )
    with torch.inference_mode():
        logits = bundle.model(*args)
        # Convert unbounded class logits into a normalized class distribution.
        probability = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
    return {
        "probabilities": probability,
        "metadata_used": metadata_used,
        "warnings": warnings,
        "timing": {
            "preprocessing_ms": preprocessing_ms,
            "model_only_batch_size_1": timing,
        },
    }


def predict_image(
    image: str | Path | Image.Image,
    *,
    checkpoint: str | Path | None = None,
    frozen_config: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    warmup: int = 3,
    repeats: int = 20,
) -> dict:
    """Predict from one photograph, using optional metadata only when compatible."""
    if (checkpoint is None) == (frozen_config is None):
        raise ValueError("Provide exactly one of checkpoint or frozen_config")
    resolved_device = torch.device(device) if device is not None else get_device()
    request_started = perf_counter()

    decode_started = perf_counter()
    photo = _load_rgb_image(image)
    image_decode_ms = (perf_counter() - decode_started) * 1000.0
    try:
        if checkpoint is not None:
            bundle = load_checkpoint_bundle(checkpoint, device=resolved_device)
            output = _predict_bundle(bundle, photo, metadata, resolved_device, warmup, repeats)
            threshold = bundle.payload.get("threshold")
            threshold = threshold.get("threshold") if isinstance(threshold, Mapping) else threshold
            result = _interpret(bundle.task, bundle.class_order, output["probabilities"], threshold)
            result.update({
                "model": bundle.strategy,
                "checkpoint": bundle.checkpoint_path,
                "metadata_used": output["metadata_used"],
                "device": str(resolved_device),
                "timing": {
                    **output["timing"],
                    "image_decode_ms": image_decode_ms,
                    "end_to_end_model_ready_mean_ms": image_decode_ms
                    + output["timing"]["preprocessing_ms"]
                    + output["timing"]["model_only_batch_size_1"]["mean_ms"],
                },
                "warnings": output["warnings"],
            })
        else:
            frozen_path = Path(frozen_config)
            if not frozen_path.is_file():
                raise FileNotFoundError(frozen_path)
            frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8")) or {}
            class_order = list(map(str, frozen.get("class_order", [])))
            if len(class_order) < 2 or not frozen.get("models"):
                raise ValueError("Frozen configuration needs class_order and at least one model")
            members, warnings, metadata_used, member_timings = [], [], False, {}

            for definition in frozen["models"]:
                # A frozen ensemble is valid only for its exact persisted artifacts.
                if (
                    definition.get("checkpoint_sha256")
                    and file_sha256(definition["checkpoint"])
                    != definition["checkpoint_sha256"]
                ):
                    raise ValueError(f"Frozen checkpoint hash mismatch: {definition['checkpoint']}")
                if (
                    definition.get("config")
                    and definition.get("config_sha256")
                    and file_sha256(definition["config"])
                    != definition["config_sha256"]
                ):
                    raise ValueError(f"Frozen model config hash mismatch: {definition['config']}")
                bundle = load_checkpoint_bundle(
                    definition["checkpoint"],
                    strategy=definition.get("strategy"),
                    config_path=definition.get("config"),
                    device=resolved_device,
                )
                if bundle.task != frozen["task"] or bundle.class_order != class_order:
                    raise ValueError("Frozen ensemble member task/class order mismatch")
                try:
                    output = _predict_bundle(bundle, photo, metadata, resolved_device, warmup, repeats)
                except ValueError as exc:
                    if (frozen.get("ensemble") or {}).get("missing_member_policy") != "renormalize_available":
                        raise
                    warnings.append(
                        f"Omitted ensemble member {definition.get('name', bundle.strategy)}: {exc}"
                    )
                    continue
                metadata_used = metadata_used or output["metadata_used"]
                warnings.extend(output["warnings"])
                member_timings[definition.get("name", Path(bundle.checkpoint_path).stem)] = output["timing"]
                members.append(
                    EnsembleMember(
                        definition.get("name", Path(bundle.checkpoint_path).stem),
                        bundle.task,
                        tuple(class_order),
                        output["probabilities"][None, :],
                    )
                )
            if not members:
                raise ValueError("No frozen ensemble member can process this photograph")
            ensemble = frozen.get("ensemble") or {}
            configured_names = ensemble.get("member_names") or [member.name for member in members]
            configured_weights = ensemble.get("weights") or [1.0] * len(configured_names)
            weights_by_name = dict(zip(configured_names, configured_weights))
            probabilities, details = average_probabilities(
                members,
                [weights_by_name[member.name] for member in members],
                missing_member_policy="renormalize_available",
            )
            probability = probabilities[0]
            calibration = frozen.get("calibration") or {}
            if calibration.get("enabled"):
                # Apply the validation-fitted calibration only after members
                # have been combined in the frozen ensemble's class order.
                probability = apply_temperature(
                    probability[None, :],
                    float(calibration["temperature"]),
                    input_type="probabilities",
                )[0]
            threshold = (frozen.get("threshold") or {}).get("threshold")
            result = _interpret(frozen["task"], class_order, probability, threshold)
            result.update({
                "model": "frozen_ensemble",
                "ensemble": details,
                "frozen_config": str(frozen_path),
                "metadata_used": metadata_used,
                "device": str(resolved_device),
                "timing": {"image_decode_ms": image_decode_ms, "members": member_timings},
                "warnings": warnings,
            })
        result.setdefault("timing", {})["request_total_including_benchmark_ms"] = (
            perf_counter() - request_started
        ) * 1000.0
        result.setdefault("warnings", []).append(
            "This output is a model prediction, not a diagnosis. No trained "
            "image-quality or OOD detector is bundled."
        )
        return result
    finally:
        photo.close()


def metadata_from_json(value: str | Path | None) -> dict | None:
    """Read optional metadata from inline JSON or a JSON/YAML file."""
    if value is None:
        return None
    text_value = str(value)
    if text_value.lstrip().startswith("{"):
        data = json.loads(text_value)
    else:
        candidate = Path(value)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        text = candidate.read_text(encoding="utf-8")
        data = (
            yaml.safe_load(text)
            if candidate.suffix.lower() in {".yaml", ".yml"}
            else json.loads(text)
        )
    if not isinstance(data, dict):
        raise ValueError("Metadata input must contain an object/mapping")
    return data


def predict_with_lesion_routing(
    image,
    *,
    lesion_presence_checkpoint,
    diagnosis_checkpoint,
    metadata=None,
    **kwargs,
) -> dict:
    """Route a photo through lesion presence before optional diagnosis inference."""
    presence = predict_image(
        image,
        checkpoint=lesion_presence_checkpoint,
        metadata=metadata,
        **kwargs,
    )
    if presence.get("task") != "lesion_presence":
        raise ValueError("lesion_presence_checkpoint must contain a lesion_presence model")
    if presence.get("lesion_presence_result") in {"normal_skin", "0"}:
        # Do not run a diagnosis model after the lesion-presence gate rejects
        # the image; the result remains a model output, not healthy-skin proof.
        return {
            "routing": "no_lesion_detected",
            "lesion_presence": presence,
            "message": (
                "No lesion was detected by the model; this is not a clinical "
                "confirmation of healthy skin."
            ),
        }
    return {
        "routing": "lesion_detected",
        "lesion_presence": presence,
        "diagnosis": predict_image(
            image,
            checkpoint=diagnosis_checkpoint,
            metadata=metadata,
            **kwargs,
        ),
    }
