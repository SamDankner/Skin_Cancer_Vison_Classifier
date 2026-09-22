"""Unified, photo-first inference for persisted single models and ensembles."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
from src.ensemble.adaptive_ensemble import AdaptiveEnsemble


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
APP_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
FINAL_ENSEMBLE_MEMBERS = ("convnext", "efficientnet", "multimodal")
DEPLOYMENT_VARIANTS = ("v1_frozen", "v2_adaptive")


@dataclass(frozen=True)
class PredictionResult:
    """A UI-neutral prediction from the immutable final ensemble."""

    predicted_class: str
    malignant_probability: float
    threshold: float
    individual_models: dict[str, float]
    metadata_used: dict[str, bool]
    active_models: tuple[str, ...] = ()
    inactive_models: tuple[str, ...] = ()
    model_weights: dict[str, float] | None = None
    strategy_name: str = "equal_probability_average"


def validate_uploaded_image(
    content: bytes,
    filename: str,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Image.Image:
    """Decode a Streamlit upload in memory and return a detached RGB image."""
    if not content:
        raise ValueError("The uploaded image is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"The uploaded image exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    suffix = Path(filename).suffix.lower()
    if suffix not in APP_SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Unsupported image format. Upload a PNG, JPG, or JPEG image.")
    from io import BytesIO

    try:
        with Image.open(BytesIO(content)) as uploaded:
            uploaded.load()
            return _load_rgb_image(uploaded)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded image could not be decoded. Please choose a valid PNG, JPG, or JPEG file.") from exc


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
        if len(bands) not in {1, 2, 3, 4}:
            raise ValueError(f"Unsupported image color mode {source.mode!r}")
        return source.convert("RGB").copy()
    finally:
        if should_close:
            source.close()


def _metadata_frame(metadata: Mapping[str, Any] | None) -> pd.DataFrame:
    """Create one metadata row while retaining true missing values."""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    return pd.DataFrame([dict(metadata or {})])


class SkinCancerPredictor:
    """Reusable service for the protected v1 and separately configured v2 ensembles."""

    def __init__(self, frozen_config: Mapping[str, Any], bundles: Mapping[str, CheckpointBundle], device: torch.device):
        self.frozen_config = dict(frozen_config)
        self.bundles = dict(bundles)
        self.device = device
        self._validate_contract()

    @classmethod
    def from_frozen_config(
        cls,
        frozen_config: str | Path | None = None,
        *,
        variant: str | None = None,
        model_root: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "SkinCancerPredictor":
        """Load exactly the three checkpoints named by the immutable YAML file.

        ``FROZEN_CONFIG_PATH`` and ``MODEL_ROOT`` make a container-mounted model
        directory possible while preserving repository-local defaults.
        """
        repository_root = Path(__file__).resolve().parents[1]
        selected_variant = variant or os.environ.get("MODEL_VARIANT", "v2_adaptive")
        if frozen_config is None and "FROZEN_CONFIG_PATH" not in os.environ and selected_variant not in DEPLOYMENT_VARIANTS:
            raise ValueError("MODEL_VARIANT must be 'v1_frozen' or 'v2_adaptive'.")
        default_config = repository_root / "configs" / "deployments" / selected_variant / "ensemble.yaml"
        # Retain the historical path as the explicit v1 compatibility default.
        config_path = Path(frozen_config or os.environ.get("FROZEN_CONFIG_PATH") or default_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Frozen model configuration was not found: {config_path}")
        frozen = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        resolved_device = torch.device(device) if device is not None else get_device()
        configured_root = Path(model_root or os.environ.get("MODEL_ROOT", "")) if (model_root or os.environ.get("MODEL_ROOT")) else None
        bundles: dict[str, CheckpointBundle] = {}
        for definition in frozen.get("models", []):
            name = definition.get("name")
            # Frozen YAML was authored on Windows; resolve it portably in a
            # Linux/macOS container as well as in the local checkout.
            raw_checkpoint = Path(str(definition.get("checkpoint", "")).replace("\\", "/"))
            if raw_checkpoint.is_absolute():
                checkpoint = raw_checkpoint
            elif configured_root is not None:
                checkpoint = configured_root / Path(*raw_checkpoint.parts[1:]) if raw_checkpoint.parts and raw_checkpoint.parts[0] == "models" else configured_root / raw_checkpoint
            else:
                checkpoint = repository_root / raw_checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Frozen checkpoint for {name!r} was not found: {checkpoint}")
            expected_hash = definition.get("checkpoint_sha256")
            if expected_hash and file_sha256(checkpoint) != expected_hash:
                raise ValueError(f"Frozen checkpoint hash mismatch: {checkpoint}")
            bundles[str(name)] = load_checkpoint_bundle(checkpoint, strategy=definition.get("strategy"), device=resolved_device)
        return cls(frozen, bundles, resolved_device)

    def _validate_contract(self) -> None:
        config = self.frozen_config
        if config.get("status") not in {"frozen", "experimental"} or config.get("task") != "diagnosis_binary":
            raise ValueError("The app requires a diagnosis_binary deployment configuration.")
        if list(config.get("class_order", [])) != ["benign", "malignant"]:
            raise ValueError("Frozen class order must be benign, malignant.")
        ensemble = config.get("ensemble") or {}
        names = tuple(ensemble.get("member_names") or ())
        if names != FINAL_ENSEMBLE_MEMBERS or tuple(self.bundles) != FINAL_ENSEMBLE_MEMBERS:
            raise ValueError("The application only supports ConvNeXt, EfficientNet, and multimodal DINOv2 final members.")
        if config.get("status") == "frozen":
            weights = tuple(float(weight) for weight in ensemble.get("weights") or ())
            if len(weights) != 3 or any(weight != (1.0 / 3.0) for weight in weights):
                raise ValueError("Frozen v1 ensemble must retain its three equal weights.")
            if float((config.get("threshold") or {}).get("threshold", -1)) != 0.51:
                raise ValueError("Frozen v1 ensemble threshold must be 0.51.")
        elif not {"no_metadata_policy", "metadata_available_policy"}.issubset(ensemble):
            raise ValueError("v2 configuration needs policies for metadata-present and metadata-absent requests.")
        definitions = {item.get("name"): item for item in config.get("models", [])}
        if definitions.get("multimodal", {}).get("strategy") != "multimodal":
            raise ValueError("The final DINOv2 member must be the multimodal checkpoint.")
        for name, bundle in self.bundles.items():
            if bundle.strategy != definitions[name].get("strategy") or bundle.task != config["task"] or bundle.class_order != config["class_order"]:
                raise ValueError(f"Frozen inference contract mismatch for {name}.")

    def predict(
        self,
        image: Image.Image,
        *,
        age: float | int | None = None,
        sex: str | None = None,
        anatomical_site: str | None = None,
    ) -> PredictionResult:
        """Run image models and apply the configured deployment policy."""
        photo = _load_rgb_image(image)
        metadata = {"age": age, "sex": sex, "anatomical_site": anatomical_site}
        provided = {key: value is not None and str(value).strip() != "" for key, value in metadata.items()}
        try:
            probabilities: dict[str, float] = {}
            with torch.inference_mode():
                active = FINAL_ENSEMBLE_MEMBERS if any(provided.values()) or self.frozen_config.get("status") == "frozen" else FINAL_ENSEMBLE_MEMBERS[:2]
                for name in active:
                    bundle = self.bundles[name]
                    args, _, _ = _bundle_input(bundle, photo, metadata, self.device)
                    logits = bundle.model.to(self.device).eval()(*args)
                    output = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
                    probabilities[name] = float(output[bundle.class_order.index("malignant")])
            if self.frozen_config.get("status") == "frozen":
                weights = dict(zip(FINAL_ENSEMBLE_MEMBERS, (self.frozen_config.get("ensemble") or {})["weights"]))
                malignant_probability = float(sum(probabilities[name] * float(weights[name]) for name in FINAL_ENSEMBLE_MEMBERS))
                threshold = float((self.frozen_config.get("threshold") or {})["threshold"])
                decision = None
            else:
                decision = AdaptiveEnsemble(self.frozen_config).combine(probabilities, metadata=metadata)
                malignant_probability, threshold, weights = decision.malignant_probability, decision.threshold, decision.model_weights
            return PredictionResult(
                predicted_class=decision.predicted_class if decision else ("malignant" if malignant_probability >= threshold else "benign"),
                malignant_probability=malignant_probability,
                threshold=threshold,
                individual_models=probabilities,
                metadata_used=provided,
                active_models=decision.active_models if decision else FINAL_ENSEMBLE_MEMBERS,
                inactive_models=decision.inactive_models if decision else (),
                model_weights=weights,
                strategy_name=decision.strategy_name if decision else "equal_probability_average",
            )
        finally:
            photo.close()

    def attribute(
        self,
        image: Image.Image,
        model_name: str,
        *,
        age: float | int | None = None,
        sex: str | None = None,
        anatomical_site: str | None = None,
    ):
        """Explain one active deployment member using its persisted input contract.

        This deliberately has a separate gradient-enabled path from ``predict``.
        The caller is responsible for ensuring the requested member participated
        in the current deployment result.
        """
        from src.explainability import generate_attribution

        if model_name not in FINAL_ENSEMBLE_MEMBERS:
            raise ValueError(f"Unknown deployment model: {model_name!r}")
        photo = _load_rgb_image(image)
        metadata = {"age": age, "sex": sex, "anatomical_site": anatomical_site}
        try:
            bundle = self.bundles[model_name]
            args, _, _ = _bundle_input(bundle, photo, metadata, self.device)
            malignant_index = bundle.class_order.index("malignant")
            return generate_attribution(
                bundle.model.to(self.device), args, model_name=model_name, malignant_index=malignant_index
            )
        finally:
            photo.close()


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
            "target_lesion_present" if predicted_index == 1 else "no_target_lesion"
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
    if presence.get("lesion_presence_result") in {
        "no_target_lesion", "no_lesion", "normal_skin", "0"
    }:
        # Do not run a diagnosis model after the lesion-presence gate rejects
        # the image; the result remains a model output, not healthy-skin proof.
        return {
            "routing": "no_target_lesion",
            "lesion_presence": presence,
            "message": (
                "No target focal lesion was detected by the binary gate, so "
                "tumor diagnosis was not run. This is not confirmation that "
                "the skin is medically normal."
            ),
        }
    return {
        "routing": "target_lesion_present",
        "lesion_presence": presence,
        "diagnosis": predict_image(
            image,
            checkpoint=diagnosis_checkpoint,
            metadata=metadata,
            **kwargs,
        ),
    }
