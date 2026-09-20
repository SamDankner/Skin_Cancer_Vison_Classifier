"""Canonical task targets, independent from human-readable class labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable

import numpy as np
import torch


TASK_TARGETS = {
    "lesion_presence": "lesion_present",
    "diagnosis_binary": "binary_target",
    "diagnosis_multiclass": "harmonized_diagnosis",
    "image_quality": "image_quality_label",
}


BINARY_CLASS_NAMES = {
    "lesion_presence": ("no_lesion", "lesion_present"),
    "diagnosis_binary": ("benign", "malignant"),
}


def canonical_target(value, task: str) -> int | str:
    """Normalize one legitimate task target and reject ambiguous coercions."""
    if task in BINARY_CLASS_NAMES:
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, Real):
            number = float(value)
            if math.isfinite(number) and number in (0.0, 1.0):
                return int(number)
        raise ValueError(
            f"Invalid {task} target {value!r}; expected numeric/bool 0 or 1"
        )
    if task == "diagnosis_multiclass":
        if isinstance(value, str) and value.strip():
            return value
        raise ValueError(
            f"Invalid diagnosis_multiclass target {value!r}; expected a non-empty label"
        )
    raise ValueError(f"Unsupported target-encoding task: {task!r}")


@dataclass(frozen=True)
class TargetEncoding:
    """Map canonical task values to model indices while retaining display labels."""

    task: str
    class_values: tuple[int | str, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.class_values) != len(self.class_names):
            raise ValueError("class_values and class_names must have equal length")
        if len(set(self.class_values)) != len(self.class_values):
            raise ValueError("class_values must be unique")

    @property
    def class_to_index(self) -> dict[int | str, int]:
        return {value: index for index, value in enumerate(self.class_values)}

    @property
    def display_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    def encode(self, value) -> int:
        canonical = canonical_target(value, self.task)
        try:
            return self.class_to_index[canonical]
        except KeyError as exc:
            raise ValueError(
                f"Target {canonical!r} is not in the fitted classes {self.class_values!r}"
            ) from exc

    def encode_tensor(self, values: Iterable) -> torch.Tensor:
        return torch.tensor([self.encode(value) for value in values], dtype=torch.long)


def build_target_encoding(values: Iterable, task: str) -> TargetEncoding:
    """Fit deterministic canonical classes from training targets for one task."""
    canonical_values = [canonical_target(value, task) for value in values]
    observed = set(canonical_values)
    if task in BINARY_CLASS_NAMES:
        if observed != {0, 1}:
            raise ValueError(
                f"{task} training targets must contain canonical classes 0 and 1; "
                f"observed {sorted(observed)}"
            )
        return TargetEncoding(task, (0, 1), BINARY_CLASS_NAMES[task])
    classes = tuple(sorted(observed))
    if len(classes) < 2:
        raise ValueError("Training targets must contain at least two classes")
    return TargetEncoding(task, classes, tuple(str(value) for value in classes))


def target_encoding_for_manifest(manifest, task: str) -> TargetEncoding:
    """Fit target encoding from the training split of a selected manifest."""
    if task not in TASK_TARGETS:
        raise ValueError(f"Unknown task {task!r}")
    target = TASK_TARGETS[task]
    values = manifest.loc[manifest.split.eq("train"), target].dropna().tolist()
    return build_target_encoding(values, task)


__all__ = [
    "BINARY_CLASS_NAMES",
    "TargetEncoding",
    "TASK_TARGETS",
    "build_target_encoding",
    "canonical_target",
    "target_encoding_for_manifest",
]
