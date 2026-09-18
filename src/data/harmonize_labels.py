"""Explicit label harmonization for macro-photo skin datasets."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Mapping
import pandas as pd

@dataclass(frozen=True)
class LabelDecision:
    harmonized_diagnosis: str | None
    binary_target: int | None
    reason: str

MALIGNANT = {"mel", "melanoma", "malignant melanoma", "bcc", "basal cell carcinoma", "scc", "squamous cell carcinoma", "squamous cell carcinoma in situ", "bowen disease", "merkel cell carcinoma", "dermatofibrosarcoma protuberans"}
BENIGN = {"nv", "nevus", "naevus", "melanocytic nevus", "seborrheic keratosis", "sk", "dermatofibroma", "df", "vascular lesion", "angioma", "lentigo", "benign keratosis", "solar lentigo", "blue nevus"}

def normalize_label(value: Any) -> str | None:
    """Normalize a source label without changing its medical meaning."""
    if value is None or pd.isna(value): return None
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(value).strip().lower())) or None

def harmonize_label(original_label: Any, custom_mapping: Mapping[str, Any] | None = None) -> LabelDecision:
    """Return an explicit binary decision, or ``None`` where mapping is unclear."""
    label = normalize_label(original_label)
    if label is None: return LabelDecision(None, None, "missing source label")
    if custom_mapping and label in custom_mapping:
        mapped = custom_mapping[label]
        if isinstance(mapped, Mapping): return LabelDecision(mapped.get("harmonized_diagnosis", label), mapped.get("binary_target"), "custom mapping")
        return LabelDecision(label, mapped, "custom mapping")
    if label in MALIGNANT: return LabelDecision(label, 1, "unambiguous malignant diagnosis")
    if label in BENIGN: return LabelDecision(label, 0, "unambiguous benign diagnosis")
    return LabelDecision(label, None, "not mapped: requires medical label-policy review")

def harmonize_manifest(manifest: pd.DataFrame, label_column: str = "original_label", custom_mapping: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Append diagnoses and nullable binary targets to a manifest."""
    if label_column not in manifest: raise KeyError(f"Manifest has no {label_column!r} column")
    result = manifest.copy(); decisions = result[label_column].map(lambda value: harmonize_label(value, custom_mapping))
    result["harmonized_diagnosis"] = [x.harmonized_diagnosis for x in decisions]
    result["binary_target"] = pd.array([x.binary_target for x in decisions], dtype="Int64")
    result["binary_mapping_reason"] = [x.reason for x in decisions]
    return result
