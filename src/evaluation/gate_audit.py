"""Reproducible, development-only prediction audit for the focal-lesion gate."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.data.datasets import validate_manifest
from src.inference import predict_image


def _audit_category(frame: pd.DataFrame) -> pd.Series:
    category = pd.Series("unsupported", index=frame.index, dtype="string")
    quality = frame.image_quality_label.fillna("").astype(str).str.lower()
    category.loc[quality.str.contains("poor")] = "poor_quality_unsupported"
    category.loc[frame.other_skin_condition.fillna(False)] = "other_skin_condition"
    eligible = frame.supported_for_lesion_presence.fillna(False)
    category.loc[eligible & frame.lesion_present.eq(0)] = "no_visible_target_lesion"
    category.loc[
        eligible & frame.lesion_present.eq(1) & frame.binary_target.eq(0)
    ] = "benign_focal_lesion"
    category.loc[
        eligible & frame.lesion_present.eq(1) & frame.binary_target.eq(1)
    ] = "malignant_focal_lesion"
    category.loc[
        eligible & frame.lesion_present.eq(1) & frame.binary_target.isna()
    ] = "focal_lesion_diagnosis_unmapped"
    return category


def create_gate_prediction_audit(
    manifest: pd.DataFrame,
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
) -> pd.DataFrame:
    """Pick one deterministic sample per available audit category and predict it."""
    checked = validate_manifest(manifest)
    if checked.dataset.fillna("").str.upper().eq("DDI").any():
        raise PermissionError("DDI cannot be used for a development gate audit")
    candidates = checked.copy()
    candidates["audit_category"] = _audit_category(candidates)
    test_rows = candidates.loc[candidates.split.eq("test")].copy()
    # Unsupported rows can lack a split when their modality is invalid/no-skin;
    # they remain development data and are selected only if no split row exists.
    rows = []
    for category in (
        "no_visible_target_lesion", "benign_focal_lesion",
        "malignant_focal_lesion", "other_skin_condition",
        "poor_quality_unsupported", "focal_lesion_diagnosis_unmapped",
        "unsupported",
    ):
        pool = test_rows.loc[test_rows.audit_category.eq(category)]
        if pool.empty and category in {"poor_quality_unsupported", "unsupported"}:
            pool = candidates.loc[candidates.audit_category.eq(category)]
        if pool.empty:
            continue
        chosen_index = min(
            pool.index,
            key=lambda index: sha256(
                f"{seed}:{pool.loc[index, 'image_id']}".encode()
            ).hexdigest(),
        )
        source = pool.loc[chosen_index]
        prediction = predict_image(
            source.image_path, checkpoint=checkpoint, device=device,
            warmup=0, repeats=1,
        )
        predicted = prediction["lesion_presence_result"]
        source_category = source.original_label
        if str(source.source_dataset).upper() == "SCIN" and pd.notna(
            source.self_reported_related_category
        ):
            source_category = source.self_reported_related_category
        rows.append({
            "audit_category": category,
            "source_dataset": source.source_dataset,
            "image_id": source.image_id,
            "source_ground_truth_category": source_category,
            "gate_target_eligible": bool(source.supported_for_lesion_presence),
            "expected_binary_target": (
                None if pd.isna(source.lesion_present) else int(source.lesion_present)
            ),
            "gate_label_strength": source.gate_label_strength,
            "gate_negative_subtype": source.gate_negative_subtype,
            "predicted_probability_target_lesion": prediction.get("lesion_probability"),
            "predicted_gate_class": predicted,
            "downstream_diagnosis_triggered": predicted == "target_lesion_present",
        })
    result = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


__all__ = ["create_gate_prediction_audit"]
