"""Leakage-safe, group-aware manifest splitting."""
from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def group_key(manifest: pd.DataFrame) -> pd.Series:
    """Prefer patient, then lesion, then image identity to keep correlated images together."""
    patient = manifest.patient_id.astype("string").fillna("").str.strip()
    lesion = manifest.lesion_id.astype("string").fillna("").str.strip()

    # The most stable identity available wins, keeping related photographs in
    # one split even when a source omits patient or lesion identifiers.
    return (
        ("image:" + manifest.image_id.astype(str))
        .where(lesion.eq(""), "lesion:" + lesion)
        .where(patient.eq(""), "patient:" + patient)
    )

def make_group_splits(
    manifest: pd.DataFrame,
    train_fraction: float = .70,
    validation_fraction: float = .15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign train/validation/test splits by group; DDI is never development data."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid split fractions")
    if manifest.dataset.fillna("").str.upper().eq("DDI").any():
        raise PermissionError("Do not split DDI; it is untouched final external test data")

    result = manifest.copy()
    groups = group_key(result)
    group_count = groups.nunique()
    held_group_count = math.ceil((1 - train_fraction) * group_count)
    if held_group_count < 2:
        raise ValueError("Need enough independent patient/lesion/image groups to place at least one group in both validation and test splits")
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=1 - train_fraction,
        random_state=random_state,
    )
    _, held_idx = next(first.split(result, groups=groups))
    held = result.iloc[held_idx]
    held_groups = groups.iloc[held_idx]
    test_share = (1 - train_fraction - validation_fraction) / (1 - train_fraction)

    # Split only the held-out groups, so no group can bridge train and either
    # validation or development test.
    second = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=random_state + 1)
    val_local, test_local = next(second.split(held, groups=held_groups))
    result["split"] = "train"
    result.iloc[held_idx[val_local], result.columns.get_loc("split")] = "validation"
    result.iloc[held_idx[test_local], result.columns.get_loc("split")] = "test"
    assert not result.assign(_group=groups).groupby("_group").split.nunique().gt(1).any()
    return result


def class_coverage_report(manifest: pd.DataFrame, target_column: str, required_splits: tuple[str, ...] = ("train", "validation")) -> pd.DataFrame:
    """Report classes absent from development splits without changing assignments."""
    if target_column not in manifest: raise KeyError(f"Manifest has no {target_column!r} column")
    if "split" not in manifest: raise KeyError("Manifest has no 'split' column")
    classes = set(manifest[target_column].dropna().tolist())
    rows = []
    for split in required_splits:
        present = set(manifest.loc[manifest.split.eq(split), target_column].dropna().tolist())
        rows.append({"split": split, "present_classes": sorted(present, key=str), "missing_classes": sorted(classes - present, key=str), "complete": classes == present})
    return pd.DataFrame(rows)


def validate_split_class_coverage(
    manifest: pd.DataFrame,
    target_column: str,
    required_splits: tuple[str, ...] = ("train", "validation"),
) -> None:
    """Stop development training when a required split lacks an observed class."""
    report = class_coverage_report(manifest, target_column, required_splits)
    incomplete = report.loc[~report.complete]
    if not incomplete.empty:
        details = "; ".join(f"{row.split}: {row.missing_classes}" for row in incomplete.itertuples(index=False))
        raise ValueError(f"Required split class coverage is incomplete for {target_column}: {details}")

def leakage_report(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return patient/lesion/image groups that cross assigned splits."""
    groups = group_key(manifest)
    reports = [
        manifest.assign(_group=groups)
        .groupby("_group").split.nunique()
        .rename("split_count").loc[lambda value: value > 1].reset_index()
    ]
    if "file_sha256" in manifest:
        hashes = manifest.file_sha256.astype("string").fillna("").str.strip()
        duplicate_rows = manifest.loc[hashes.ne("")].assign(_group="sha256:" + hashes.loc[hashes.ne("")])
        reports.append(
            duplicate_rows.groupby("_group").split.nunique()
            .rename("split_count").loc[lambda value: value > 1].reset_index()
        )
    return pd.concat(reports, ignore_index=True).drop_duplicates() if reports else pd.DataFrame(columns=["_group", "split_count"])
