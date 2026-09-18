"""Leakage-safe, group-aware manifest splitting."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def group_key(manifest: pd.DataFrame) -> pd.Series:
    """Prefer patient, then lesion, then image identity to keep correlated images together."""
    patient = manifest.patient_id.astype("string").fillna("").str.strip(); lesion = manifest.lesion_id.astype("string").fillna("").str.strip()
    return ("image:" + manifest.image_id.astype(str)).where(lesion.eq(""), "lesion:" + lesion).where(patient.eq(""), "patient:" + patient)

def make_group_splits(manifest: pd.DataFrame, train_fraction: float = .70, validation_fraction: float = .15, random_state: int = 42) -> pd.DataFrame:
    """Assign train/validation/test splits by group; DDI is never development data."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction: raise ValueError("Invalid split fractions")
    if manifest.dataset.fillna("").str.upper().eq("DDI").any(): raise PermissionError("Do not split DDI; it is untouched final external test data")
    result = manifest.copy(); groups = group_key(result)
    if groups.nunique() < 3: raise ValueError("Need at least three independent patient/lesion/image groups")
    first = GroupShuffleSplit(n_splits=1, test_size=1-train_fraction, random_state=random_state)
    train_idx, held_idx = next(first.split(result, groups=groups)); held = result.iloc[held_idx]
    held_groups = groups.iloc[held_idx]; test_share = (1-train_fraction-validation_fraction)/(1-train_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=random_state + 1)
    val_local, test_local = next(second.split(held, groups=held_groups))
    result["split"] = "train"; result.iloc[held_idx[val_local], result.columns.get_loc("split")] = "validation"; result.iloc[held_idx[test_local], result.columns.get_loc("split")] = "test"
    assert not result.assign(_group=groups).groupby("_group").split.nunique().gt(1).any()
    return result

def leakage_report(manifest: pd.DataFrame) -> pd.DataFrame:
    groups = group_key(manifest); return manifest.assign(_group=groups).groupby("_group").split.nunique().rename("split_count").loc[lambda x: x > 1].reset_index()
