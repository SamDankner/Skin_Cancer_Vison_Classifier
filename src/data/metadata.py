"""Auditable normalization and audit helpers for optional clinical metadata.

Raw manifest values are deliberately never overwritten.  These functions are
used at model-preprocessing time, where each raw value can be paired with its
canonical value and source for later inspection.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


MISSING_CATEGORY = "<MISSING>"

# These tables are intentionally compact and source-agnostic.  A value not in
# a table remains explicit as ``unmapped:<normalized raw value>`` rather than
# being guessed into a clinical region.
SEX_ALIASES = {
    "f": "female", "female": "female", "woman": "female", "girl": "female",
    "m": "male", "male": "male", "man": "male", "boy": "male",
}
ANATOMICAL_SITE_ALIASES = {
    "arm": "upper_extremity", "forearm": "upper_extremity", "upper arm": "upper_extremity",
    "upper extremity": "upper_extremity", "upper limb": "upper_extremity", "shoulder": "upper_extremity",
    "leg": "lower_extremity", "thigh": "lower_extremity", "calf": "lower_extremity",
    "lower extremity": "lower_extremity", "lower limb": "lower_extremity", "foot": "hand_foot",
    "feet": "hand_foot", "hand": "hand_foot", "hands": "hand_foot", "palm": "hand_foot", "sole": "hand_foot",
    "face": "head_neck", "scalp": "head_neck", "head": "head_neck", "neck": "head_neck",
    "head/neck": "head_neck", "head and neck": "head_neck", "ear": "head_neck", "nose": "head_neck", "lip": "head_neck",
    "chest": "trunk", "abdomen": "trunk", "back": "trunk", "trunk": "trunk", "torso": "trunk",
    "buttock": "trunk", "groin": "trunk",
}
SKIN_TONE_ALIASES = {
    "i": "fitzpatrick_1", "1": "fitzpatrick_1", "type i": "fitzpatrick_1", "fitzpatrick i": "fitzpatrick_1",
    "ii": "fitzpatrick_2", "2": "fitzpatrick_2", "type ii": "fitzpatrick_2", "fitzpatrick ii": "fitzpatrick_2",
    "iii": "fitzpatrick_3", "3": "fitzpatrick_3", "type iii": "fitzpatrick_3", "fitzpatrick iii": "fitzpatrick_3",
    "iv": "fitzpatrick_4", "4": "fitzpatrick_4", "type iv": "fitzpatrick_4", "fitzpatrick iv": "fitzpatrick_4",
    "v": "fitzpatrick_5", "5": "fitzpatrick_5", "type v": "fitzpatrick_5", "fitzpatrick v": "fitzpatrick_5",
    "vi": "fitzpatrick_6", "6": "fitzpatrick_6", "type vi": "fitzpatrick_6", "fitzpatrick vi": "fitzpatrick_6",
    "light": "light", "fair": "light", "medium": "medium", "tan": "medium", "dark": "dark", "deep": "dark",
}


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = " ".join(str(value).strip().casefold().replace("_", " ").replace("-", " ").split())
    return normalized or None


def normalize_metadata_value(field: str, value: Any) -> str | None:
    """Return a conservative canonical value, retaining unknown raw concepts.

    ``None`` denotes missing data; it is converted to the dedicated missing
    model ID later.  Nonempty values that lack an audited mapping are explicit
    ``unmapped:`` categories, which preserves provenance and prevents silent
    medical assumptions.
    """
    # CSV readers commonly represent Fitzpatrick values as 1.0 through 6.0.
    # Treat integral numeric representations as their explicit equivalent,
    # without applying that conversion to free-text metadata fields.
    if field == "skin_tone" and not isinstance(value, bool):
        # CSV inference can yield a Python float while deliberately quoted
        # values remain strings ("1.0"). Both spellings denote the same
        # Fitzpatrick level; normalize only integral levels 1--6.
        try:
            numeric = float(str(value).strip())
            if pd.notna(numeric) and numeric.is_integer() and 1 <= numeric <= 6:
                value = str(int(numeric))
        except (TypeError, ValueError):
            pass
    text = _text(value)
    if text is None:
        return None
    aliases = {"sex": SEX_ALIASES, "anatomical_site": ANATOMICAL_SITE_ALIASES, "skin_tone": SKIN_TONE_ALIASES}
    if field not in aliases:
        return text
    return aliases[field].get(text, f"unmapped:{text}")


def normalize_metadata_frame(frame: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    """Expose raw, normalized, and source metadata without changing input rows."""
    result = pd.DataFrame(index=frame.index)
    result["metadata_source"] = frame.get("source_dataset", frame.get("dataset", pd.Series(None, index=frame.index)))
    for field in fields:
        raw = frame[field] if field in frame else pd.Series(None, index=frame.index)
        result[f"raw_{field}"] = raw
        result[f"normalized_{field}"] = raw.map(lambda value: normalize_metadata_value(field, value)) if field != "age" else pd.to_numeric(raw, errors="coerce")
    return result


def _field_summary(raw: pd.Series, normalized: pd.Series, field: str) -> dict:
    """Separate availability, raw values, canonical values, and unmapped IDs."""
    present = raw.notna() & raw.astype(str).str.strip().ne("")
    values = normalized.loc[present].dropna().astype(str)
    unmapped = values.loc[values.str.startswith("unmapped:")]
    return {
        "available": int(present.sum()), "missing": int((~present).sum()),
        "available_percent": float(100 * present.mean()) if len(raw) else 0.0,
        "raw_unique_values": sorted(raw.loc[present].astype(str).unique().tolist()),
        "normalized_unique_values": sorted(values.unique().tolist()),
        "unmapped_count": int(unmapped.size),
        "unmapped_unique_values": sorted(unmapped.unique().tolist()),
        "unknown_count": 0,
        "value_type": "numeric" if field == "age" else "categorical",
    }


def metadata_audit_report(frame: pd.DataFrame, *, task: str = "diagnosis_binary", fields: Iterable[str] = ("age", "sex", "anatomical_site", "skin_tone")) -> dict:
    """Summarize optional metadata and source/label associations for development rows."""
    if frame.get("dataset", pd.Series(dtype="object")).fillna("").astype(str).str.upper().eq("DDI").any():
        raise PermissionError("DDI is final test data and cannot be audited for development preprocessing")
    fields = tuple(fields)
    eligible = frame.copy()
    if task == "diagnosis_binary" and "binary_target" in eligible:
        eligible = eligible.loc[eligible.binary_target.isin([0, 1])].copy()
    normalized = normalize_metadata_frame(eligible, fields)
    source = eligible.get("source_dataset", eligible.get("dataset", pd.Series("<missing>", index=eligible.index))).fillna("<missing>").astype(str)
    overall = {field: _field_summary(normalized[f"raw_{field}"], normalized[f"normalized_{field}"], field) for field in fields}
    by_source = []
    for name in sorted(source.unique()):
        index = source.eq(name)
        row = {"source_dataset": name, "eligible_examples": int(index.sum()), "fields": {}}
        for field in fields:
            raw = normalized.loc[index, f"raw_{field}"]
            row["fields"][field] = _field_summary(raw, normalized.loc[index, f"normalized_{field}"], field)
        by_source.append(row)
    mappings = {}
    for field in (field for field in fields if field != "age"):
        pairs = normalized[[f"raw_{field}", f"normalized_{field}"]].dropna().drop_duplicates()
        mappings[field] = [
            {"raw": str(row[f"raw_{field}"],), "normalized": str(row[f"normalized_{field}"])}
            for _, row in pairs.sort_values([f"normalized_{field}", f"raw_{field}"]).iterrows()
        ]
    label_association = []
    if "binary_target" in eligible:
        for name, group in eligible.assign(_source=source).groupby("_source", dropna=False):
            target = pd.to_numeric(group.binary_target, errors="coerce")
            label_association.append({"source_dataset": str(name), "examples": int(len(group)), "malignant_percent": float(100 * target.mean()) if target.notna().any() else None})
    return {"task": task, "eligible_examples": int(len(eligible)), "field_summary_overall": overall, "raw_to_normalized_mappings": mappings, "availability_by_source": by_source, "source_label_association": label_association, "state_definitions": {"missing": "raw value absent or blank", "unknown": "unseen nonmissing value at transform time; assigned vocabulary ID 1", "unmapped": "nonempty raw value with no audited canonical mapping; retained as unmapped:<value>"}, "note": "Associations are audit flags, not grounds to remove metadata fields."}
