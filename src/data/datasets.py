"""Unified manifest and PyTorch dataset support for clinical macro photographs."""
from __future__ import annotations
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.targets import TASK_TARGETS, canonical_target

MANIFEST_COLUMNS = [
    "dataset", "source_dataset", "image_path", "image_id", "patient_id", "case_id",
    "lesion_id", "original_label", "harmonized_diagnosis", "binary_target",
    "lesion_present", "normal_skin", "normal_label_strength", "normal_label_method",
    "gate_mapping_reason", "gate_label_strength", "gate_label_source",
    "gate_negative_subtype", "other_skin_condition", "image_quality_label", "localization_available",
    "bounding_box", "segmentation_mask_path", "supported_for_lesion_detection",
    "supported_for_lesion_presence", "supported_for_diagnosis", "age", "age_group",
    "sex", "anatomical_site", "skin_tone", "monk_skin_tone", "image_modality",
    "label_source", "ground_truth_method", "self_reported_related_category",
    "dermatologist_skin_condition_label", "weighted_skin_condition_label",
    "dermatologist_fitzpatrick_skin_type", "symptoms", "lesion_diameter_mm",
    "perceptual_hash", "duplicate_group_id", "split",
]
DATASET_SETUP = {"PAD-UFES-20": "https://data.mendeley.com/datasets/zr7vgbcyr2/1", "MILK10k": "https://doi.org/10.1038/s41597-024-03501-y", "Fitzpatrick17k": "https://github.com/mattgroh/fitzpatrick17k", "SCIN": "https://github.com/google-research-datasets/scin", "DDI": "https://stanfordaimi.github.io/digital-dermatology/"}


def _nullable_boolean(values: pd.Series, name: str) -> pd.Series:
    text_mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False, "yes": True, "no": False,
        "y": True, "n": False,
    }

    def parse(value):
        if pd.isna(value) or str(value).strip() == "":
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, Real):
            return {0.0: False, 1.0: True}.get(float(value), "<INVALID>")
        return text_mapping.get(str(value).strip().lower(), "<INVALID>")

    converted = values.map(parse)
    if converted.eq("<INVALID>").any():
        invalid = sorted(values.loc[converted.eq("<INVALID>")].astype(str).unique().tolist())
        raise ValueError(f"Manifest column {name!r} has invalid boolean values: {invalid}")
    return converted.astype("boolean")


def empty_manifest() -> pd.DataFrame:
    """Return an empty manifest with the complete public schema."""
    return pd.DataFrame(columns=MANIFEST_COLUMNS)

def validate_manifest(
    manifest: pd.DataFrame,
    allow_final_test: bool = False,
) -> pd.DataFrame:
    """Validate schema and reject non-clinical or unapproved final-test samples."""
    # Older local manifests remain readable, then acquire explicit null
    # provenance fields on their next write.  This is a schema migration, not
    # an inference from absent metadata.
    result = manifest.copy()
    for column in MANIFEST_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    if result["source_dataset"].isna().all():
        result["source_dataset"] = result["dataset"]
    if result["supported_for_lesion_presence"].isna().all():
        result["supported_for_lesion_presence"] = result["supported_for_lesion_detection"]
    for column in (
        "lesion_present", "normal_skin", "localization_available",
        "supported_for_lesion_detection", "supported_for_lesion_presence",
        "supported_for_diagnosis", "other_skin_condition",
    ):
        result[column] = _nullable_boolean(result[column], column)

    # Modality and DDI checks happen before task selection so an unsafe row
    # cannot become eligible merely because its target happens to be present.
    modality = result.image_modality.fillna("clinical").str.lower()
    if modality.str.contains("dermoscop|micro|patholog", regex=True).any():
        raise ValueError("Only ordinary clinical/macro photographs are allowed")
    if result.dataset.fillna("").str.upper().eq("DDI").any() and not allow_final_test:
        raise PermissionError("DDI is final external test data; pass allow_final_test=True explicitly")
    normal = result.normal_skin.fillna(False)
    if normal.any() and result.loc[normal, "harmonized_diagnosis"].notna().any(): raise ValueError("Normal skin cannot be assigned a lesion diagnosis")
    if normal.any() and result.loc[normal, "binary_target"].notna().any(): raise ValueError("Normal skin cannot be assigned a benign/malignant target")
    if normal.any() and result.loc[normal, "lesion_present"].fillna(1).astype(int).ne(0).any():
        raise ValueError("Normal skin must have lesion_present=0")
    if result.loc[normal, "supported_for_diagnosis"].fillna(False).any():
        raise ValueError("Normal skin cannot be supported for lesion diagnosis")
    other = result.other_skin_condition.fillna(False)
    if result.loc[other, "normal_skin"].fillna(False).any():
        raise ValueError("Other skin conditions cannot be called normal skin")
    if result.loc[other, "supported_for_diagnosis"].fillna(False).any():
        raise ValueError("Other non-target skin conditions cannot enter tumor diagnosis")
    subtype = result.gate_negative_subtype.fillna("").astype(str).str.strip().str.lower()
    invalid_subtype = ~subtype.isin({"", "healthy_no_visible_lesion", "other_skin_condition"})
    if invalid_subtype.any():
        raise ValueError("gate_negative_subtype must be healthy_no_visible_lesion or other_skin_condition")
    other_with_gate_target = other & result.lesion_present.notna()
    if other_with_gate_target.any() and not (
        result.loc[other_with_gate_target, "lesion_present"].eq(False)
        & subtype.loc[other_with_gate_target].eq("other_skin_condition")
    ).all():
        raise ValueError("Other skin conditions may only be explicit gate-negative hard negatives")
    if result.loc[other, "supported_for_lesion_presence"].fillna(False).any() and not (
        subtype.loc[other & result.supported_for_lesion_presence.fillna(False)]
        .eq("other_skin_condition")
        .all()
    ):
        raise ValueError("Eligible other skin conditions require gate_negative_subtype=other_skin_condition")
    explicit_healthy = subtype.eq("healthy_no_visible_lesion")
    if explicit_healthy.any() and not (
        result.loc[explicit_healthy, "normal_skin"].fillna(False)
        & result.loc[explicit_healthy, "lesion_present"].eq(False)
    ).all():
        raise ValueError("Healthy negative subtype requires normal_skin=true and lesion_present=0")
    return result

def write_manifest(
    manifest: pd.DataFrame,
    path: str | Path,
    allow_final_test: bool = False,
) -> Path:
    """Validate and save a manifest CSV, returning its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_manifest(manifest, allow_final_test).to_csv(path, index=False)
    return path

def file_sha256(path: str | Path, chunk_size: int = 1_048_576) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def add_exact_hashes(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add exact hashes and flags; never delete questionable cases."""
    result = manifest.copy()
    result["file_sha256"] = [file_sha256(path) for path in result.image_path]
    result["exact_duplicate_flag"] = result.file_sha256.duplicated(keep=False)
    return result


def image_dhash(path: str | Path, hash_size: int = 8) -> int:
    """Return a deterministic 64-bit difference hash for duplicate auditing."""
    with Image.open(path) as source:
        resized = source.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        getter = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(getter())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def _image_duplicate_fingerprint(path: str | Path) -> tuple[int, tuple[int, int, int]]:
    with Image.open(path) as source:
        # Decode once: preparation previously opened every image twice (once
        # here and again through ``image_dhash``), making the required full
        # duplicate audit unnecessarily slow on network-synced storage.
        grayscale = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        grayscale_getter = getattr(grayscale, "get_flattened_data", grayscale.getdata)
        grayscale_pixels = list(grayscale_getter())
        dhash = 0
        for row in range(8):
            offset = row * 9
            for column in range(8):
                dhash = (dhash << 1) | int(
                    grayscale_pixels[offset + column] > grayscale_pixels[offset + column + 1]
                )
        rgb = source.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
        getter = getattr(rgb, "get_flattened_data", rgb.getdata)
        values = list(getter())
    means = tuple(round(sum(pixel[channel] for pixel in values) / len(values)) for channel in range(3))
    return dhash, means


def add_perceptual_duplicate_groups(
    manifest: pd.DataFrame,
    max_distance: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash images and group near duplicates with a Hamming-distance BK-tree."""
    result = manifest.copy()
    fingerprints = [_image_duplicate_fingerprint(path) for path in result.image_path]
    hashes = [fingerprint[0] for fingerprint in fingerprints]
    parent = list(range(len(result)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # Each node is [hash_value, positions_with_that_hash, {distance: child}].
    # Mean RGB prevents uniform but visibly different synthetic/real images from
    # collapsing merely because their edge hashes are identical.
    tree = None
    pairs = []
    for position, value in enumerate(hashes):
        if tree is None:
            tree = [value, [position], {}]
            continue
        stack = [tree]
        matches = []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= max_distance:
                for candidate in node[1]:
                    color_distance = max(
                        abs(left - right)
                        for left, right in zip(
                            fingerprints[position][1], fingerprints[candidate][1]
                        )
                    )
                    if color_distance <= 12:
                        matches.append((candidate, distance))
            lower, upper = distance - max_distance, distance + max_distance
            stack.extend(
                child for edge, child in node[2].items() if lower <= edge <= upper
            )
        for other, distance in matches:
            union(position, other)
            pairs.append({
                "left_index": result.index[other],
                "right_index": result.index[position],
                "distance": distance,
                "possible_duplicate": True,
            })
        node = tree
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(position)
                break
            if distance not in node[2]:
                node[2][distance] = [value, [position], {}]
                break
            node = node[2][distance]

    members: dict[int, list[int]] = {}
    for position in range(len(result)):
        members.setdefault(find(position), []).append(position)
    group_ids = [None] * len(result)
    for component in members.values():
        if len(component) < 2:
            continue
        label = min(str(result.iloc[position].image_id) for position in component)
        for position in component:
            group_ids[position] = f"perceptual:{label}"
    result["perceptual_hash"] = [
        f"{value:016x}:{red:02x}{green:02x}{blue:02x}"
        for value, (red, green, blue) in fingerprints
    ]
    result["duplicate_group_id"] = group_ids
    pair_frame = pd.DataFrame(
        pairs,
        columns=["left_index", "right_index", "distance", "possible_duplicate"],
    )
    return result, pair_frame

def find_perceptual_duplicates(manifest: pd.DataFrame, hasher: Callable[[str], object], distance: Callable[[object, object], float], max_distance: float = 4) -> pd.DataFrame:
    """Flag pairs using a caller-provided perceptual-hash implementation."""
    values = [(index, hasher(path)) for index, path in manifest.image_path.items()]
    rows = []
    for pos, (left_idx, left) in enumerate(values):
        for right_idx, right in values[pos + 1:]:
            value = distance(left, right)
            if value <= max_distance:
                rows.append(
                    {
                        "left_index": left_idx,
                        "right_index": right_idx,
                        "distance": value,
                        "possible_duplicate": True,
                    }
                )
    return pd.DataFrame(rows, columns=["left_index", "right_index", "distance", "possible_duplicate"])

def select_task_manifest(
    manifest: pd.DataFrame,
    task: str,
    *,
    include_hard_negatives: bool = False,
) -> pd.DataFrame:
    """Filter one manifest for a separable learning task without relabeling normal skin."""
    if task not in TASK_TARGETS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASK_TARGETS)}")

    result = manifest.copy()
    target = TASK_TARGETS[task]
    if task == "lesion_presence":
        supported = _nullable_boolean(
            result.supported_for_lesion_presence,
            "supported_for_lesion_presence",
        ).fillna(False)
        other = _nullable_boolean(
            result.other_skin_condition,
            "other_skin_condition",
        ).fillna(False)
        result = result.loc[supported & (~other | include_hard_negatives)]
    elif task.startswith("diagnosis"):
        # Diagnosis tasks exclude both normal skin and ambiguous lesion status;
        # absence of metadata is not treated as a negative label.
        supported = _nullable_boolean(
            result.supported_for_diagnosis,
            "supported_for_diagnosis",
        ).fillna(False)
        lesion = _nullable_boolean(result.lesion_present, "lesion_present").fillna(False)
        result = result.loc[supported & lesion]
    return result.loc[result[target].notna()].copy()


def build_lesion_presence_manifest(
    manifest: pd.DataFrame,
    normal_label_strengths: tuple[str, ...] = ("strong", "moderate", "weak"),
    *,
    include_hard_negatives: bool = False,
) -> pd.DataFrame:
    """Select explicit lesion-presence labels and requested normal reliability tiers.

    Auxiliary non-lesional images are intentionally excluded: they support
    appearance-diversity analysis, not final normal-skin ground truth.
    """
    checked = validate_manifest(manifest)
    result = select_task_manifest(
        checked, "lesion_presence", include_hard_negatives=include_hard_negatives
    )
    normal = result.normal_skin.fillna(False)
    allowed = {str(value).lower() for value in normal_label_strengths}
    return result.loc[~normal | result.normal_label_strength.fillna("").str.lower().isin(allowed)].copy()

class ManifestImageDataset(Dataset):
    """Load RGB macro photographs for lesion presence, diagnosis, or quality tasks."""
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str | None = None,
        transform: Callable | None = None,
        task: str = "diagnosis_binary",
        target_column: str | None = None,
        allow_final_test: bool = False,
        include_hard_negatives: bool = False,
    ):
        frame = validate_manifest(manifest, allow_final_test)
        frame = select_task_manifest(
            frame, task, include_hard_negatives=include_hard_negatives
        )
        self.frame = frame if split is None else frame.loc[frame.split.eq(split)].reset_index(drop=True)
        self.transform = transform
        self.task = task
        self.target_column = target_column or TASK_TARGETS[task]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        value = row[self.target_column]
        target = None if pd.isna(value) else canonical_target(value, self.task)
        return {"image": image, "target": target, "metadata": row.to_dict()}
