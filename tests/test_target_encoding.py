"""Regression tests for task-aware model target encoding."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.targets import TargetEncoding, build_target_encoding, canonical_target


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (1, 1), (False, 0), (True, 1), (0.0, 0), (1.0, 1),
     (np.int64(0), 0), (np.float64(1.0), 1), (np.bool_(False), 0)],
)
def test_lesion_presence_accepts_equivalent_binary_values(value, expected):
    assert canonical_target(value, "lesion_presence") == expected


@pytest.mark.parametrize("value", [2, -1, 0.5, "1", "lesion", None])
def test_lesion_presence_rejects_invalid_targets(value):
    with pytest.raises(ValueError, match="expected numeric/bool 0 or 1"):
        canonical_target(value, "lesion_presence")


def test_binary_class_names_are_display_only():
    encoding = TargetEncoding(
        task="lesion_presence",
        class_values=(0, 1),
        class_names=("custom negative display", "custom positive display"),
    )
    targets = encoding.encode_tensor([False, 1.0, 0, True])
    assert targets.dtype == torch.long
    assert targets.tolist() == [0, 1, 0, 1]
    assert encoding.display_to_index == {
        "custom negative display": 0,
        "custom positive display": 1,
    }


def test_task_class_names_and_multiclass_values_are_canonical():
    lesion = build_target_encoding([False, True], "lesion_presence")
    binary = build_target_encoding([0.0, 1.0], "diagnosis_binary")
    multiclass = build_target_encoding(
        ["melanoma", "nevus", "melanoma"], "diagnosis_multiclass"
    )

    assert lesion.class_values == (0, 1)
    assert lesion.class_names == ("no_lesion", "lesion_present")
    assert binary.class_values == (0, 1)
    assert binary.class_names == ("benign", "malignant")
    assert multiclass.class_values == ("melanoma", "nevus")
    assert multiclass.class_names == ("melanoma", "nevus")
    assert multiclass.encode("nevus") == 1
