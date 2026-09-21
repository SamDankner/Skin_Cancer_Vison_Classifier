from __future__ import annotations

import pandas as pd
import pytest

from src.training.gate_sampling import gate_sampling_weights, label_confidence_weights


def test_label_strength_weights_follow_configured_defaults():
    assert label_confidence_weights(["strong", "moderate", "weak", "auxiliary", "ambiguous"]).tolist() == [1.0, .75, .4, .25, 0.0]


def test_gate_sampler_balances_classes_subtypes_and_sources_without_model_fields():
    frame = pd.DataFrame({
        "lesion_present": [1, 1, 0, 0, 0, 0],
        "gate_negative_subtype": ["", "", "healthy_no_visible_lesion", "healthy_no_visible_lesion", "other_skin_condition", "other_skin_condition"],
        "source_dataset": ["positive_a", "positive_b", "healthy_a", "healthy_b", "hard_a", "hard_b"],
        "gate_label_strength": ["strong", "moderate", "strong", "weak", "moderate", "weak"],
    })
    weights = gate_sampling_weights(frame, hard_negative_fraction=.30, source_balance=True)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:2].sum() == pytest.approx(.5)
    assert weights[2:4].sum() == pytest.approx(.35)
    assert weights[4:].sum() == pytest.approx(.15)
    assert "source_dataset" not in {"image", "target"}
