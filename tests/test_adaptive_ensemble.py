from __future__ import annotations

from pathlib import Path
import yaml
import pytest

from src.ensemble.adaptive_ensemble import AdaptiveEnsemble, metadata_available

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("metadata, expected", [
    ({"age": None, "sex": None, "anatomical_site": None}, False),
    ({"age": 35, "sex": None, "anatomical_site": None}, True),
    ({"age": None, "sex": "female", "anatomical_site": None}, True),
    ({"age": None, "sex": None, "anatomical_site": "trunk"}, True),
    ({"age": 35, "sex": "female", "anatomical_site": "trunk"}, True),
])
def test_multimodal_activation_is_explicit(metadata, expected):
    assert metadata_available(metadata) is expected


def test_v2_two_and_three_model_modes_and_outlier_behavior():
    config = yaml.safe_load((ROOT / "configs/deployments/v2_adaptive/ensemble.yaml").read_text())
    ensemble = AdaptiveEnsemble(config)
    values = {"convnext": .86, "efficientnet": .92, "multimodal": .05}
    no_metadata = ensemble.combine(values, metadata={"age": None, "sex": None, "anatomical_site": None})
    with_metadata = ensemble.combine(values, metadata={"age": 35, "sex": None, "anatomical_site": None})
    assert no_metadata.active_models == ("convnext", "efficientnet")
    assert no_metadata.inactive_models == ("multimodal",)
    assert with_metadata.active_models == ("convnext", "efficientnet", "multimodal")
    assert with_metadata.model_weights["multimodal"] < .10
    ordinary = ensemble.combine({"convnext": .20, "efficientnet": .40, "multimodal": .65}, metadata={"age": 35})
    assert ordinary.model_weights["multimodal"] > .30


def test_versioned_configuration_separates_v1_and_v2():
    v1 = yaml.safe_load((ROOT / "configs/deployments/v1_frozen/ensemble.yaml").read_text())
    v2 = yaml.safe_load((ROOT / "configs/deployments/v2_adaptive/ensemble.yaml").read_text())
    assert v1["status"] == "frozen" and v1["threshold"]["threshold"] == .51
    assert v1["ensemble"]["weights"] == pytest.approx([1 / 3] * 3)
    assert v2["status"] == "experimental" and "metadata_available_policy" in v2["ensemble"]
    assert v1["ensemble"] != v2["ensemble"]
