"""Runtime v1/v2 ensemble selection with explicit metadata eligibility."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any
from .weighting import aggregate

MEMBERS = ("convnext", "efficientnet", "multimodal")


def metadata_available(metadata: Mapping[str, Any]) -> bool:
    return any(value is not None and str(value).strip() for value in metadata.values())


@dataclass(frozen=True)
class EnsembleDecision:
    malignant_probability: float
    predicted_class: str
    threshold: float
    active_models: tuple[str, ...]
    inactive_models: tuple[str, ...]
    model_weights: dict[str, float]
    strategy_name: str


class AdaptiveEnsemble:
    def __init__(self, config: Mapping):
        self.config = config

    def combine(self, probabilities: Mapping[str, float], *, metadata: Mapping[str, Any]) -> EnsembleDecision:
        has_metadata = metadata_available(metadata)
        active = ("convnext", "efficientnet", "multimodal") if has_metadata else ("convnext", "efficientnet")
        if set(active) - set(probabilities):
            raise ValueError("Missing probability for an active ensemble member")
        policy_key = "metadata_available_policy" if has_metadata else "no_metadata_policy"
        policy = self.config["ensemble"][policy_key]
        values = {name: float(probabilities[name]) for name in active}
        probability, weights = aggregate(values, policy)
        threshold = float(self.config["threshold"]["threshold"])
        return EnsembleDecision(probability, "malignant" if probability >= threshold else "benign", threshold,
                                active, tuple(name for name in MEMBERS if name not in active), weights,
                                str(policy["strategy"]))
