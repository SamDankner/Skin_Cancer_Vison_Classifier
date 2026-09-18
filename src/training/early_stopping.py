"""Reusable validation-based early stopping for model training."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EarlyStopping:
    """Track the best validation value and retain its model state."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"
    monitor: str = "validation_macro_f1"
    best_value: float | None = None
    best_epoch: int | None = None
    stale_epochs: int = 0
    best_state: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def update(self, value: float, epoch: int, state_dict: dict[str, Any]) -> bool:
        """Record one validation result and return whether it is a new best."""
        if self._improved(float(value)):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
            self.best_state = {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
                for key, value in state_dict.items()
            }
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Return whether the configured patience has been exhausted."""
        return self.stale_epochs >= self.patience

    def summary(self, epochs_completed: int, maximum_epochs: int) -> dict[str, Any]:
        """Return persisted stopping details for a completed training run."""
        stopped_early = epochs_completed < maximum_epochs and self.should_stop
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "epoch_stopped": epochs_completed,
            "stopped_early": stopped_early,
            "stop_reason": "early_stopping_patience_exhausted" if stopped_early else "maximum_epochs_reached",
        }
