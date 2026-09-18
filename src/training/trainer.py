"""Shared training-result contract for every experiment strategy."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    """Describe model selection, evaluation, timing, and persisted artifacts."""

    history: list[dict] = field(default_factory=list)
    validation_history: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    best_checkpoint: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    development_test_metrics: dict[str, Any] = field(default_factory=dict)
    early_stopping: dict[str, Any] = field(default_factory=dict)
    run_directory: str | None = None

    def as_dict(self) -> dict:
        """Return a serializable shallow mapping of result fields."""
        return self.__dict__.copy()
