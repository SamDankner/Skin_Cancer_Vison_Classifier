"""Minimal shared training-result contract for strategy implementations."""
from dataclasses import dataclass, field
from typing import Any
@dataclass
class TrainingResult:
    """Common result returned by every strategy."""
    history: list[dict] = field(default_factory=list)
    validation_history: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    best_checkpoint: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    def as_dict(self) -> dict: return self.__dict__.copy()
