"""Configurable optimizer and learning-rate scheduler construction."""
from __future__ import annotations

import torch


def build_optimizer(parameters, config: dict, *, default_lr: float):
    """Build AdamW or momentum SGD from a strategy configuration."""
    name = str(config.get("optimizer", "adamw")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=default_lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=default_lr,
            weight_decay=weight_decay,
            momentum=float(config.get("momentum", 0.9)),
            nesterov=bool(config.get("nesterov", True)),
        )
    raise ValueError("optimizer must be 'adamw' or 'sgd'")


def build_scheduler(optimizer, config: dict):
    """Build a plateau or cosine scheduler, or disable scheduling."""
    name = str(config.get("scheduler", "reduce_on_plateau")).lower()
    if name in {"none", "disabled"}:
        return None
    if name == "reduce_on_plateau":
        monitor = str(config.get("selection_metric", "macro_f1"))
        mode = str(config.get("selection_mode", "min" if "loss" in monitor else "max"))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 2)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config.get("epochs", 1))),
            eta_min=float(config.get("minimum_learning_rate", 0.0)),
        )
    raise ValueError("scheduler must be reduce_on_plateau, cosine, or none")


def step_scheduler(scheduler, config: dict, validation_score: float) -> None:
    """Step a scheduler at the correct time with the correct argument."""
    if scheduler is None:
        return
    if str(config.get("scheduler", "reduce_on_plateau")).lower() == "reduce_on_plateau":
        scheduler.step(validation_score)
    else:
        scheduler.step()
