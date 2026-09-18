"""Standardized experiment reporting utilities."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def flatten_dict(values: dict, prefix: str = "") -> dict:
    """Flatten nested mappings for a one-row experiment summary."""
    output = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        output.update(flatten_dict(value, name) if isinstance(value, dict) else {name: value})
    return output

def report_run(run: dict) -> str:
    """Render the common result contract as concise readable text."""
    cfg, metrics = run.get("config", {}), run.get("metrics", {})
    keys = ["strategy_name", "backbone", "dataset_combination", "image_size", "batch_size", "optimizer", "learning_rate", "weight_decay", "scheduler", "frozen", "epochs"]
    lines = [f"{key}: {cfg.get(key, run.get(key, 'n/a'))}" for key in keys]
    lines += [f"best_epoch: {run.get('best_epoch', 'n/a')}", f"training_time_seconds: {run.get('timing', {}).get('training_seconds', 'n/a')}"]
    lines += [f"{key}: {metrics.get(key, 'n/a')}" for key in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "sensitivity", "specificity", "roc_auc", "pr_auc"]]
    return "\n".join(lines)

def append_experiment_summary(run: dict, path: str | Path = "results/experiment_summary.csv") -> Path:
    """Append a flattened run record for sortable cross-experiment comparison."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        **flatten_dict(run.get("config", {})),
        **flatten_dict(run.get("metrics", {}), "metrics"),
        **flatten_dict(run.get("timing", {}), "timing"),
        "run_name": run.get("run_name"), "best_epoch": run.get("best_epoch"),
        "best_checkpoint": run.get("best_checkpoint"), "status": run.get("status", "complete"),
        "leakage_free": run.get("leakage_free", False),
    }
    existing = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False); return path
