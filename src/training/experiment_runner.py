"""Experiment configuration and result persistence."""
from __future__ import annotations
import json, time
from pathlib import Path
import yaml
from src.utils.reporting import append_experiment_summary
def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle: return yaml.safe_load(handle) or {}
def create_run_directory(run_name: str, root: str | Path = "results/runs") -> Path:
    path = Path(root) / run_name; path.mkdir(parents=True, exist_ok=False); return path
def persist_run(run: dict, run_directory: str | Path, summary_path: str | Path = "results/experiment_summary.csv") -> dict:
    path = Path(run_directory); path.mkdir(parents=True, exist_ok=True); run = {**run, "run_name": run.get("run_name", path.name), "saved_at": time.time()}
    (path / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8"); (path / "config.yaml").write_text(yaml.safe_dump(run.get("config", {}), sort_keys=False), encoding="utf-8")
    if run.get("history"): __import__("pandas").DataFrame(run["history"]).to_csv(path / "epoch_history.csv", index=False)
    append_experiment_summary(run, summary_path); return run
