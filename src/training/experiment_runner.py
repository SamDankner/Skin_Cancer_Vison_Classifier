"""Configuration-driven orchestration and persistence for development runs."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Iterable, Mapping

import pandas as pd
import yaml

from src.data.datasets import validate_manifest
from src.data.splits import leakage_report
from src.utils.reporting import append_experiment_summary
from src.utils.seed import seed_everything
from src.utils.timing import environment_summary


SUPPORTED_STRATEGIES = {"efficientnet", "convnext", "dinov2", "multimodal", "lesion_presence"}


def load_config(path: str | Path) -> dict:
    """Load a YAML mapping from disk."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return config


def create_run_directory(run_name: str, root: str | Path = "results/runs") -> Path:
    """Create one new, non-overwriting run directory."""
    path = Path(root) / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def persist_run(
    run: dict,
    run_directory: str | Path,
    summary_path: str | Path = "results/experiment_summary.csv",
) -> dict:
    """Persist the full run contract and update the cross-run summary."""
    path = Path(run_directory)
    path.mkdir(parents=True, exist_ok=True)
    saved = {**run, "run_name": run.get("run_name", path.name), "saved_at": time.time()}
    (path / "run.json").write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
    (path / "config.yaml").write_text(
        yaml.safe_dump(saved.get("config", {}), sort_keys=False), encoding="utf-8"
    )
    if saved.get("history"):
        pd.DataFrame(saved["history"]).to_csv(path / "epoch_history.csv", index=False)
    append_experiment_summary(saved, summary_path)
    return saved


def _contains_ddi_reference(value, field: str | None = None) -> bool:
    data_fields = {
        "dataset", "datasets", "manifest", "manifest_path", "data_path", "data_dir",
        "source", "sources",
    }
    if isinstance(value, Mapping):
        return any(_contains_ddi_reference(nested, str(key)) for key, nested in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_ddi_reference(item, field) for item in value)
    if field is None:
        return False
    name = field.lower()
    is_data_field = name in data_fields or any(token in name for token in ("dataset", "manifest"))
    return is_data_field and "DDI" in str(value).upper()


def validate_experiment_config(config: Mapping, *, root: str | Path = ".") -> dict:
    """Validate a top-level development configuration without starting training."""
    if not isinstance(config, Mapping):
        raise ValueError("Top-level experiment configuration must be a mapping")
    if _contains_ddi_reference(config):
        raise PermissionError("Normal experiment configuration cannot reference DDI")
    manifest_path = config.get("manifest_path")
    if not manifest_path:
        raise ValueError("manifest_path is required")
    strategies = config.get("strategies")
    if not isinstance(strategies, Mapping) or not strategies:
        raise ValueError("strategies must be a non-empty mapping")
    unknown = set(strategies) - SUPPORTED_STRATEGIES
    if unknown:
        raise ValueError(f"Unsupported strategies: {sorted(unknown)}")
    enabled = [name for name, definition in strategies.items() if definition.get("enabled", False)]
    if not enabled:
        raise ValueError("Enable at least one development strategy")
    for name, definition in strategies.items():
        if not isinstance(definition, Mapping):
            raise ValueError(f"Strategy {name!r} must be a mapping")
        config_path = definition.get("config")
        if definition.get("enabled") and not config_path:
            raise ValueError(f"Enabled strategy {name!r} needs a config path")
        if config_path and not (Path(root) / config_path).is_file():
            raise FileNotFoundError(Path(root) / config_path)
    search = config.get("parameter_search") or {}
    if search.get("enabled"):
        if int(search.get("top_k", 0)) < 1:
            raise ValueError("Enabled parameter search needs top_k >= 1")
        candidates = search.get("screening_candidates") or []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("Each screening candidate must be a mapping")
            forbidden = set(candidate) & {"task", "dataset", "manifest", "manifest_path"}
            if forbidden:
                raise ValueError(f"Search candidates cannot change scientific scope: {sorted(forbidden)}")
    return {"enabled_strategies": enabled, "manifest_path": str(manifest_path)}


def _strategy_train(name: str):
    if name == "efficientnet":
        from src.strategies.efficientnet_strategy import train
    elif name == "convnext":
        from src.strategies.convnext_strategy import train
    elif name == "lesion_presence":
        from src.strategies.lesion_presence_strategy import train
    elif name == "dinov2":
        from src.strategies.dinov2_strategy import train_dinov2 as train
    elif name == "multimodal":
        from src.strategies.multimodal_strategy import train
    else:
        raise ValueError(name)
    return train


def _deep_merge(base: dict, override: Mapping) -> dict:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = deepcopy(value)
    return output


def _candidate_score(result, metric: str) -> float:
    value = result.metrics.get(metric)
    if value is None:
        raise ValueError(f"Selection metric {metric!r} is undefined for candidate")
    return float(value)


def _run_one(
    name: str,
    manifest: pd.DataFrame,
    config: dict,
    run_name: str,
    run_directory: Path,
):
    trainer = _strategy_train(name)
    prepared = {
        **config,
        "run_name": run_name,
        "run_directory": str(run_directory),
    }
    return trainer(manifest, prepared, run_name=run_name) if name in {"efficientnet", "convnext", "lesion_presence", "multimodal"} else trainer(manifest, prepared)


def _run_staged_search(
    name: str,
    manifest: pd.DataFrame,
    base_config: dict,
    definition: Mapping,
    search: Mapping,
    strategy_directory: Path,
    invocation_id: str,
):
    metric = str(base_config.get("selection_metric", "macro_f1"))
    selection_mode = str(base_config.get("selection_mode", "min" if "loss" in metric else "max"))
    screening_defaults = _deep_merge(
        dict(search.get("screening_overrides") or {}),
        definition.get("screening_overrides") or {},
    )
    stage_one = []
    candidates = definition.get("screening_candidates") or search.get("screening_candidates")
    if not candidates:
        raise ValueError(f"No screening candidates configured for {name}")
    for index, candidate in enumerate(candidates, start=1):
        config = _deep_merge(_deep_merge(base_config, screening_defaults), candidate)
        config = _deep_merge(config, definition.get("overrides") or {})
        candidate_name = f"{invocation_id}_{name}_screen_{index:02d}"
        result = _run_one(name, manifest, config, candidate_name, strategy_directory / "search" / "screening" / f"candidate_{index:02d}")
        stage_one.append((result, candidate))
    ranked = sorted(
        stage_one,
        key=lambda item: _candidate_score(item[0], metric),
        reverse=selection_mode == "max",
    )
    finalists = ranked[: min(int(search["top_k"]), len(ranked))]
    stage_two = []
    for index, (_, candidate) in enumerate(finalists, start=1):
        config = _deep_merge(_deep_merge(base_config, candidate), search.get("serious_overrides") or {})
        config = _deep_merge(config, definition.get("overrides") or {})
        candidate_name = f"{invocation_id}_{name}_serious_{index:02d}"
        result = _run_one(name, manifest, config, candidate_name, strategy_directory / "search" / "serious" / f"candidate_{index:02d}")
        stage_two.append(result)
    comparison = [{
        "run_name": result.config.get("run_name"),
        "best_epoch": result.best_epoch,
        "validation_metric": _candidate_score(result, metric),
        "training_seconds": result.timing.get("training_seconds"),
        "config": result.config,
    } for result, _ in stage_one]
    comparison.extend({
        "run_name": result.config.get("run_name"),
        "best_epoch": result.best_epoch,
        "validation_metric": _candidate_score(result, metric),
        "training_seconds": result.timing.get("training_seconds"),
        "config": result.config,
        "stage": "serious",
    } for result in stage_two)
    strategy_directory.mkdir(parents=True, exist_ok=True)
    (strategy_directory / "parameter_search.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key != "config"} for row in comparison]).to_csv(
        strategy_directory / "parameter_search.csv", index=False
    )
    from src.evaluation.reporting import plot_parameter_search
    plot_parameter_search(
        pd.DataFrame([{key: value for key, value in row.items() if key != "config"} for row in comparison]),
        strategy_directory / "parameter_search.png",
        metric,
    )
    selector = max if selection_mode == "max" else min
    return selector(stage_two, key=lambda result: _candidate_score(result, metric))


def run_experiments(
    config_path: str | Path = "configs/experiments.yaml",
    *,
    selected_strategies: Iterable[str] | None = None,
    validate_only: bool = False,
) -> dict:
    """Run selected development strategies without accessing the protected DDI test."""
    path = Path(config_path)
    config = load_config(path)
    validated = validate_experiment_config(config, root=path.parent.parent if path.parent.name == "configs" else ".")
    selected = list(selected_strategies or validated["enabled_strategies"])
    invalid = set(selected) - set(validated["enabled_strategies"])
    if invalid:
        raise ValueError(f"Requested strategies are not enabled: {sorted(invalid)}")
    if validate_only:
        return {"status": "validated", **validated, "selected_strategies": selected}

    manifest_path = Path(config["manifest_path"])
    manifest = validate_manifest(pd.read_csv(manifest_path, low_memory=False))
    if manifest.dataset.fillna("").str.upper().eq("DDI").any():
        raise PermissionError("Normal experiment runs cannot contain DDI")
    if not leakage_report(manifest).empty:
        raise ValueError("Patient/lesion/image groups cross development splits")
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(set(manifest.split.dropna())):
        raise ValueError("Development manifest must contain train, validation, and test splits")

    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=bool(config.get("deterministic", False)))
    invocation_id = config.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = create_run_directory(invocation_id, config.get("output_root", "results/runs"))
    summary_path = str(config.get("summary_path", "results/experiment_summary.csv"))
    started = time.perf_counter()
    (root / "invocation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (root / "environment.json").write_text(json.dumps(environment_summary(), indent=2), encoding="utf-8")
    manifest.groupby(["dataset", "split"], dropna=False).size().rename("count").reset_index().to_csv(
        root / "dataset_split_summary.csv", index=False
    )

    results = {}
    search = config.get("parameter_search") or {}
    for name in selected:
        definition = config["strategies"][name]
        strategy_config = load_config(definition["config"])
        strategy_config = _deep_merge(strategy_config, definition.get("overrides") or {})
        strategy_config.update({
            "seed": seed,
            "models_root": config.get("models_root", "models"),
            "summary_path": summary_path,
        })
        strategy_directory = root / name
        if search.get("enabled") and definition.get("search", True):
            result = _run_staged_search(
                name, manifest, strategy_config, definition, search,
                strategy_directory, invocation_id,
            )
        else:
            result = _run_one(
                name, manifest, strategy_config, f"{invocation_id}_{name}", strategy_directory
            )
        results[name] = result.as_dict()

    summary = {
        "run_id": invocation_id,
        "status": "complete",
        "seed": seed,
        "manifest_path": str(manifest_path),
        "strategies": selected,
        "total_experiment_seconds": time.perf_counter() - started,
        "results": results,
        "ddi_accessed": False,
    }
    default_strategy = config.get("default_inference_strategy")
    if default_strategy:
        if default_strategy not in results:
            default_strategy = selected[0]
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/latest_inference.yaml").write_text(
            yaml.safe_dump({"checkpoint": results[default_strategy]["best_checkpoint"], "frozen_config": None}),
            encoding="utf-8",
        )
    (root / "invocation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
