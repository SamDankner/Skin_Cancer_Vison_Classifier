"""One-command entry point for configured development experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.experiment_runner import run_experiments


def main() -> int:
    """Validate arguments, run development experiments, and print a summary."""
    parser = argparse.ArgumentParser(description="Run leakage-safe development experiments; DDI is never accessed.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--strategies", nargs="+", help="Enabled strategy names to run")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration without loading data or models")
    args = parser.parse_args()
    result = run_experiments(args.config, selected_strategies=args.strategies, validate_only=args.validate_only)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
