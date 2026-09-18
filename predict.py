"""Command-line entry point for one-photo model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.inference import metadata_from_json, predict_image, predict_with_lesion_routing


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one ordinary clinical/macro photograph.")
    parser.add_argument("image", type=Path, help="Path to the photograph")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Saved strategy checkpoint")
    source.add_argument("--frozen-config", type=Path, help="Saved frozen ensemble configuration")
    parser.add_argument(
        "--lesion-presence-checkpoint",
        type=Path,
        help="Optional lesion-presence checkpoint used before diagnosis",
    )
    parser.add_argument("--metadata", help="Inline JSON or a JSON/YAML metadata file")
    parser.add_argument("--age", type=float)
    parser.add_argument("--sex")
    parser.add_argument("--site", dest="anatomical_site")
    parser.add_argument("--skin-tone")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--timing-runs", type=int, default=20)
    return parser


def _default_source() -> dict:
    for path in (Path("results/latest_inference.yaml"), Path("configs/inference.yaml")):
        if path.is_file():
            values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if values.get("checkpoint") or values.get("frozen_config"):
                return values
    return {}


def main() -> int:
    """Parse CLI inputs, run inference, and print machine-readable JSON."""
    args = _parser().parse_args()

    try:
        metadata = metadata_from_json(args.metadata) or {}
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid metadata: {exc}") from exc
    for field in ("age", "sex", "anatomical_site", "skin_tone"):
        value = getattr(args, field)
        if value is not None:
            metadata[field] = value

    defaults = _default_source()
    checkpoint = args.checkpoint or defaults.get("checkpoint")
    frozen_config = args.frozen_config or defaults.get("frozen_config")
    if not checkpoint and not frozen_config:
        raise SystemExit(
            "No trained system configured. Pass --checkpoint/--frozen-config or set one in configs/inference.yaml."
        )
    try:
        if args.lesion_presence_checkpoint:
            if not checkpoint or frozen_config:
                raise ValueError("Lesion routing requires one diagnostic --checkpoint, not a frozen ensemble")
            result = predict_with_lesion_routing(
                args.image,
                lesion_presence_checkpoint=args.lesion_presence_checkpoint,
                diagnosis_checkpoint=checkpoint,
                metadata=metadata or None,
                device=args.device,
                repeats=args.timing_runs,
            )
        else:
            result = predict_image(
                args.image,
                checkpoint=checkpoint,
                frozen_config=frozen_config,
                metadata=metadata or None,
                device=args.device,
                repeats=args.timing_runs,
            )
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
        raise SystemExit(f"Inference failed: {exc}") from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
