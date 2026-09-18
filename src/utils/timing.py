"""Accurate device-aware timing and environment capture utilities."""
from __future__ import annotations

import platform
import statistics
import sys
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Callable

import numpy as np
import torch


def synchronize_device(device: torch.device | str) -> None:
    """Synchronize CUDA so timings include completed GPU work."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def benchmark_callable(
    operation: Callable[[], object],
    *,
    device: torch.device | str,
    warmup: int = 3,
    repeats: int = 20,
) -> dict:
    """Time an inference operation after warm-up and return latency statistics."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    resolved = torch.device(device)
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize_device(resolved)
        durations = []
        for _ in range(repeats):
            synchronize_device(resolved)
            started = perf_counter()
            operation()
            synchronize_device(resolved)
            durations.append(perf_counter() - started)
    milliseconds = np.asarray(durations, dtype=float) * 1000.0
    return {
        "device": str(resolved),
        "warmup_runs": warmup,
        "timed_samples": repeats,
        "mean_ms": float(milliseconds.mean()),
        "median_ms": float(statistics.median(milliseconds.tolist())),
        "p95_ms": float(np.percentile(milliseconds, 95)),
    }


def environment_summary() -> dict:
    """Return reproducibility-relevant runtime and accelerator versions."""
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = None
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "Pillow", "matplotlib", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": packages,
    }
