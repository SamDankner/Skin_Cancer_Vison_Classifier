"""Compute-device selection utilities."""
import torch


def get_device() -> torch.device:
    """Select CUDA when available and otherwise return the CPU device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_summary() -> dict:
    """Return a small machine-readable accelerator summary."""
    device = get_device()
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
