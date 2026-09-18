"""Reproducibility utilities."""
import os, random
import numpy as np
import torch
def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch, including CUDA when available."""
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic; torch.backends.cudnn.benchmark = not deterministic
    return seed
