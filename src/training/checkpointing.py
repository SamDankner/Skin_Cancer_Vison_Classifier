"""Model checkpoint persistence utilities."""
from pathlib import Path
import torch
def checkpoint_path(strategy_name: str, backbone: str, run_name: str, root: str | Path = "models") -> Path:
    path = Path(root) / strategy_name; path.mkdir(parents=True, exist_ok=True); return path / f"{strategy_name}_{backbone}_{run_name}.pt"
def save_checkpoint(model, optimizer, epoch: int, path: str | Path, **extra) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer else None, "epoch": epoch, **extra}, path); return path
