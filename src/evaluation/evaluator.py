"""Safe model evaluation workflow."""
from __future__ import annotations
import numpy as np, torch
from .metrics import classification_metrics, lesion_presence_metrics


def _loader_contains_ddi(loader) -> bool:
    """Recognize manifest-backed DDI loaders even when callers omit a name."""
    dataset = getattr(loader, "dataset", None)
    while dataset is not None:
        frame = getattr(dataset, "frame", None)
        if frame is not None and "dataset" in frame and frame["dataset"].fillna("").str.upper().eq("DDI").any():
            return True
        dataset = getattr(dataset, "base", None)
    return False

def evaluate_model(model, loader, device=None, allow_final_test: bool = False, dataset_name: str | None = None, task: str = "diagnosis_binary") -> dict:
    """Evaluate a loader; evaluating DDI requires explicit consent."""
    if ((dataset_name or "").upper() == "DDI" or _loader_contains_ddi(loader)) and not allow_final_test: raise PermissionError("DDI requires allow_final_test=True")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval(); targets, predictions, scores = [], [], []
    with torch.no_grad():
        for batch in loader:
            image, target = batch["image"].to(device), batch["target"]
            valid = torch.tensor([x is not None for x in target], device=device)
            if not valid.any(): continue
            logits = model(image)[valid]; probability = torch.softmax(logits, 1); targets.extend(torch.as_tensor(target, device=device)[valid].cpu().tolist()); predictions.extend(probability.argmax(1).cpu().tolist()); scores.extend(probability.cpu().tolist())
    return lesion_presence_metrics(targets, predictions, scores) if task == "lesion_presence" else classification_metrics(targets, predictions, scores)
