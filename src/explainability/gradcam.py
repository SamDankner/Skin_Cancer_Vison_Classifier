"""Small Grad-CAM implementation used by the two CNN deployment members."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_map(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a one-image spatial map and safely normalize it to [0, 1]."""
    values = F.interpolate(values[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    values = values - values.amin()
    maximum = values.amax()
    return values / maximum if float(maximum) > 0 else torch.zeros_like(values)


def gradcam(model: torch.nn.Module, inputs: tuple, target_layer: torch.nn.Module, target_index: int) -> torch.Tensor:
    """Return a normalized Grad-CAM map for ``target_index`` from a CNN logit.

    The temporary hook is removed even when a model raises.  Gradients are
    enabled only by this caller; ordinary predictor inference remains in
    ``torch.inference_mode``.
    """
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("The selected Grad-CAM layer did not return a spatial feature tensor.")
        output.retain_grad()
        captured["activation"] = output

    was_training = model.training
    handle = target_layer.register_forward_hook(capture)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(*inputs)
        if logits.ndim != 2 or not 0 <= target_index < logits.shape[1]:
            raise ValueError("The malignant output index is not valid for this model.")
        logits[:, target_index].sum().backward()
        activation = captured.get("activation")
        if activation is None or activation.grad is None:
            raise RuntimeError("Grad-CAM could not capture activation gradients.")
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0]
        return normalized_map(cam.detach(), tuple(inputs[0].shape[-2:])).detach().cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
