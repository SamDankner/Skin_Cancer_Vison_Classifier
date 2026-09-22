"""Gradient-weighted patch-token attribution for DINOv2 vision transformers."""
from __future__ import annotations

import torch

from .gradcam import normalized_map


def _patch_grid(backbone: torch.nn.Module, image: torch.Tensor, token_count: int) -> tuple[int, int, int]:
    patch_size = getattr(getattr(backbone, "patch_embed", None), "patch_size", 14)
    if isinstance(patch_size, tuple):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_width = int(patch_size)
    grid_height, grid_width = image.shape[-2] // patch_height, image.shape[-1] // patch_width
    patch_tokens = grid_height * grid_width
    # DINOv2 exposes its CLS-token count and optional register-token count.
    # Do not infer special tokens from a remainder: a changed token layout
    # should fail clearly instead of silently shifting the spatial patch grid.
    prefix_tokens = int(getattr(backbone, "num_tokens", 1)) + int(getattr(backbone, "num_register_tokens", 0))
    if token_count != prefix_tokens + patch_tokens:
        raise ValueError(
            "DINOv2 token count does not match its CLS/register-token and image patch-grid contract."
        )
    return grid_height, grid_width, prefix_tokens


def transformer_token_attribution(model: torch.nn.Module, inputs: tuple, target_index: int) -> torch.Tensor:
    """Map malignant-logit gradients onto DINOv2 patch tokens entering its last block.

    This is intentionally token attribution, not CNN Grad-CAM. A forward
    *pre-hook* records the sequence entering the final DINOv2 transformer
    block. At that point patch tokens can still affect the CLS token through
    final-block self-attention; final-block output patch tokens cannot. The
    malignant-logit gradient is multiplied by each patch activation and the
    absolute embedding sum is used as a class-specific local-influence
    magnitude. Metadata remains fixed, so only image-token scores are shown.
    """
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if not blocks:
        raise ValueError("The DINOv2 backbone does not expose transformer blocks for token attribution.")
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, args):
        if not args:
            raise ValueError("The final DINOv2 block received no token tensor.")
        tokens = args[0]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("The final DINOv2 block did not return patch tokens.")
        tokens.retain_grad()
        captured["tokens"] = tokens

    was_training = model.training
    handle = blocks[-1].register_forward_pre_hook(capture)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(*inputs)
        if logits.ndim != 2 or not 0 <= target_index < logits.shape[1]:
            raise ValueError("The malignant output index is not valid for this model.")
        logits[:, target_index].sum().backward()
        tokens = captured.get("tokens")
        if tokens is None or tokens.grad is None:
            raise RuntimeError("Transformer attribution could not capture token gradients.")
        grid_height, grid_width, prefix_tokens = _patch_grid(backbone, inputs[0], tokens.shape[1])
        patches = tokens[:, prefix_tokens:, :]
        patch_gradients = tokens.grad[:, prefix_tokens:, :]
        # Token embeddings and their target gradients are signed. Their
        # absolute activation×gradient magnitude preserves spatial influence
        # rather than dropping all negative but informative contributions.
        scores = (patches * patch_gradients).abs().sum(dim=-1)[0]
        if scores.numel() != grid_height * grid_width:
            raise ValueError("DINOv2 patch-token count does not match the image patch grid.")
        score_range = scores.detach().amax() - scores.detach().amin()
        if not torch.isfinite(scores).all() or float(score_range) <= torch.finfo(scores.dtype).eps:
            raise RuntimeError("DINOv2 attribution was degenerate; no spatial token variation was available.")
        return normalized_map(scores.reshape(grid_height, grid_width).detach(), tuple(inputs[0].shape[-2:])).detach().cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
