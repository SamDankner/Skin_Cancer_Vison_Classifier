"""Regression coverage for CNN loss-method dispatch."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from src.strategies.cnn_common import _loss


@pytest.fixture
def loss_inputs():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.5], [0.4, 0.6]])
    targets = torch.tensor([0, 1, 1])
    return logits, targets


@pytest.mark.parametrize("method", ["cross_entropy", "weighted_cross_entropy"])
def test_standard_loss_methods_execute(loss_inputs, method):
    logits, targets = loss_inputs

    result = _loss(logits, targets, method, None, focal_gamma=2.0)

    assert result.ndim == 0
    assert torch.isfinite(result)


def test_focal_loss_executes_and_returns_finite_scalar(loss_inputs):
    logits, targets = loss_inputs

    result = _loss(logits, targets, "focal", None, focal_gamma=2.0)

    assert result.ndim == 0
    assert torch.isfinite(result)


def test_focal_loss_supports_sample_weights(loss_inputs):
    logits, targets = loss_inputs
    sample_weights = torch.tensor([1.0, 0.0, 2.0])
    base = F.cross_entropy(logits, targets, reduction="none")
    expected = ((1 - torch.exp(-base)).pow(2.0) * base * sample_weights).sum() / sample_weights.sum()

    result = _loss(logits, targets, "focal", None, focal_gamma=2.0, sample_weights=sample_weights)

    assert torch.allclose(result, expected)


def test_class_weights_are_preserved(loss_inputs):
    logits, targets = loss_inputs
    class_weights = torch.tensor([1.0, 3.0])
    expected = F.cross_entropy(logits, targets, weight=class_weights, reduction="none").mean()

    result = _loss(logits, targets, "weighted_cross_entropy", class_weights, focal_gamma=2.0)

    assert torch.allclose(result, expected)


def test_invalid_loss_method_raises_value_error(loss_inputs):
    logits, targets = loss_inputs

    with pytest.raises(ValueError, match="loss must be cross_entropy, weighted_cross_entropy, or focal"):
        _loss(logits, targets, "invalid", None, focal_gamma=2.0)
