"""Shared icon/actionability model shape, mask, and gradient tests."""

from __future__ import annotations

import torch
from torch import nn

from screen2action.perception.icon_actionability import (
    MobileNetV3IconActionability,
    masked_icon_actionability_loss,
)


def _model() -> MobileNetV3IconActionability:
    return MobileNetV3IconActionability(
        nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU()),
        backbone_dimension=8,
    )


def test_shared_model_has_independent_heads_nulls_and_gradients() -> None:
    torch.manual_seed(7)
    model = _model()
    crops = torch.randn(3, 3, 16, 16)
    valid = torch.tensor([True, False, True])

    output = model(crops, valid)

    assert output.icon_logits.shape == (3, 87)
    assert output.actionability_logits.shape == (3, 4)
    assert output.visual_features.shape == (3, 256)
    assert torch.equal(output.icon_logits[1], model.null_icon_logits)
    assert torch.equal(output.actionability_logits[1], model.null_actionability_logits)
    assert torch.equal(output.visual_features[1], model.null_visual_feature)
    probabilities = output.actionability_logits.sigmoid()
    assert not torch.allclose(probabilities.sum(dim=1), torch.ones(3))

    loss = masked_icon_actionability_loss(
        output,
        icon_targets=torch.tensor([1, 2, 3]),
        icon_mask=torch.tensor([True, True, False]),
        actionability_targets=torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
        ),
        actionability_mask=torch.tensor(
            [[True, True, False, False], [True, True, True, True], [False, True, False, True]]
        ),
    )
    assert loss.icon_count == 1
    assert loss.actionability_count == 4
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.icon_head.weight.grad is not None
    assert model.actionability_head.weight.grad is not None
    assert next(model.backbone.parameters()).grad is not None


def test_backbone_freezing_is_stage_controlled() -> None:
    model = _model()
    model.set_backbone_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.icon_head.parameters())

    model.set_backbone_trainable(True)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_fixed_export_visual_path_preserves_invalid_crop_nulls() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    crops = torch.randn(3, 3, 16, 16)
    valid = torch.tensor([True, False, True])

    compact = model(crops, valid)
    exported = model.forward_export(crops, valid)

    assert torch.allclose(compact.icon_logits, exported.icon_logits)
    assert torch.allclose(compact.actionability_logits, exported.actionability_logits)
    assert torch.allclose(compact.visual_features, exported.visual_features, atol=1e-6, rtol=1e-5)
