"""Shared MobileNetV3 icon, actionability, and node-feature reconstruction."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class IconActionabilityOutput:
    """Independent icon/action logits and 256-D visual node features."""

    icon_logits: torch.Tensor
    actionability_logits: torch.Tensor
    visual_features: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class IconActionabilityLoss:
    """Masked multi-task losses with explicit supervised counts."""

    total: torch.Tensor
    icon: torch.Tensor
    actionability: torch.Tensor
    icon_count: int
    actionability_count: int


class MobileNetV3IconActionability(nn.Module):
    """One shared backbone with three heads and learned missing-crop values."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_dimension: int,
        icon_class_count: int = 87,
        visual_dimension: int = 256,
    ) -> None:
        super().__init__()
        if backbone_dimension <= 0 or icon_class_count <= 1 or visual_dimension <= 0:
            raise ValueError("backbone, icon, and visual dimensions must be positive")
        self.backbone = backbone
        self.backbone_dimension = backbone_dimension
        self.icon_class_count = icon_class_count
        self.visual_dimension = visual_dimension
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.icon_head = nn.Linear(backbone_dimension, icon_class_count)
        self.actionability_head = nn.Linear(backbone_dimension, 4)
        self.visual_projection = nn.Sequential(
            nn.Linear(backbone_dimension, visual_dimension),
            nn.LayerNorm(visual_dimension),
        )
        self.null_icon_logits = nn.Parameter(torch.zeros(icon_class_count))
        self.null_actionability_logits = nn.Parameter(torch.zeros(4))
        self.null_visual_feature = nn.Parameter(torch.zeros(visual_dimension))

    @classmethod
    def from_locked_torchvision(
        cls,
        weight_path: str | Path,
        *,
        icon_class_count: int = 87,
        visual_dimension: int = 256,
    ) -> MobileNetV3IconActionability:
        """Build MobileNetV3-small from a locally locked state dictionary."""

        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"locked MobileNetV3 weight does not exist: {path}")
        try:
            models = importlib.import_module("torchvision.models")
        except ImportError as error:
            raise RuntimeError("install the perception extra for MobileNetV3") from error
        model = models.mobilenet_v3_small(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(
            model.features,
            backbone_dimension=576,
            icon_class_count=icon_class_count,
            visual_dimension=visual_dimension,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the shared backbone for a declared stage."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def forward(
        self,
        crops: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> IconActionabilityOutput:
        """Return fixed-shape outputs; invalid crops use learned null parameters."""

        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("crops must have shape [batch, 3, height, width]")
        batch = crops.shape[0]
        if valid_mask is None:
            valid_mask = torch.ones(batch, dtype=torch.bool, device=crops.device)
        if valid_mask.shape != (batch,) or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean with shape [batch]")
        icon_logits = self.null_icon_logits.unsqueeze(0).expand(batch, -1).clone()
        action_logits = self.null_actionability_logits.unsqueeze(0).expand(batch, -1).clone()
        visual = self.null_visual_feature.unsqueeze(0).expand(batch, -1).clone()
        if bool(valid_mask.any()):
            features = self.backbone(crops[valid_mask])
            if features.ndim != 4 or features.shape[1] != self.backbone_dimension:
                raise ValueError("backbone output does not match declared feature dimension")
            pooled = self.pool(features).flatten(1)
            icon_logits[valid_mask] = self.icon_head(pooled)
            action_logits[valid_mask] = self.actionability_head(pooled)
            visual[valid_mask] = self.visual_projection(pooled)
        return IconActionabilityOutput(icon_logits, action_logits, visual, valid_mask)


def masked_icon_actionability_loss(
    output: IconActionabilityOutput,
    *,
    icon_targets: torch.Tensor,
    icon_mask: torch.Tensor,
    actionability_targets: torch.Tensor,
    actionability_mask: torch.Tensor,
    icon_weight: float = 1.0,
    actionability_weight: float = 1.0,
) -> IconActionabilityLoss:
    """Apply multiclass icon and independent multilabel action losses only where labeled."""

    batch = output.icon_logits.shape[0]
    if icon_targets.shape != (batch,) or icon_targets.dtype is not torch.long:
        raise ValueError("icon_targets must be int64 with shape [batch]")
    if icon_mask.shape != (batch,) or icon_mask.dtype is not torch.bool:
        raise ValueError("icon_mask must be boolean with shape [batch]")
    if actionability_targets.shape != (batch, 4):
        raise ValueError("actionability_targets must have shape [batch, 4]")
    if actionability_mask.shape != (batch, 4) or actionability_mask.dtype is not torch.bool:
        raise ValueError("actionability_mask must be boolean with shape [batch, 4]")
    effective_icon = icon_mask & output.valid_mask
    effective_actions = actionability_mask & output.valid_mask.unsqueeze(1)
    zero = output.icon_logits.sum() * 0.0 + output.actionability_logits.sum() * 0.0
    icon_loss = (
        F.cross_entropy(output.icon_logits[effective_icon], icon_targets[effective_icon])
        if bool(effective_icon.any())
        else zero
    )
    action_loss = (
        F.binary_cross_entropy_with_logits(
            output.actionability_logits[effective_actions],
            actionability_targets[effective_actions],
        )
        if bool(effective_actions.any())
        else zero
    )
    total = icon_loss * icon_weight + action_loss * actionability_weight
    return IconActionabilityLoss(
        total,
        icon_loss,
        action_loss,
        int(effective_icon.sum()),
        int(effective_actions.sum()),
    )
