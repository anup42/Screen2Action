"""Locked MobileViT-S spatial crop-token adapter."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F


class MobileVitSCropEncoder(nn.Module):
    """Project the final MobileViT-S feature map to a 12x12 grid of 256-D tokens."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_dimension: int,
        embedding_dim: int = 256,
        grid_size: int = 12,
        input_size: int = 192,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        if min(backbone_dimension, embedding_dim, grid_size, input_size) <= 0:
            raise ValueError("MobileViT dimensions must be positive")
        self.backbone = backbone
        self.backbone_dimension = backbone_dimension
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.input_size = input_size
        self.normalize_input = normalize_input
        self.projection = nn.Linear(backbone_dimension, embedding_dim)
        self.input_mean: torch.Tensor
        self.register_buffer(
            "input_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.input_std: torch.Tensor
        self.register_buffer(
            "input_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )

    @property
    def token_count(self) -> int:
        return self.grid_size * self.grid_size

    @classmethod
    def from_locked_timm(cls, weight_path: str | Path) -> MobileVitSCropEncoder:
        """Reconstruct timm locally and load only a caller-selected locked file."""

        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"locked MobileViT weight does not exist: {path}")
        try:
            timm = importlib.import_module("timm")
        except ImportError as error:
            raise RuntimeError("install the perception extra for MobileViT-S") from error
        backbone = timm.create_model(
            "mobilevit_s.cvnets_in1k",
            pretrained=False,
            features_only=True,
            out_indices=(-1,),
        )
        state = _load_state_dict(path)
        incompatible = backbone.load_state_dict(state, strict=False)
        disallowed_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith(("classifier", "head", "fc"))
        ]
        disallowed_unexpected = [
            name
            for name in incompatible.unexpected_keys
            if not name.startswith(("classifier", "head", "fc"))
        ]
        if disallowed_missing or disallowed_unexpected:
            raise ValueError(
                "locked MobileViT state does not match the configured feature backbone: "
                f"missing={disallowed_missing[:5]}, unexpected={disallowed_unexpected[:5]}"
            )
        channels = backbone.feature_info.channels()
        return cls(backbone, backbone_dimension=int(channels[-1]))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Return `[B, 144, 256]` for `[B, 3, 192, 192]` paper-profile crops."""

        if crops.ndim != 4 or crops.shape[1:] != (3, self.input_size, self.input_size):
            raise ValueError(
                f"MobileViT crops must have shape [B,3,{self.input_size},{self.input_size}]"
            )
        values = crops.float()
        if self.normalize_input:
            values = (values - self.input_mean) / self.input_std
        raw = self.backbone(values)
        if isinstance(raw, Sequence) and not isinstance(raw, torch.Tensor):
            if not raw:
                raise ValueError("MobileViT backbone returned no feature maps")
            raw = raw[-1]
        if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
            raise ValueError("MobileViT backbone must return a four-dimensional feature map")
        if raw.shape[1] != self.backbone_dimension:
            raise ValueError("MobileViT feature channels do not match the declared dimension")
        pooled = F.adaptive_avg_pool2d(raw, (self.grid_size, self.grid_size))
        tokens = pooled.flatten(2).transpose(1, 2)
        return cast(torch.Tensor, self.projection(tokens))

    def shape_probe(
        self, *, batch_size: int = 1, device: str | torch.device = "cpu"
    ) -> dict[str, object]:
        """Run a gradient-preserving shape probe under caller-controlled module mode."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        target = torch.device(device)
        sample = torch.zeros(
            (batch_size, 3, self.input_size, self.input_size),
            dtype=torch.float32,
            device=target,
        )
        output = self(sample)
        return {
            "input_shape": list(sample.shape),
            "output_shape": list(output.shape),
            "device": output.device.type,
            "finite": bool(torch.isfinite(output).all()),
        }


def _load_state_dict(path: Path) -> Mapping[str, Any]:
    if path.suffix.casefold() == ".safetensors":
        try:
            safetensors = importlib.import_module("safetensors.torch")
        except ImportError as error:
            raise RuntimeError("install safetensors to load this MobileViT lock") from error
        state = safetensors.load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("locked MobileViT file does not contain a state dictionary")
    return cast(Mapping[str, Any], state)
