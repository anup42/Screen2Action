"""Tiny CPU crop encoder and a MobileViT-compatible token interface."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class CropTokenEncoder(nn.Module):
    """Encode a batch of crops into exactly P visual tokens."""

    def __init__(
        self, *, embedding_dim: int = 256, token_count: int = 144, tiny: bool = True
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or token_count <= 0:
            raise ValueError("embedding_dim and token_count must be positive")
        hidden = max(8, embedding_dim // 2)
        self.embedding_dim = embedding_dim
        self.token_count = token_count
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.token_projection = nn.Linear(embedding_dim, embedding_dim)
        self.tiny = tiny

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Return `[batch, P, D]` tokens."""

        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("crops must have shape [batch, 3, H, W]")
        features = self.encoder(crops)
        grid_size = max(1, math.ceil(math.sqrt(self.token_count)))
        features = F.adaptive_avg_pool2d(features, (grid_size, grid_size))
        tokens = features.flatten(2).transpose(1, 2)
        if tokens.shape[1] < self.token_count:
            padding = tokens[:, -1:].expand(-1, self.token_count - tokens.shape[1], -1)
            tokens = torch.cat((tokens, padding), dim=1)
        tokens = tokens[:, : self.token_count]
        return cast(torch.Tensor, self.token_projection(tokens))
