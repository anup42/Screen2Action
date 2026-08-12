"""Configurable command Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CommandEncoding:
    """Transformer token states and pooled command representation."""

    token_states: torch.Tensor
    pooled: torch.Tensor


class CommandEncoder(nn.Module):
    """Six-layer reference command encoder with CPU-friendly tiny settings."""

    def __init__(
        self,
        *,
        vocab_size: int = 16_384,
        embedding_dim: int = 256,
        layers: int = 6,
        heads: int = 4,
        feedforward_dim: int = 1024,
        max_length: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if min(vocab_size, embedding_dim, layers, heads, feedforward_dim, max_length) <= 0:
            raise ValueError("command encoder dimensions must be positive")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position = nn.Embedding(max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CommandEncoding:
        """Encode `[B,T]` IDs, returning states and first-token pooling."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        batch, length = input_ids.shape
        if length > self.max_length:
            raise ValueError("command length exceeds configured maximum")
        if attention_mask is None:
            attention_mask = input_ids != 0
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        hidden = self.embedding(input_ids) + self.position(positions)
        hidden = self.transformer(hidden, src_key_padding_mask=~attention_mask.bool())
        hidden = self.norm(hidden)
        pooled = hidden[:, 0]
        return CommandEncoding(token_states=hidden, pooled=pooled)
