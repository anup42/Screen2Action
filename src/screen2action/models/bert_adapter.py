"""Offline-only compact BERT command adapter with optional shared WordPiece embeddings."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from screen2action.models.command_encoder import CommandEncoding


class CompactBertAdapter(nn.Module):
    """Expose CLS retrieval and token states from a locally locked AutoModel."""

    def __init__(
        self,
        model: nn.Module,
        *,
        tokenizer: object | None = None,
        max_length: int = 64,
        expected_dimension: int = 256,
    ) -> None:
        super().__init__()
        if max_length <= 0 or expected_dimension <= 0:
            raise ValueError("BERT length and dimension must be positive")
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", expected_dimension)
        if int(hidden_size) != expected_dimension:
            raise ValueError(
                f"compact BERT hidden size must be {expected_dimension}, got {hidden_size}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_dim = expected_dimension

    @classmethod
    def from_locked_pretrained(
        cls,
        model_directory: str | Path,
        *,
        max_length: int = 64,
    ) -> CompactBertAdapter:
        """Load tokenizer/model from local bytes and explicitly forbid Hub access."""

        path = Path(model_directory)
        if not path.is_dir():
            raise FileNotFoundError(f"locked compact BERT directory does not exist: {path}")
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("install the perception extra for compact BERT") from error
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            use_fast=True,
        )
        model = transformers.AutoModel.from_pretrained(path, local_files_only=True)
        return cls(model, tokenizer=tokenizer, max_length=max_length)

    @property
    def wordpiece_embeddings(self) -> nn.Embedding:
        """Return the model's input embeddings for the shared node-text strategy."""

        getter = getattr(self.model, "get_input_embeddings", None)
        if getter is None:
            raise ValueError("compact BERT model does not expose input embeddings")
        embeddings = getter()
        if not isinstance(embeddings, nn.Embedding):
            raise ValueError("compact BERT input embeddings are not an nn.Embedding")
        return embeddings

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(trainable)

    def tokenize(
        self,
        commands: Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize commands to fixed maximum length with attention masks."""

        if self.tokenizer is None:
            raise RuntimeError("no tokenizer was attached to the compact BERT adapter")
        if not commands or any(not command.strip() for command in commands):
            raise ValueError("commands must be non-empty strings")
        encoded = cast(Any, self.tokenizer)(
            list(commands),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = cast(torch.Tensor, encoded["input_ids"])
        attention_mask = cast(torch.Tensor, encoded["attention_mask"]).bool()
        if device is not None:
            target = torch.device(device)
            input_ids = input_ids.to(target)
            attention_mask = attention_mask.to(target)
        return input_ids, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CommandEncoding:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_length:
            raise ValueError("compact BERT input_ids must have shape [B,T] with T <= 64")
        if attention_mask is None:
            attention_mask = input_ids != 0
        if attention_mask.shape != input_ids.shape:
            raise ValueError("compact BERT attention mask must match input IDs")
        output = cast(Any, self.model)(
            input_ids=input_ids,
            attention_mask=attention_mask.to(torch.long),
            return_dict=True,
        )
        states = cast(torch.Tensor, output.last_hidden_state)
        if states.shape[:2] != input_ids.shape or states.shape[-1] != self.embedding_dim:
            raise ValueError("compact BERT output violates the token-state contract")
        return CommandEncoding(token_states=states, pooled=states[:, 0])
