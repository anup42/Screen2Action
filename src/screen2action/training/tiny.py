"""Tiny CPU model and deterministic overfit workflow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.models.command_encoder import CommandEncoder
from screen2action.training.seed import seed_everything


@dataclass(frozen=True, slots=True)
class TinyDataset:
    """Tensor dataset for a small candidate/point overfit."""

    node_features: torch.Tensor
    command_ids: torch.Tensor
    command_mask: torch.Tensor
    target_index: torch.Tensor
    target_point: torch.Tensor


@dataclass(frozen=True, slots=True)
class TinyTrainingResult:
    """Tiny training measurements and trained model."""

    model: TinyGroundingModel
    initial_loss: float
    final_loss: float
    final_accuracy: float
    steps: int


class TinyGroundingModel(nn.Module):
    """CPU model preserving command encoding, cosine retrieval, and point heads."""

    def __init__(self, *, vocab_size: int, feature_dim: int = 4, embedding_dim: int = 32) -> None:
        super().__init__()
        self.command_encoder = CommandEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            layers=1,
            heads=2,
            feedforward_dim=embedding_dim * 4,
            max_length=16,
            dropout=0.0,
        )
        self.node_projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.point_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 2),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        command_ids: torch.Tensor,
        command_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidate logits `[B,N]` and local points `[B,N,2]`."""

        command = self.command_encoder(command_ids, command_mask).pooled
        nodes = self.node_projection(node_features)
        query = F.normalize(command, dim=-1)
        keys = F.normalize(nodes, dim=-1)
        logits = torch.einsum("bd,bnd->bn", query, keys) * 8.0
        query_expanded = command.unsqueeze(1).expand_as(nodes)
        points = torch.sigmoid(self.point_head(torch.cat((nodes, query_expanded), dim=-1)))
        return logits, points


def build_tiny_dataset(tokenizer: VocabularyTokenizer, count: int = 32) -> TinyDataset:
    """Create 16-32 deterministic examples with distinct command/node matches."""

    if count <= 0:
        raise ValueError("count must be positive")
    commands = ("tap red", "tap blue", "tap green", "tap yellow")
    target_indices = (0, 1, 2, 3)
    features = torch.eye(4, dtype=torch.float32)
    node_features = features.unsqueeze(0).expand(count, -1, -1).clone()
    texts = [commands[index % len(commands)] for index in range(count)]
    command_ids = torch.tensor(tokenizer.encode_batch(texts, max_length=16), dtype=torch.long)
    return TinyDataset(
        node_features=node_features,
        command_ids=command_ids,
        command_mask=command_ids != tokenizer.pad_id,
        target_index=torch.tensor(
            [target_indices[index % len(target_indices)] for index in range(count)],
            dtype=torch.long,
        ),
        target_point=torch.full((count, 2), 0.5, dtype=torch.float32),
    )


def train_tiny_model(
    dataset: TinyDataset,
    *,
    vocab_size: int,
    steps: int = 120,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> TinyTrainingResult:
    """Overfit the tiny fixture and return loss/accuracy evidence."""

    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    seed_everything(seed)
    model = TinyGroundingModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    model.train()
    with torch.no_grad():
        initial_logits, initial_points = model(
            dataset.node_features,
            dataset.command_ids,
            dataset.command_mask,
        )
        initial_loss = float(
            (
                F.cross_entropy(initial_logits, dataset.target_index)
                + F.smooth_l1_loss(
                    initial_points[torch.arange(len(dataset.target_index)), dataset.target_index],
                    dataset.target_point,
                )
            ).item()
        )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, points = model(dataset.node_features, dataset.command_ids, dataset.command_mask)
        selected_points = points[torch.arange(len(dataset.target_index)), dataset.target_index]
        loss = F.cross_entropy(logits, dataset.target_index) + F.smooth_l1_loss(
            selected_points, dataset.target_point
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(dataset.node_features, dataset.command_ids, dataset.command_mask)
        final_loss = float((F.cross_entropy(logits, dataset.target_index)).item())
        final_accuracy = float((logits.argmax(dim=-1) == dataset.target_index).float().mean())
    return TinyTrainingResult(model, initial_loss, final_loss, final_accuracy, steps)
