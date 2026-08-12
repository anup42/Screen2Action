"""Genuinely grouped candidate-conditioned cross-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GroundingTensorOutput:
    """Tensor predictions from candidate-conditioned grounding."""

    candidate_logits: torch.Tensor
    point_local: torch.Tensor
    action_type_logits: torch.Tensor
    action_parameter: torch.Tensor
    confidence_logits: torch.Tensor
    command_states: torch.Tensor
    candidate_states: torch.Tensor


class SparseCandidateGrounder(nn.Module):
    """Two-stream sparse attention with one isolated batch group per candidate.

    Command tokens attend to the flattened candidate groups. Candidate groups
    are then processed as independent `[batch * K]` items whose key/value
    context contains only the shared command stream and that candidate's own
    node/crop tokens. No dense cross-candidate attention matrix is built.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        heads: int = 4,
        blocks: int = 2,
        action_count: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0 or min(embedding_dim, heads, blocks) <= 0:
            raise ValueError("invalid sparse-grounder dimensions")
        self.embedding_dim = embedding_dim
        self.blocks = blocks
        self.command_attn = nn.ModuleList(
            nn.MultiheadAttention(
                embedding_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(blocks)
        )
        self.local_attn = nn.ModuleList(
            nn.MultiheadAttention(
                embedding_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(blocks)
        )
        self.command_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(blocks))
        self.candidate_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(blocks))
        self.command_output = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(blocks)
        )
        self.candidate_output = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(blocks)
        )
        self.candidate_head = nn.Linear(embedding_dim, 1)
        self.point_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 2)
        )
        self.action_head = nn.Linear(embedding_dim, action_count)
        self.action_parameter_head = nn.Linear(embedding_dim, 4)
        self.confidence_head = nn.Linear(embedding_dim, 1)

    def _command_update(
        self,
        command_states: torch.Tensor,
        group_states: torch.Tensor,
        command_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        batch, candidates, group_length, embedding = group_states.shape
        flattened = group_states.reshape(batch, candidates * group_length, embedding)
        key_padding = (
            (~candidate_mask.bool())
            .unsqueeze(-1)
            .expand(-1, -1, group_length)
            .reshape(batch, candidates * group_length)
        )
        attended, _ = self.command_attn[block_index](
            command_states,
            flattened,
            flattened,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        updated = self.command_norm[block_index](
            command_states + self.command_output[block_index](attended)
        )
        return cast(torch.Tensor, updated * command_mask.unsqueeze(-1).to(updated.dtype))

    def local_candidate_update(
        self,
        command_states: torch.Tensor,
        group_states: torch.Tensor,
        command_mask: torch.Tensor,
        block_index: int = 0,
    ) -> torch.Tensor:
        """Update each group with a fixed shared command stream for isolation tests."""

        batch, candidates, group_length, embedding = group_states.shape
        flattened_groups = group_states.reshape(batch * candidates, group_length, embedding)
        repeated_command = command_states.unsqueeze(1).expand(-1, candidates, -1, -1)
        repeated_command = repeated_command.reshape(
            batch * candidates, command_states.shape[1], embedding
        )
        repeated_mask = (~command_mask.bool()).unsqueeze(1).expand(-1, candidates, -1)
        repeated_mask = repeated_mask.reshape(batch * candidates, command_states.shape[1])
        local_context = torch.cat((repeated_command, flattened_groups), dim=1)
        context_mask = torch.cat(
            (
                repeated_mask,
                torch.zeros(
                    (batch * candidates, group_length), dtype=torch.bool, device=group_states.device
                ),
            ),
            dim=1,
        )
        attended, _ = self.local_attn[block_index](
            flattened_groups,
            local_context,
            local_context,
            key_padding_mask=context_mask,
            need_weights=False,
        )
        updated = self.candidate_norm[block_index](
            flattened_groups + self.candidate_output[block_index](attended)
        )
        return cast(torch.Tensor, updated.reshape(batch, candidates, group_length, embedding))

    def forward(
        self,
        command_states: torch.Tensor,
        candidate_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        *,
        command_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> GroundingTensorOutput:
        """Run two sparse conditioning blocks and predict click/action outputs."""

        if command_states.ndim != 3 or candidate_tokens.ndim != 4 or node_tokens.ndim != 3:
            raise ValueError("command/candidate/node tensors have invalid ranks")
        batch, candidates, _, embedding = candidate_tokens.shape
        if node_tokens.shape != (batch, candidates, embedding):
            raise ValueError("node_tokens must have shape [B, K, D]")
        if command_mask is None:
            command_mask = torch.ones(
                command_states.shape[:2], dtype=torch.bool, device=command_states.device
            )
        if candidate_mask is None:
            candidate_mask = torch.ones(
                (batch, candidates), dtype=torch.bool, device=command_states.device
            )
        group_states = torch.cat((node_tokens.unsqueeze(2), candidate_tokens), dim=2)
        for block_index in range(self.blocks):
            command_states = self._command_update(
                command_states,
                group_states,
                command_mask,
                candidate_mask,
                block_index,
            )
            group_states = self.local_candidate_update(
                command_states,
                group_states,
                command_mask,
                block_index,
            )
        candidate_states = group_states[:, :, 0]
        candidate_logits = self.candidate_head(candidate_states).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask.bool(), -1e9)
        point_local = torch.sigmoid(self.point_head(candidate_states))
        action_type_logits = self.action_head(candidate_states)
        action_parameter = torch.tanh(self.action_parameter_head(candidate_states))
        confidence_logits = self.confidence_head(candidate_states).squeeze(-1)
        return GroundingTensorOutput(
            candidate_logits=candidate_logits,
            point_local=point_local,
            action_type_logits=action_type_logits,
            action_parameter=action_parameter,
            confidence_logits=confidence_logits,
            command_states=command_states,
            candidate_states=candidate_states,
        )
