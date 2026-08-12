"""Fixed-shape export contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExportShapeContract:
    """Maximum tensor dimensions that a mobile host preallocates."""

    max_nodes: int
    max_edges: int
    max_command_tokens: int
    top_k: int
    crop_tokens: int
    embedding_dim: int

    def __post_init__(self) -> None:
        if (
            min(
                self.max_nodes,
                self.max_edges,
                self.max_command_tokens,
                self.top_k,
                self.crop_tokens,
                self.embedding_dim,
            )
            <= 0
        ):
            raise ValueError("export shape dimensions must be positive")
