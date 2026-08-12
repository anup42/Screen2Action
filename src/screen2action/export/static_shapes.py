"""Fixed-shape contracts for separately deployable neural partitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExportShapeContract:
    """Maximum tensor dimensions that a runtime host preallocates.

    ``max_edges`` is a host validation limit. Exported graph tensors use a
    dense ``[relation,max_nodes,max_nodes]`` masks so tensor ranks never depend
    on the number of detected edges and overlapping relation types survive.
    """

    max_nodes: int
    max_edges: int
    max_command_tokens: int
    top_k: int
    crop_tokens: int
    embedding_dim: int
    batch_size: int = 1
    node_text_tokens: int = 16
    icon_class_count: int = 87
    visual_feature_dim: int = 16
    geometry_dim: int = 12
    crop_size: int = 64
    semantic_crop_size: int = 64
    visual_batch_size: int = 8
    action_count: int = 4

    def __post_init__(self) -> None:
        values = asdict(self)
        if min(values.values()) <= 0:
            raise ValueError("export shape dimensions must be positive")
        if self.top_k > self.max_nodes:
            raise ValueError("export top_k cannot exceed max_nodes")
        if self.max_edges > 3 * self.max_nodes * self.max_nodes:
            raise ValueError("export max_edges exceeds the dense graph capacity")
        if self.embedding_dim % 8:
            raise ValueError("export embedding_dim must be divisible by eight")
        if self.action_count != 4:
            raise ValueError("the current action contract requires exactly four actions")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe contract document."""

        return {
            "schema_version": 1,
            "box_format": "normalized_xyxy",
            "dense_relation_layout": "relation_source_destination",
            **asdict(self),
        }
