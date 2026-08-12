"""Host-side validation before fixed neural graph execution."""

from __future__ import annotations

import torch

from screen2action.export.static_shapes import ExportShapeContract


def validate_fixed_graph_inputs(
    contract: ExportShapeContract,
    relation_mask: torch.Tensor,
    relative_geometry: torch.Tensor,
    valid_nodes: torch.Tensor,
) -> int:
    """Validate shape, masks, and typed-edge capacity without truncation."""

    batch = contract.batch_size
    nodes = contract.max_nodes
    if relation_mask.shape != (batch, 3, nodes, nodes) or relation_mask.dtype is not torch.bool:
        raise ValueError("relation_mask must be boolean with shape [B,3,N,N]")
    if relative_geometry.shape != (batch, 3, nodes, nodes, contract.geometry_dim):
        raise ValueError("relative_geometry must have shape [B,3,N,N,G]")
    if not relative_geometry.is_floating_point():
        raise ValueError("relative_geometry must use a floating dtype")
    if not bool(torch.isfinite(relative_geometry).all()):
        raise ValueError("relative_geometry must be finite")
    if valid_nodes.shape != (batch, nodes) or valid_nodes.dtype is not torch.bool:
        raise ValueError("valid_nodes must be boolean with shape [B,N]")
    valid_pairs = valid_nodes.unsqueeze(1).unsqueeze(3) & valid_nodes.unsqueeze(1).unsqueeze(2)
    invalid_edges = relation_mask & ~valid_pairs
    if bool(invalid_edges.any()):
        raise ValueError("relation_mask references a padded node")
    edge_counts = relation_mask.sum(dim=(1, 2, 3))
    if bool((edge_counts > contract.max_edges).any()):
        raise ValueError("typed edge count exceeds export max_edges")
    return int(edge_counts.max())
