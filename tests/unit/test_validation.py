from __future__ import annotations

import pytest

from screen2action.data.schema import EdgeRecord, RelationDirection, RelationType
from screen2action.ssb.validation import remap_selected_edges, validate_graph


def test_graph_validation_rejects_missing_endpoints(make_node) -> None:
    node = make_node(0, (0.0, 0.0, 0.2, 0.2))
    edge = EdgeRecord(0, 1, RelationType.PROXIMITY, RelationDirection.RIGHT)
    with pytest.raises(ValueError, match="missing endpoint"):
        validate_graph([node], [edge])


def test_selected_edge_remapping_removes_dangling_edges(make_node) -> None:
    edge = EdgeRecord(0, 1, RelationType.PROXIMITY, RelationDirection.RIGHT)
    assert remap_selected_edges([0], [edge]) == ()
