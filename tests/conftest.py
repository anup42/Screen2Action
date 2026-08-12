from __future__ import annotations

import pytest

from screen2action.data.schema import NodeRecord, NodeType


@pytest.fixture
def make_node():
    def factory(
        node_id: int,
        box: tuple[float, float, float, float],
        *,
        node_type: NodeType = NodeType.CONTROL,
        confidence: float = 0.5,
        text_token_ids: tuple[int, ...] = (),
        mandatory: bool = False,
    ) -> NodeRecord:
        return NodeRecord(
            node_id=node_id,
            node_type=node_type,
            box_xyxy_norm=box,
            detector_confidence=confidence,
            text_token_ids=text_token_ids,
            mandatory=mandatory,
            retention_score=confidence,
        )

    return factory
