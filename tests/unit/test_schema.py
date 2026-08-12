from __future__ import annotations

import pytest

from screen2action.data.schema import ActionType, CommandRecord, NodeRecord, NodeType


def test_node_schema_requires_normalized_ordered_boxes() -> None:
    with pytest.raises(ValueError, match="ordered"):
        NodeRecord(1, NodeType.TEXT, (0.8, 0.1, 0.2, 0.9))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NodeRecord(1, NodeType.TEXT, (-0.1, 0.1, 0.2, 0.9))


def test_command_schema_masks_missing_target_point() -> None:
    command = CommandRecord(
        command_id="c1",
        screen_id="s1",
        text="open settings",
        action_type=ActionType.CLICK,
        target_box_xyxy_norm=(0.1, 0.2, 0.4, 0.5),
    )
    assert command.target_point_xy_norm is None


def test_node_schema_requires_four_actionability_logits() -> None:
    with pytest.raises(ValueError, match="four"):
        NodeRecord(
            1,
            NodeType.CONTROL,
            (0.0, 0.0, 0.2, 0.2),
            actionability_logits=(0.0, 0.0, 0.0),  # type: ignore[arg-type]
        )


def test_node_schema_preserves_unknown_actionability_mask() -> None:
    node = NodeRecord(
        1,
        NodeType.CONTROL,
        (0.0, 0.0, 0.2, 0.2),
        actionability_mask=(True, False, True, False),
    )
    assert node.actionability_mask == (True, False, True, False)
