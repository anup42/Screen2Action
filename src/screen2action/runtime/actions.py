"""Factorized action decoding interfaces beyond the click MVP."""

from __future__ import annotations

from dataclasses import dataclass

from screen2action.data.schema import ActionType, Point


@dataclass(frozen=True, slots=True)
class DecodedAction:
    """Action parameters in normalized screen coordinates or bounded deltas."""

    action_type: ActionType
    point: Point | None
    destination: Point | None
    displacement: tuple[float, float] | None
    duration: float | None


def decode_action(
    action_type: ActionType,
    point: Point,
    parameters: tuple[float, ...] = (),
) -> DecodedAction:
    """Decode click, long-press, scroll, and drag parameter contracts."""

    action_type = ActionType(action_type)
    if action_type in {ActionType.CLICK, ActionType.LONG_PRESS}:
        duration = parameters[0] if action_type is ActionType.LONG_PRESS and parameters else None
        return DecodedAction(action_type, point, None, None, duration)
    if action_type is ActionType.SCROLL:
        if len(parameters) < 2:
            raise ValueError("scroll decoding requires dx and dy")
        return DecodedAction(action_type, point, None, (parameters[0], parameters[1]), None)
    if action_type is ActionType.DRAG:
        if len(parameters) < 3:
            raise ValueError("drag decoding requires destination x/y and duration")
        return DecodedAction(
            action_type, point, (parameters[0], parameters[1]), None, parameters[2]
        )
    raise ValueError(f"unsupported action type {action_type}")
