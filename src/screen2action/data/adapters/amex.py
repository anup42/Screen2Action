"""Official AMEX multi-step mobile action adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from screen2action.data.adapters.base import CanonicalExample
from screen2action.data.adapters.common import (
    ConversionRejected,
    MaterializedPublicAdapter,
    build_elements,
    build_screen,
    canonical_split,
    map_action,
    normalize_box,
    normalize_point,
    record_dimensions,
    smallest_containing_element,
)
from screen2action.data.schema import ActionType, CommandRecord, NodeType, PointSource


class AmexAdapter(MaterializedPublicAdapter):
    """Emit one command-state-action example per supported AMEX trace step."""

    source_name = "amex"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        raw_steps = record.get("steps")
        if not isinstance(raw_steps, Iterable) or isinstance(raw_steps, str | bytes | Mapping):
            raise ConversionRejected("missing_steps", "AMEX episode must contain a steps list")
        episode_id = str(record.get("episode_id") or source_item_id)
        instruction = str(record.get("instruction") or "").strip()
        if not instruction:
            raise ConversionRejected("missing_command", "AMEX episode instruction is empty")
        results: list[CanonicalExample] = []
        for index, raw_step in enumerate(raw_steps):
            step_item_id = f"{episode_id}:step:{index}"
            if not isinstance(raw_step, Mapping):
                self.reject_item(step_item_id, "invalid_step", "step is not a mapping")
                continue
            step_id = str(raw_step.get("step_id") or index)
            step_item_id = f"{episode_id}:step:{step_id}"
            try:
                action_type = map_action(raw_step.get("action"))
            except ConversionRejected as error:
                self.reject_item(step_item_id, error.reason_code, str(error))
                continue
            step_record = dict(record)
            step_record.update(raw_step)
            step_record["id"] = step_item_id
            width, height = record_dimensions(step_record)
            clickable = raw_step.get("clickable_elements") or raw_step.get("elements") or ()
            scrollable = raw_step.get("scrollable_elements") or ()
            elements = list(
                build_elements(
                    self,
                    clickable,
                    source_item_id=step_item_id,
                    width=width,
                    height=height,
                    default_type=NodeType.CONTROL,
                    annotation_source="amex_clickable_element",
                )
            )
            elements.extend(
                build_elements(
                    self,
                    scrollable,
                    source_item_id=f"{step_item_id}:scrollable",
                    width=width,
                    height=height,
                    default_type=NodeType.CONTAINER,
                    annotation_source="amex_scrollable_element",
                )
            )
            screen, image_bytes, image_format = build_screen(
                self,
                step_record,
                source_item_id=step_item_id,
                elements=tuple(elements),
                platform="android",
                split=canonical_split(record.get("split", "train")),
                app_raw=raw_step.get("package_name") or record.get("package_name"),
                image_path=str(raw_step.get("image_path") or ""),
            )
            point = normalize_point(raw_step.get("touch_coord"), width, height)
            target = smallest_containing_element(point, elements)
            interest_region = raw_step.get("interest_region")
            if target is not None:
                target_box = target.box_xyxy_norm
                target_match = "smallest_source_element_containing_true_point"
            elif action_type in {ActionType.SCROLL, ActionType.DRAG}:
                target_box = (0.0, 0.0, 1.0, 1.0)
                target_match = "full_screen_gesture_region"
            elif interest_region and interest_region != [[0, 0], [0, 0]]:
                flattened = tuple(value for pair in interest_region for value in pair)
                target_box = normalize_box(flattened, width, height)
                target_match = "source_interest_region"
            else:
                self.reject_item(
                    step_item_id,
                    "target_point_unmatched",
                    "AMEX action point has no containing element or interest region",
                )
                continue
            action_parameters: dict[str, float] = {}
            if action_type in {ActionType.SCROLL, ActionType.DRAG}:
                destination = normalize_point(raw_step.get("lift_coord"), width, height)
                action_parameters = {
                    "dx": destination[0] - point[0],
                    "dy": destination[1] - point[1],
                    "destination_x": destination[0],
                    "destination_y": destination[1],
                }
            masks = {
                "target": True,
                "point": True,
                "action": True,
                "parameters": bool(action_parameters),
            }
            provenance = replace(
                screen.provenance,
                target_match_method=target_match,
                action_trace_id=episode_id,
                action_step_id=step_id,
                label_masks=masks,
            )
            command = CommandRecord(
                command_id=f"{screen.screen_id}:command",
                screen_id=screen.screen_id,
                text=str(raw_step.get("step_instruction") or instruction),
                action_type=action_type,
                target_box_xyxy_norm=target_box,
                target_point_xy_norm=point,
                target_point_source=PointSource.TRUE,
                action_parameters=action_parameters,
                label_masks=masks,
                provenance=provenance,
            )
            results.append(CanonicalExample(screen, (command,), image_bytes, image_format))
        return tuple(results)
