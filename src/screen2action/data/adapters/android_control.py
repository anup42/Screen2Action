"""Official AndroidControl GZIP TFRecord and action adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path

from screen2action.data.adapters.base import CanonicalExample
from screen2action.data.adapters.common import (
    ConversionRejected,
    MaterializedPublicAdapter,
    build_elements,
    build_screen,
    canonical_split,
    map_action,
    normalize_point,
    smallest_containing_element,
)
from screen2action.data.schema import ActionType, CommandRecord, NodeType, PointSource


def _decode_bytes(value: bytes) -> str:
    return value.decode("utf-8", errors="strict")


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        raise ConversionRejected("invalid_episode", f"AndroidControl {name} must be a list")
    return list(value)


def iter_android_control_tfrecords(
    paths: Iterable[Path],
    *,
    official_splits: Mapping[str, str],
) -> Iterator[Mapping[str, object]]:
    """Parse official GZIP TFRecords through the optional TensorFlow dependency."""

    try:
        import tensorflow as tf  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "AndroidControl TFRecords require: pip install -e .[android-control]"
        ) from error
    filenames = [str(Path(path)) for path in paths]
    if not filenames:
        raise ValueError("at least one AndroidControl TFRecord is required")
    dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    for serialized in dataset:
        example = tf.train.Example.FromString(bytes(serialized.numpy()))
        features = example.features.feature
        episode_id = str(features["episode_id"].int64_list.value[0])
        if episode_id not in official_splits:
            raise ValueError(f"episode {episode_id!r} is absent from the official split file")
        yield {
            "episode_id": episode_id,
            "goal": _decode_bytes(features["goal"].bytes_list.value[0]),
            "screenshots": list(features["screenshots"].bytes_list.value),
            "accessibility_trees": list(features["accessibility_trees"].bytes_list.value),
            "screenshot_widths": list(features["screenshot_widths"].int64_list.value),
            "screenshot_heights": list(features["screenshot_heights"].int64_list.value),
            "actions": [
                json.loads(_decode_bytes(value)) for value in features["actions"].bytes_list.value
            ],
            "step_instructions": [
                _decode_bytes(value) for value in features["step_instructions"].bytes_list.value
            ],
            "split": official_splits[episode_id],
        }


class AndroidControlAdapter(MaterializedPublicAdapter):
    """Convert supported AndroidControl steps and reject non-paper action types."""

    source_name = "android_control"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        episode_id = str(record.get("episode_id") or source_item_id)
        actions = record.get("actions")
        screenshots = record.get("screenshots")
        widths = record.get("screenshot_widths")
        heights = record.get("screenshot_heights")
        instructions = record.get("step_instructions")
        action_list = _sequence(actions, "actions")
        screenshot_list = _sequence(screenshots, "screenshots")
        width_list = _sequence(widths, "screenshot_widths")
        height_list = _sequence(heights, "screenshot_heights")
        instruction_list = _sequence(instructions, "step_instructions")
        if not (
            len(screenshot_list) == len(action_list) + 1
            and len(width_list) == len(screenshot_list)
            and len(height_list) == len(screenshot_list)
            and len(instruction_list) == len(action_list)
        ):
            raise ConversionRejected(
                "sequence_length_mismatch", "official sequence lengths disagree"
            )
        element_steps_list = _sequence(
            record.get("elements_by_step") or [()] * len(action_list),
            "elements_by_step",
        )
        current_app = str(record.get("app_id") or "com.google.android.apps.nexuslauncher")
        results: list[CanonicalExample] = []
        for index, raw_action in enumerate(action_list):
            step_item_id = f"{episode_id}:step:{index}"
            if not isinstance(raw_action, Mapping):
                self.reject_item(step_item_id, "invalid_action", "action is not a mapping")
                continue
            raw_action_type = str(raw_action.get("action_type", ""))
            if raw_action_type == "open_app":
                current_app = str(raw_action.get("app_name") or current_app)
                self.reject_item(
                    step_item_id,
                    "unsupported_action",
                    "open_app updates app identity but is outside the paper action taxonomy",
                )
                continue
            try:
                action_type = map_action(raw_action_type)
            except ConversionRejected as error:
                self.reject_item(step_item_id, error.reason_code, str(error))
                continue
            width = int(float(str(width_list[index])))
            height = int(float(str(height_list[index])))
            raw_elements = element_steps_list[index] if index < len(element_steps_list) else ()
            elements = build_elements(
                self,
                raw_elements,
                source_item_id=step_item_id,
                width=width,
                height=height,
                default_type=NodeType.OTHER,
                annotation_source="android_accessibility_tree",
            )
            step_record = {
                "width": width,
                "height": height,
                "image": screenshot_list[index],
                "image_format": "png",
                "split": record.get("split"),
                "app_id": current_app,
            }
            screen, image_bytes, image_format = build_screen(
                self,
                step_record,
                source_item_id=step_item_id,
                elements=elements,
                platform="android",
                split=canonical_split(record.get("split")),
                app_raw=current_app,
            )
            point = None
            point_source = PointSource.NONE
            action_parameters: dict[str, float] = {}
            if action_type in {ActionType.CLICK, ActionType.LONG_PRESS}:
                point = normalize_point((raw_action.get("x"), raw_action.get("y")), width, height)
                point_source = PointSource.TRUE
                target = smallest_containing_element(point, elements)
                target_box = target.box_xyxy_norm if target is not None else (*point, *point)
                target_match = (
                    "smallest_accessibility_element_containing_true_point"
                    if target is not None
                    else "true_point_degenerate_box"
                )
            else:
                target_box = (0.0, 0.0, 1.0, 1.0)
                target_match = "full_screen_gesture_region"
                direction = str(raw_action.get("direction", "")).casefold()
                vectors = {
                    "up": (0.0, -1.0),
                    "down": (0.0, 1.0),
                    "left": (-1.0, 0.0),
                    "right": (1.0, 0.0),
                }
                if direction not in vectors:
                    self.reject_item(step_item_id, "invalid_scroll_direction", direction)
                    continue
                action_parameters = dict(zip(("dx", "dy"), vectors[direction], strict=True))
            masks = {
                "target": target_match != "true_point_degenerate_box",
                "point": point is not None,
                "action": True,
                "parameters": bool(action_parameters),
            }
            provenance = replace(
                screen.provenance,
                split_origin="official_android_control",
                target_match_method=target_match,
                action_trace_id=episode_id,
                action_step_id=str(index),
                label_masks=masks,
                transformation_provenance={
                    "app_identity": "open_app_action_state_or_launcher_default",
                    "accessibility_tree": "preserved_by_source_loader",
                },
            )
            command = CommandRecord(
                command_id=f"{screen.screen_id}:command",
                screen_id=screen.screen_id,
                text=str(instruction_list[index] or record.get("goal") or ""),
                action_type=action_type,
                target_box_xyxy_norm=target_box,
                target_point_xy_norm=point,
                target_point_source=point_source,
                action_parameters=action_parameters,
                label_masks=masks,
                provenance=provenance,
            )
            results.append(CanonicalExample(screen, (command,), image_bytes, image_format))
        return tuple(results)
