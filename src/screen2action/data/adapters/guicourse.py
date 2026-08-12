"""Official GUIAct and GUIEnv source adapters."""

from __future__ import annotations

from collections.abc import Mapping
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


class GuiActAdapter(MaterializedPublicAdapter):
    """Convert official GUIAct web-single or smartphone rows."""

    source_name = "guicourse_guiact"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        subset = str(record.get("subset", "smartphone"))
        if subset not in {"smartphone", "web_single", "web-single"}:
            raise ConversionRejected(
                "unsupported_subset", "only GUIAct smartphone and web-single map safely"
            )
        width, height = record_dimensions(record)
        elements = build_elements(
            self,
            record.get("elements"),
            source_item_id=source_item_id,
            width=width,
            height=height,
            box_key="position",
            box_mode="xywh",
        )
        platform = "android" if subset == "smartphone" else "web"
        screen, image_bytes, image_format = build_screen(
            self,
            record,
            source_item_id=source_item_id,
            elements=elements,
            platform=str(record.get("platform", platform)),
            split=canonical_split(record.get("split", "train")),
            app_raw=record.get("package_name") or record.get("app_id") or record.get("domain"),
            domain_raw=record.get("domain"),
        )
        action = record.get("actions_label") or record.get("action")
        if not isinstance(action, Mapping):
            raise ConversionRejected("missing_action", "GUIAct row has no action mapping")
        action_type = map_action(action.get("name") or action.get("type"))
        if action_type not in {ActionType.CLICK, ActionType.LONG_PRESS}:
            raise ConversionRejected(
                "unsupported_action", f"GUIAct action {action_type.value!r} is not safely mapped"
            )
        point = normalize_point(action.get("point"), width, height)
        target = smallest_containing_element(point, elements)
        if target is None:
            raw_target = action.get("target_box") or record.get("target_box")
            if raw_target is None:
                raise ConversionRejected(
                    "target_point_unmatched", "action point is not contained by a source element"
                )
            target_box = normalize_box(raw_target, width, height)
            target_match = "source_target_box"
        else:
            target_box = target.box_xyxy_norm
            target_match = "smallest_source_element_containing_true_point"
        command_text = str(record.get("question") or record.get("instruction") or "").strip()
        if not command_text:
            raise ConversionRejected("missing_command", "GUIAct command text is empty")
        provenance = replace(
            screen.provenance,
            target_match_method=target_match,
            action_trace_id=str(record.get("episode_id") or source_item_id.split("_step_")[0]),
            action_step_id=str(record.get("step_id") or source_item_id),
            label_masks={"target": True, "point": True, "action": True, "parameters": False},
        )
        command = CommandRecord(
            command_id=f"{screen.screen_id}:command",
            screen_id=screen.screen_id,
            text=command_text,
            action_type=action_type,
            target_box_xyxy_norm=target_box,
            target_point_xy_norm=point,
            target_point_source=PointSource.TRUE,
            label_masks=provenance.label_masks,
            provenance=provenance,
        )
        return (CanonicalExample(screen, (command,), image_bytes, image_format),)


class GuiEnvAdapter(MaterializedPublicAdapter):
    """Convert GUIEnv OCR/grounding rows without fabricating action labels."""

    source_name = "guicourse_guienv"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        width, height = record_dimensions(record)
        task_type = str(record.get("task_type", ""))
        if task_type == "text2bbox":
            target_box = normalize_box(record.get("answer"), width, height)
            target_text = str(record.get("question", "")).strip()
            command_text = (
                f"locate {target_text}" if target_text else "locate the described element"
            )
        elif task_type == "bbox2text":
            target_box = normalize_box(record.get("question"), width, height)
            answer = record.get("answer")
            target_text = str(answer.get("text") if isinstance(answer, Mapping) else answer)
            command_text = f"read the text in {target_text}" if target_text else "read this region"
        else:
            raise ConversionRejected("unsupported_task", f"unsupported GUIEnv task {task_type!r}")
        element = build_elements(
            self,
            [{"id": "target", "bbox": target_box, "type": "text", "text": target_text}],
            source_item_id=source_item_id,
            width=width,
            height=height,
            default_type=NodeType.TEXT,
        )
        screen, image_bytes, image_format = build_screen(
            self,
            record,
            source_item_id=source_item_id,
            elements=element,
            platform=str(record.get("platform", "web")),
            split=canonical_split(record.get("split", "train")),
            app_raw=record.get("app_id") or record.get("domain") or record.get("image_id"),
            domain_raw=record.get("domain"),
        )
        x1, y1, x2, y2 = target_box
        pseudo_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        provenance = replace(
            screen.provenance,
            target_match_method="source_bbox",
            label_masks={"target": True, "point": False, "action": False, "parameters": False},
        )
        command = CommandRecord(
            command_id=f"{screen.screen_id}:command",
            screen_id=screen.screen_id,
            text=command_text,
            action_type=ActionType.UNKNOWN,
            target_box_xyxy_norm=target_box,
            target_point_xy_norm=pseudo_point,
            target_point_source=PointSource.PSEUDO_BOX_CENTER,
            label_masks=provenance.label_masks,
            provenance=provenance,
        )
        return (CanonicalExample(screen, (command,), image_bytes, image_format),)
