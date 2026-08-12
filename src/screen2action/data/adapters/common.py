"""Shared, deterministic conversion helpers for public source adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from screen2action.data.adapters.base import (
    AdapterAudit,
    AdapterReject,
    CanonicalExample,
)
from screen2action.data.schema import (
    ActionType,
    Box,
    CommandRecord,
    ElementAnnotation,
    NodeType,
    Point,
    RecordProvenance,
    ScreenRecord,
)

_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


class ConversionRejected(ValueError):
    """Structured source conversion failure."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code


def load_json_records(path: Path) -> list[Mapping[str, object]]:
    """Load a fixture or source JSON/JSONL file without schema guessing."""

    if path.suffix.casefold() == ".jsonl":
        values = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            for key in ("records", "items", "episodes", "rows"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    values = candidate
                    break
            else:
                values = [payload]
        else:
            raise ValueError(f"JSON source must contain a mapping or list: {path}")
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"all source records must be mappings: {path}")
    return values


def extract_numbers(value: object, *, expected: int) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        if expected == 2 and {"x", "y"} <= value.keys():
            return (float(value["x"]), float(value["y"]))
        for key in ("related", "absolute"):
            if key in value:
                values = value[key]
                if isinstance(values, list) and values:
                    values = values[0]
                return extract_numbers(values, expected=expected)
    if isinstance(value, str):
        values = tuple(float(match) for match in _NUMBER.findall(value))
    elif isinstance(value, Iterable):
        values = tuple(float(str(item)) for item in value)
    else:
        values = ()
    if len(values) != expected:
        raise ConversionRejected(
            "invalid_geometry", f"expected {expected} coordinates, got {value!r}"
        )
    return tuple(values)


def normalize_point(value: object, width: int, height: int) -> Point:
    x, y = extract_numbers(value, expected=2)
    if max(abs(x), abs(y)) > 1.0:
        x, y = x / width, y / height
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ConversionRejected("point_out_of_bounds", f"point {(x, y)} is outside the screen")
    return (x, y)


def normalize_box(
    value: object,
    width: int,
    height: int,
    *,
    mode: str = "xyxy",
) -> Box:
    first, second, third, fourth = extract_numbers(value, expected=4)
    if mode == "xywh":
        first, second, third, fourth = first, second, first + third, second + fourth
    elif mode != "xyxy":
        raise ValueError(f"unsupported source box mode {mode!r}")
    if max(abs(first), abs(second), abs(third), abs(fourth)) > 1.0:
        first, third = first / width, third / width
        second, fourth = second / height, fourth / height
    box = (first, second, third, fourth)
    if any(value < 0.0 or value > 1.0 for value in box):
        raise ConversionRejected("box_out_of_bounds", f"box {box} is outside the screen")
    if first > third or second > fourth:
        raise ConversionRejected("box_not_xyxy", f"box {box} is not ordered xyxy")
    return box


def canonical_app_identity(
    raw_id: object,
    *,
    source: str,
    platform: str,
    domain: object | None = None,
) -> tuple[str, str, str, str]:
    raw = str(raw_id or "").strip()
    raw_domain = str(domain or "").strip()
    if not raw_domain and platform.casefold() == "web" and raw:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        raw_domain = parsed.hostname or ""
    canonical_domain = raw_domain.casefold().removeprefix("www.")
    canonical = raw.casefold().strip().replace(" ", "_")
    if platform.casefold() == "web" and canonical_domain:
        canonical = canonical_domain
    if not canonical:
        canonical = f"{source}:unknown_app"
    return canonical, raw, canonical_domain, raw_domain


def source_node_type(value: object) -> NodeType:
    label = str(value or "").casefold().replace("-", "_").replace(" ", "_")
    if any(token in label for token in ("text", "label", "heading", "paragraph")):
        return NodeType.TEXT
    if "icon" in label:
        return NodeType.ICON
    if any(token in label for token in ("input", "field", "textbox", "edittext")):
        return NodeType.INPUT
    if any(token in label for token in ("image", "photo", "picture")):
        return NodeType.IMAGE
    if any(token in label for token in ("container", "list", "scroll", "group", "layout")):
        return NodeType.CONTAINER
    if any(token in label for token in ("button", "link", "tab", "checkbox", "radio", "control")):
        return NodeType.CONTROL
    return NodeType.OTHER


def map_action(value: object) -> ActionType:
    label = str(value or "").casefold().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "tap": ActionType.CLICK,
        "click": ActionType.CLICK,
        "press": ActionType.CLICK,
        "long_press": ActionType.LONG_PRESS,
        "long_tap": ActionType.LONG_PRESS,
        "swipe": ActionType.SCROLL,
        "scroll": ActionType.SCROLL,
        "drag": ActionType.DRAG,
    }
    if label in aliases:
        return aliases[label]
    raise ConversionRejected("unsupported_action", f"unsupported action {value!r}")


def canonical_split(value: object, *, evaluation_only: bool = False) -> str:
    if evaluation_only:
        return "test"
    label = str(value or "train").casefold().strip()
    aliases = {"training": "train", "validation": "val", "valid": "val", "dev": "val"}
    label = aliases.get(label, label)
    if label not in {"train", "val", "test"}:
        raise ConversionRejected("invalid_split", f"unsupported split {value!r}")
    return label


def decode_image_bytes(record: Mapping[str, object]) -> tuple[bytes | None, str]:
    for key in ("image_bytes_base64", "base64", "image"):
        value = record.get(key)
        if isinstance(value, Mapping) and "bytes" in value:
            value = value["bytes"]
        if isinstance(value, Mapping) and "__bytes_base64__" in value:
            value = value["__bytes_base64__"]
        if isinstance(value, bytes | bytearray | memoryview):
            payload = bytes(value)
            break
        if isinstance(value, str) and key != "image":
            try:
                payload = base64.b64decode(value, validate=True)
                break
            except ValueError:
                continue
    else:
        return None, str(record.get("image_format", "png")).casefold()
    image_format = "png" if payload.startswith(b"\x89PNG") else "jpeg"
    return payload, image_format


def record_dimensions(record: Mapping[str, object]) -> tuple[int, int]:
    size = record.get("image_size") or record.get("resolution") or record.get("device_dim")
    if isinstance(size, Mapping):
        width, height = int(float(str(size["width"]))), int(float(str(size["height"])))
    elif isinstance(size, Iterable) and not isinstance(size, str | bytes):
        width, height = (int(float(str(value))) for value in size)
    else:
        width = int(float(str(record.get("width", 0))))
        height = int(float(str(record.get("height", 0))))
    if width <= 0 or height <= 0:
        raise ConversionRejected("missing_dimensions", "positive width and height are required")
    return width, height


def smallest_containing_element(
    point: Point,
    elements: Iterable[ElementAnnotation],
) -> ElementAnnotation | None:
    x, y = point
    containing = [
        element
        for element in elements
        if element.box_xyxy_norm[0] <= x <= element.box_xyxy_norm[2]
        and element.box_xyxy_norm[1] <= y <= element.box_xyxy_norm[3]
    ]
    return min(
        containing,
        key=lambda item: (
            (item.box_xyxy_norm[2] - item.box_xyxy_norm[0])
            * (item.box_xyxy_norm[3] - item.box_xyxy_norm[1]),
            item.element_id,
        ),
        default=None,
    )


class MaterializedPublicAdapter:
    """Shared public adapter implementation with exact conversion audits."""

    source_name = "unknown"

    def __init__(
        self,
        records: Iterable[Mapping[str, object]],
        *,
        revision: str,
        license_acknowledgement_id: str,
    ) -> None:
        self.revision = revision
        self.license_acknowledgement_id = license_acknowledgement_id
        examples: list[CanonicalExample] = []
        rejects: list[AdapterReject] = []
        self._conversion_rejects = rejects
        for index, record in enumerate(records):
            item_id = str(record.get("id") or record.get("uid") or record.get("image_id") or index)
            try:
                examples.extend(self.convert_record(record, source_item_id=item_id))
            except ConversionRejected as error:
                rejects.append(
                    AdapterReject(self.source_name, item_id, error.reason_code, str(error))
                )
            except (KeyError, TypeError, ValueError) as error:
                rejects.append(
                    AdapterReject(self.source_name, item_id, "invalid_source_record", str(error))
                )
        self._examples = tuple(examples)
        self._rejects = tuple(rejects)
        del self._conversion_rejects

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        revision: str,
        license_acknowledgement_id: str,
    ) -> MaterializedPublicAdapter:
        return cls(
            load_json_records(path),
            revision=revision,
            license_acknowledgement_id=license_acknowledgement_id,
        )

    def provenance(
        self,
        source_item_id: str,
        *,
        screen_sha256: str = "",
        image_format: str = "",
        app_id_canonical: str = "",
        app_id_raw: str = "",
        domain_id_canonical: str = "",
        domain_id_raw: str = "",
        platform: str = "unknown",
        split_origin: str = "source",
        annotation_source: str = "source",
        annotation_confidence: float | None = 1.0,
        label_masks: Mapping[str, bool] | None = None,
        target_match_method: str = "unknown",
        reference_match_method: str = "none",
        action_trace_id: str = "",
        action_step_id: str = "",
        synthetic_parent_id: str = "",
        transformation_provenance: Mapping[str, str] | None = None,
    ) -> RecordProvenance:
        return RecordProvenance(
            source_dataset=self.source_name,
            source_item_id=source_item_id,
            source_revision=self.revision,
            source_license_ack_id=self.license_acknowledgement_id,
            screen_sha256=screen_sha256,
            image_format=image_format,
            app_id_canonical=app_id_canonical,
            app_id_raw=app_id_raw,
            domain_id_canonical=domain_id_canonical,
            domain_id_raw=domain_id_raw,
            platform=platform,
            split_origin=split_origin,
            annotation_source=annotation_source,
            annotation_confidence=annotation_confidence,
            label_masks=label_masks or {},
            target_match_method=target_match_method,
            reference_match_method=reference_match_method,
            action_trace_id=action_trace_id,
            action_step_id=action_step_id,
            synthetic_parent_id=synthetic_parent_id,
            transformation_provenance=transformation_provenance or {},
        )

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        raise NotImplementedError

    def reject_item(self, source_item_id: str, reason_code: str, detail: str = "") -> None:
        """Record a rejected sub-item while retaining other valid steps."""

        self._conversion_rejects.append(
            AdapterReject(self.source_name, source_item_id, reason_code, detail)
        )

    def examples(self) -> tuple[CanonicalExample, ...]:
        return self._examples

    def screens(self) -> tuple[ScreenRecord, ...]:
        return tuple(example.screen for example in self._examples)

    def commands(self) -> tuple[CommandRecord, ...]:
        return tuple(command for example in self._examples for command in example.commands)

    def rejects(self) -> tuple[AdapterReject, ...]:
        return self._rejects

    def audit(self) -> AdapterAudit:
        action_counts = Counter(command.action_type.value for command in self.commands())
        rejection_counts = Counter(reject.reason_code for reject in self._rejects)
        return AdapterAudit(
            source=self.source_name,
            accepted_screens=len(self._examples),
            rejected_screens=len(self._rejects),
            accepted_commands=len(self.commands()),
            rejected_commands=sum(rejection_counts.values()),
            accepted_elements=sum(len(example.screen.elements) for example in self._examples),
            action_counts=dict(sorted(action_counts.items())),
            rejection_counts=dict(sorted(rejection_counts.items())),
        )


def build_screen(
    adapter: MaterializedPublicAdapter,
    record: Mapping[str, object],
    *,
    source_item_id: str,
    elements: tuple[ElementAnnotation, ...],
    platform: str,
    split: str,
    app_raw: object,
    domain_raw: object | None = None,
    image_path: str | None = None,
) -> tuple[ScreenRecord, bytes | None, str]:
    width, height = record_dimensions(record)
    canonical_app, raw_app, canonical_domain, raw_domain = canonical_app_identity(
        app_raw, source=adapter.source_name, platform=platform, domain=domain_raw
    )
    image_bytes, image_format = decode_image_bytes(record)
    digest = (
        hashlib.sha256(image_bytes).hexdigest()
        if image_bytes is not None
        else str(record.get("screen_sha256", ""))
    )
    path = image_path or str(record.get("image_path") or record.get("image_id") or "")
    provenance = adapter.provenance(
        source_item_id,
        screen_sha256=digest,
        image_format=image_format,
        app_id_canonical=canonical_app,
        app_id_raw=raw_app,
        domain_id_canonical=canonical_domain,
        domain_id_raw=raw_domain,
        platform=platform,
        split_origin=str(record.get("split_origin", "source")),
    )
    screen = ScreenRecord(
        screen_id=f"{adapter.source_name}:{source_item_id}",
        app_id=canonical_app,
        image_path=path,
        width=width,
        height=height,
        split=split,
        source=adapter.source_name,
        elements=elements,
        provenance=provenance,
    )
    return screen, image_bytes, image_format


def build_elements(
    adapter: MaterializedPublicAdapter,
    raw_elements: object,
    *,
    source_item_id: str,
    width: int,
    height: int,
    default_type: NodeType = NodeType.OTHER,
    box_key: str = "bbox",
    box_mode: str = "xyxy",
    annotation_source: str = "source",
) -> tuple[ElementAnnotation, ...]:
    if raw_elements is None:
        return ()
    if not isinstance(raw_elements, Iterable) or isinstance(raw_elements, str | bytes | Mapping):
        raise ConversionRejected("invalid_elements", "elements must be a list")
    result: list[ElementAnnotation] = []
    for index, raw in enumerate(raw_elements):
        if not isinstance(raw, Mapping):
            adapter.reject_item(
                f"{source_item_id}:element:{index}", "invalid_element", "element is not a mapping"
            )
            continue
        element_id = str(raw.get("id") if raw.get("id") is not None else index)
        raw_box = raw.get(box_key) or raw.get("box") or raw.get("position") or raw.get("rect")
        if raw_box is None and {"xmin", "ymin", "xmax", "ymax"} <= raw.keys():
            raw_box = (raw["xmin"], raw["ymin"], raw["xmax"], raw["ymax"])
        if isinstance(raw_box, Mapping):
            if {"x", "y", "width", "height"} <= raw_box.keys():
                raw_box = (
                    raw_box["x"],
                    raw_box["y"],
                    raw_box["width"],
                    raw_box["height"],
                )
                selected_mode = "xywh"
            elif {"left", "top", "right", "bottom"} <= raw_box.keys():
                raw_box = (
                    raw_box["left"],
                    raw_box["top"],
                    raw_box["right"],
                    raw_box["bottom"],
                )
                selected_mode = "xyxy"
            else:
                adapter.reject_item(
                    f"{source_item_id}:element:{element_id}",
                    "invalid_element_box",
                    f"unsupported box mapping keys: {sorted(raw_box)}",
                )
                continue
        else:
            selected_mode = box_mode
        try:
            box = normalize_box(raw_box, width, height, mode=selected_mode)
        except ConversionRejected as error:
            adapter.reject_item(
                f"{source_item_id}:element:{element_id}", error.reason_code, str(error)
            )
            continue
        raw_type = (
            raw.get("ui_type") or raw.get("type") or raw.get("componentLabel") or raw.get("label")
        )
        node_type = source_node_type(raw_type) if raw_type else default_type
        text_value = raw.get("text") or raw.get("name") or raw.get("functionality")
        if not text_value and isinstance(raw.get("xml_desc"), list):
            text_value = next((value for value in raw["xml_desc"] if value), None)
        masks = {
            "box": True,
            "node_type": raw_type is not None,
            "text": text_value is not None,
            "icon": raw.get("iconClass") is not None,
            "actionability": raw.get("clickable") is not None,
        }
        actionability = (
            bool(raw.get("clickable")) if raw.get("clickable") is not None else None,
            None,
            bool(raw.get("scrollable")) if raw.get("scrollable") is not None else None,
            bool(raw.get("long_clickable")) if raw.get("long_clickable") is not None else None,
        )
        provenance = adapter.provenance(
            f"{source_item_id}:element:{element_id}",
            annotation_source=annotation_source,
            annotation_confidence=float(raw.get("confidence", 1.0)),
            label_masks=masks,
        )
        result.append(
            ElementAnnotation(
                element_id=element_id,
                node_type=node_type,
                box_xyxy_norm=box,
                text=str(text_value) if text_value is not None else None,
                actionability_labels=actionability,
                icon_class_id=(
                    int(raw["icon_class_id"]) if raw.get("icon_class_id") is not None else None
                ),
                parent_element_id=(
                    str(raw["parent_id"]) if raw.get("parent_id") is not None else None
                ),
                metadata={
                    "raw_type": str(raw_type or ""),
                    "source_element_id": element_id,
                },
                annotation_source=annotation_source,
                annotation_confidence=float(raw.get("confidence", 1.0)),
                label_masks=masks,
                provenance=provenance,
            )
        )
    return tuple(result)


def with_screen_digest(
    example: CanonicalExample,
    digest: str,
    image_format: str,
    image_path: str,
) -> CanonicalExample:
    """Return an example updated after content-addressed image materialization."""

    provenance = replace(
        example.screen.provenance,
        screen_sha256=digest,
        image_format=image_format,
    )
    elements = tuple(
        replace(
            element,
            provenance=replace(
                element.provenance,
                screen_sha256=digest,
                image_format=image_format,
            ),
        )
        for element in example.screen.elements
    )
    screen = replace(
        example.screen,
        image_path=image_path,
        elements=elements,
        provenance=provenance,
    )
    commands = tuple(
        replace(
            command,
            provenance=replace(
                command.provenance,
                screen_sha256=digest,
                image_format=image_format,
            ),
        )
        for command in example.commands
    )
    return replace(example, screen=screen, commands=commands, image_format=image_format)
