"""Small deterministic synthetic screens for CPU integration tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from screen2action.data.schema import (
    ActionType,
    CommandRecord,
    ElementAnnotation,
    NodeType,
    ScreenRecord,
)


@dataclass(frozen=True, slots=True)
class SyntheticExample:
    """One screenshot/command pair with source element supervision."""

    screenshot: torch.Tensor
    screen: ScreenRecord
    command: CommandRecord


def _paint_box(
    image: torch.Tensor, box: tuple[float, float, float, float], color: tuple[float, float, float]
) -> None:
    height, width = image.shape[-2:]
    x1 = max(0, min(width, int(box[0] * width)))
    y1 = max(0, min(height, int(box[1] * height)))
    x2 = max(x1 + 1, min(width, int(box[2] * width)))
    y2 = max(y1 + 1, min(height, int(box[3] * height)))
    image[:, y1:y2, x1:x2] = torch.tensor(color, dtype=image.dtype).view(3, 1, 1)


def make_synthetic_examples() -> tuple[SyntheticExample, ...]:
    """Build direct, icon, ordinal, containment, proximity, and edge cases."""

    width, height = 160, 96
    screenshot = torch.full((3, height, width), 0.08, dtype=torch.float32)
    elements = (
        ElementAnnotation(
            "toolbar",
            NodeType.CONTAINER,
            (0.04, 0.04, 0.96, 0.25),
            actionability_labels=(False, False, False, False),
        ),
        ElementAnnotation(
            "search",
            NodeType.TEXT,
            (0.10, 0.09, 0.42, 0.19),
            text="Search",
            actionability_labels=(True, False, False, False),
        ),
        ElementAnnotation(
            "cart",
            NodeType.ICON,
            (0.76, 0.09, 0.90, 0.20),
            actionability_labels=(True, False, False, False),
            icon_class_id=7,
        ),
        ElementAnnotation(
            "settings_list",
            NodeType.CONTAINER,
            (0.04, 0.30, 0.96, 0.94),
            actionability_labels=(False, False, True, False),
            metadata={"is_scroll_container": "true"},
        ),
        ElementAnnotation(
            "wifi",
            NodeType.TEXT,
            (0.10, 0.38, 0.44, 0.48),
            text="Wi-Fi",
            actionability_labels=(True, False, False, False),
        ),
        ElementAnnotation(
            "bluetooth",
            NodeType.TEXT,
            (0.54, 0.38, 0.90, 0.48),
            text="Bluetooth",
            actionability_labels=(True, False, False, False),
        ),
        ElementAnnotation(
            "tiny_toggle",
            NodeType.CONTROL,
            (0.86, 0.82, 0.90, 0.87),
            actionability_labels=(True, False, False, False),
        ),
        ElementAnnotation(
            "missing_ocr_icon",
            NodeType.ICON,
            (0.10, 0.70, 0.18, 0.78),
            actionability_labels=(True, False, False, False),
        ),
    )
    for element, color in zip(
        elements,
        (
            (0.16, 0.16, 0.16),
            (0.18, 0.55, 0.90),
            (0.90, 0.40, 0.18),
            (0.12, 0.12, 0.18),
            (0.22, 0.70, 0.32),
            (0.82, 0.38, 0.24),
            (0.94, 0.86, 0.18),
            (0.58, 0.24, 0.75),
        ),
        strict=True,
    ):
        _paint_box(screenshot, element.box_xyxy_norm, color)
    screen = ScreenRecord(
        screen_id="synthetic-settings",
        app_id="synthetic-settings-app",
        image_path="<synthetic>",
        width=width,
        height=height,
        split="train",
        source="synthetic",
        elements=elements,
    )
    commands = (
        ("open search", "search", "direct", ()),
        ("press the cart icon", "cart", "direct", ()),
        ("select the item before Bluetooth", "wifi", "ordinal", ("bluetooth",)),
        ("select a setting in the list", "wifi", "containment", ("settings_list",)),
        ("tap the nearby small toggle", "tiny_toggle", "proximity", ("bluetooth",)),
        ("tap the purple icon", "missing_ocr_icon", "direct", ()),
    )
    return tuple(
        SyntheticExample(
            screenshot=screenshot.clone(),
            screen=screen,
            command=CommandRecord(
                command_id=f"synthetic-command-{index}",
                screen_id=screen.screen_id,
                text=text,
                action_type=ActionType.CLICK,
                target_box_xyxy_norm=next(
                    element.box_xyxy_norm for element in elements if element.element_id == target_id
                ),
                target_point_xy_norm=None,
                relation_type=relation_type,
                reference_element_ids=references,
            ),
        )
        for index, (text, target_id, relation_type, references) in enumerate(commands)
    )


def repeat_examples(
    examples: Iterable[SyntheticExample], count: int
) -> tuple[SyntheticExample, ...]:
    """Repeat a finite fixture deterministically for tiny training smoke tests."""

    source = tuple(examples)
    if not source or count <= 0:
        raise ValueError("examples must be non-empty and count must be positive")
    return tuple(source[index % len(source)] for index in range(count))
