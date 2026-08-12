"""WaveUI, RICO Semantics, and ScreenSpot grounding adapters."""

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
    normalize_box,
    record_dimensions,
)
from screen2action.data.schema import (
    ActionType,
    CommandRecord,
    NodeType,
    PointSource,
)


class WaveUiAdapter(MaterializedPublicAdapter):
    """Convert official AgentSea WaveUI grounding rows."""

    source_name = "wave_ui"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        row_source = str(record.get("source") or "").strip()
        if not row_source:
            raise ConversionRejected(
                "missing_row_license_source",
                "WaveUI requires each row's source for downstream license review",
            )
        width, height = record_dimensions(record)
        target_box = normalize_box(record.get("bbox"), width, height)
        raw_type = record.get("type") or "control"
        elements = build_elements(
            self,
            [
                {
                    "id": "target",
                    "bbox": record.get("bbox"),
                    "type": raw_type,
                    "text": record.get("name") or record.get("OCR"),
                }
            ],
            source_item_id=source_item_id,
            width=width,
            height=height,
            default_type=NodeType.CONTROL,
            annotation_source="wave_ui_grounding",
        )
        screen, image_bytes, image_format = build_screen(
            self,
            record,
            source_item_id=source_item_id,
            elements=elements,
            platform=str(record.get("platform") or "unknown"),
            split=canonical_split(record.get("split", "train")),
            app_raw=record.get("app_id") or record.get("domain") or row_source,
            domain_raw=record.get("domain"),
        )
        instruction = str(record.get("instruction") or "").strip()
        if not instruction:
            raise ConversionRejected("missing_command", "WaveUI instruction is empty")
        x1, y1, x2, y2 = target_box
        pseudo_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        masks = {"target": True, "point": False, "action": False, "parameters": False}
        provenance = replace(
            screen.provenance,
            target_match_method="source_bbox",
            label_masks=masks,
            transformation_provenance={
                "wave_ui_row_source": row_source,
                "wave_ui_source_license": str(record.get("source_license", "review_required")),
            },
        )
        command = CommandRecord(
            command_id=f"{screen.screen_id}:command",
            screen_id=screen.screen_id,
            text=instruction,
            action_type=ActionType.UNKNOWN,
            target_box_xyxy_norm=target_box,
            target_point_xy_norm=pseudo_point,
            target_point_source=PointSource.PSEUDO_BOX_CENTER,
            label_masks=masks,
            provenance=provenance,
        )
        return (CanonicalExample(screen, (command,), image_bytes, image_format),)


class RicoSemanticsAdapter(MaterializedPublicAdapter):
    """Convert RICO Semantics elements without inventing commands or actions."""

    source_name = "rico_semantics"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        width = int(float(str(record.get("width", 1440))))
        height = int(float(str(record.get("height", 2560))))
        enriched = dict(record)
        enriched["width"] = width
        enriched["height"] = height
        app_id = record.get("app_id") or record.get("package_name")
        if not app_id:
            raise ConversionRejected(
                "missing_app_identity",
                "RICO Semantics must be joined with RICO UI metadata before normalization",
            )
        raw_elements = record.get("screen_elements") or record.get("elements")
        elements = build_elements(
            self,
            raw_elements,
            source_item_id=source_item_id,
            width=width,
            height=height,
            box_key="bbox",
            default_type=NodeType.ICON,
            annotation_source="rico_semantics_human",
        )
        if not elements:
            raise ConversionRejected(
                "no_valid_elements", "RICO screen has no convertible annotations"
            )
        screen, image_bytes, image_format = build_screen(
            self,
            enriched,
            source_item_id=source_item_id,
            elements=elements,
            platform="android",
            split=canonical_split(record.get("split", "train")),
            app_raw=app_id,
        )
        return (CanonicalExample(screen, (), image_bytes, image_format),)


class ScreenSpotAdapter(MaterializedPublicAdapter):
    """Evaluation-only ScreenSpot adapter with hard training leakage provenance."""

    source_name = "screenspot"

    def convert_record(
        self,
        record: Mapping[str, object],
        *,
        source_item_id: str,
    ) -> tuple[CanonicalExample, ...]:
        width, height = record_dimensions(record)
        target_box = normalize_box(record.get("bbox"), width, height)
        target_type = str(record.get("data_type") or record.get("target_type") or "control")
        elements = build_elements(
            self,
            [{"id": "target", "bbox": record.get("bbox"), "type": target_type}],
            source_item_id=source_item_id,
            width=width,
            height=height,
            default_type=NodeType.CONTROL,
            annotation_source="screenspot_benchmark",
        )
        platform = str(record.get("data_source") or record.get("data_souce") or "unknown")
        screen, image_bytes, image_format = build_screen(
            self,
            record,
            source_item_id=source_item_id,
            elements=elements,
            platform=platform,
            split=canonical_split(record.get("split"), evaluation_only=True),
            app_raw=record.get("app_id") or record.get("domain") or platform,
            domain_raw=record.get("domain"),
            image_path=str(record.get("img_filename") or record.get("image_path") or ""),
        )
        instruction = str(record.get("instruction") or "").strip()
        if not instruction:
            raise ConversionRejected("missing_command", "ScreenSpot instruction is empty")
        x1, y1, x2, y2 = target_box
        pseudo_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        masks = {"target": True, "point": False, "action": True, "parameters": False}
        provenance = replace(
            screen.provenance,
            target_match_method="official_screenspot_bbox",
            split_origin="official_evaluation_only",
            label_masks=masks,
            transformation_provenance={"hard_leakage_guard": "true"},
        )
        command = CommandRecord(
            command_id=f"{screen.screen_id}:command",
            screen_id=screen.screen_id,
            text=instruction,
            action_type=ActionType.CLICK,
            target_box_xyxy_norm=target_box,
            target_point_xy_norm=pseudo_point,
            target_point_source=PointSource.PSEUDO_BOX_CENTER,
            label_masks=masks,
            provenance=provenance,
        )
        return (CanonicalExample(screen, (command,), image_bytes, image_format),)
