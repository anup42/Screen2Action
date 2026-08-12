"""Typed adapter boundary and conversion audit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from screen2action.data.schema import CommandRecord, ScreenRecord
from screen2action.data.splits import validate_app_disjoint


@dataclass(frozen=True, slots=True)
class AdapterAudit:
    """Conversion counts and warnings retained with a dataset manifest."""

    source: str
    accepted_screens: int
    rejected_screens: int
    accepted_commands: int
    rejected_commands: int
    accepted_elements: int = 0
    rejected_elements: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdapterReject:
    """One source record that could not be converted without semantic coercion."""

    source: str
    source_item_id: str
    reason_code: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalExample:
    """Streaming adapter output for one unique screenshot and its commands."""

    screen: ScreenRecord
    commands: tuple[CommandRecord, ...]
    image_bytes: bytes | None = None
    image_format: str = "png"
    source_metadata: Mapping[str, str] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    """Protocol implemented by external dataset adapters."""

    source_name: str

    def examples(self) -> Iterable[CanonicalExample]:
        """Yield each unique screen once with all command-state labels."""

    def screens(self) -> Iterable[ScreenRecord]:
        """Yield canonical screens without downloading data."""

    def commands(self) -> Iterable[CommandRecord]:
        """Yield canonical commands without downloading data."""

    def audit(self) -> AdapterAudit:
        """Return conversion audit metadata."""

    def rejects(self) -> Iterable[AdapterReject]:
        """Return explicit conversion rejections with reason codes."""


def validate_adapter_output(
    screens: Iterable[ScreenRecord],
    commands: Iterable[CommandRecord],
) -> None:
    """Validate canonical output and application-disjoint membership."""

    screen_list = list(screens)
    command_list = list(commands)
    screen_ids = {screen.screen_id for screen in screen_list}
    if len(screen_ids) != len(screen_list):
        raise ValueError("adapter emitted duplicate screen IDs")
    if any(command.screen_id not in screen_ids for command in command_list):
        raise ValueError("adapter emitted a command for an unknown screen")
    validate_app_disjoint(screen_list)
