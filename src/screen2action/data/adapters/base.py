"""Typed adapter boundary and conversion audit."""

from __future__ import annotations

from collections.abc import Iterable
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
    action_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class DatasetAdapter(Protocol):
    """Protocol implemented by external dataset adapters."""

    source_name: str

    def screens(self) -> Iterable[ScreenRecord]:
        """Yield canonical screens without downloading data."""

    def commands(self) -> Iterable[CommandRecord]:
        """Yield canonical commands without downloading data."""

    def audit(self) -> AdapterAudit:
        """Return conversion audit metadata."""


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
