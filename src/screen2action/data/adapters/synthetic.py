"""Adapter wrapper around the offline synthetic fixture."""

from __future__ import annotations

from dataclasses import dataclass

from screen2action.data.adapters.base import AdapterAudit, AdapterReject, CanonicalExample
from screen2action.data.schema import CommandRecord, ScreenRecord
from screen2action.data.synthetic import SyntheticExample, make_synthetic_examples


@dataclass(slots=True)
class SyntheticAdapter:
    """Expose synthetic examples through the dataset adapter protocol."""

    source_name: str = "synthetic"

    def _examples(self) -> tuple[SyntheticExample, ...]:
        return make_synthetic_examples()

    def screens(self) -> tuple[ScreenRecord, ...]:
        return (self._examples()[0].screen,)

    def examples(self) -> tuple[CanonicalExample, ...]:
        examples = self._examples()
        return (
            CanonicalExample(
                screen=examples[0].screen,
                commands=tuple(example.command for example in examples),
            ),
        )

    def commands(self) -> tuple[CommandRecord, ...]:
        return tuple(example.command for example in self._examples())

    def audit(self) -> AdapterAudit:
        commands = self.commands()
        return AdapterAudit(
            source=self.source_name,
            accepted_screens=1,
            rejected_screens=0,
            accepted_commands=len(commands),
            rejected_commands=0,
            action_counts={"click": len(commands)},
        )

    def rejects(self) -> tuple[AdapterReject, ...]:
        return ()
