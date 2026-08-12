"""Application-disjoint split validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from screen2action.data.schema import ScreenRecord


def validate_app_disjoint(screens: Iterable[ScreenRecord]) -> None:
    """Raise when an app identity appears in more than one split."""

    apps: defaultdict[str, set[str]] = defaultdict(set)
    for screen in screens:
        apps[screen.app_id].add(screen.split)
    conflicts = {app: sorted(splits) for app, splits in apps.items() if len(splits) > 1}
    if conflicts:
        raise ValueError(f"application-disjoint split violation: {conflicts}")
