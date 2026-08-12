"""Small JSON event logger for reproducible runtime diagnostics."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, TextIO


class JsonEventFormatter(logging.Formatter):
    """Render one structured Screen2Action event per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": record.created,
        }
        fields = getattr(record, "screen2action_fields", {})
        if isinstance(fields, Mapping):
            payload.update(fields)
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_structured_logging(
    name: str = "screen2action",
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure an explicit logger with deterministic JSON event output."""

    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    fields: Mapping[str, Any] | None = None,
) -> None:
    """Emit an event without serializing screenshots or other raw data."""

    logger.log(
        level,
        event,
        extra={"screen2action_fields": dict(fields or {})},
    )
