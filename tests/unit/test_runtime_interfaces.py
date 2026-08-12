from __future__ import annotations

import io
import json
import logging

from screen2action.data.schema import ActionType
from screen2action.data.synthetic import make_synthetic_examples
from screen2action.runtime.actions import decode_action
from screen2action.runtime.frame_cache import FrameCache, frame_cache_key
from screen2action.runtime.structured_logging import configure_structured_logging, log_event


def test_action_decoding_and_frame_cache() -> None:
    click = decode_action(ActionType.CLICK, (0.2, 0.3))
    scroll = decode_action(ActionType.SCROLL, (0.2, 0.3), (0.0, -0.5))
    drag = decode_action(ActionType.DRAG, (0.2, 0.3), (0.8, 0.9, 0.4))
    assert click.point == (0.2, 0.3)
    assert scroll.displacement == (0.0, -0.5)
    assert drag.destination == (0.8, 0.9)
    screenshot = make_synthetic_examples()[0].screenshot
    cache = FrameCache[int](max_entries=1)
    key = frame_cache_key(screenshot, "portrait")
    cache.put(key, 7)
    assert cache.get(key) == 7
    cache.clear()
    assert cache.get(key) is None


def test_structured_runtime_events_are_json_and_do_not_include_pixels() -> None:
    stream = io.StringIO()
    logger = configure_structured_logging(
        "screen2action.test.logging", level=logging.INFO, stream=stream
    )
    log_event(logger, "frame_prepared", fields={"node_count": 3, "width": 160})
    payload = json.loads(stream.getvalue())
    assert payload["message"] == "frame_prepared"
    assert payload["node_count"] == 3
    assert "screenshot" not in payload
