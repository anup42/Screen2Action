from __future__ import annotations

import torch

from screen2action.runtime.cropper import (
    batch_crop_tensor,
    crop_tensor,
    crop_to_screen_point,
    expand_box,
    screen_to_crop_point,
)


def test_crop_transform_round_trip_and_batch_shape() -> None:
    screenshot = torch.arange(3 * 32 * 48, dtype=torch.float32).reshape(3, 32, 48)
    box = (0.25, 0.25, 0.50, 0.75)
    expanded = expand_box(box, margin_fraction=0.12)
    screen_point = (0.375, 0.50)
    local = screen_to_crop_point(screen_point, expanded)
    restored = crop_to_screen_point(local, expanded)
    assert restored == screen_point
    spec = crop_tensor(screenshot, box, output_size=16)
    assert spec.crop.shape == (3, 16, 16)
    batch, specs = batch_crop_tensor(screenshot, [box, (0.0, 0.0, 0.2, 0.2)], output_size=16)
    assert batch.shape == (2, 3, 16, 16)
    assert len(specs) == 2
