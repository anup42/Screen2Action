"""Screen-grouped sampling and explicit command-label mask tests."""

from __future__ import annotations

import torch

from screen2action.models.batching import CommandBatch, ScreenGroupedSampler


def test_screen_grouped_sampler_keeps_all_commands_for_a_screen_together() -> None:
    sampler = ScreenGroupedSampler(
        ("screen-a", "screen-b", "screen-a", "screen-c", "screen-b"),
        screens_per_batch=2,
    )

    assert list(sampler) == [[0, 2, 1, 4], [3]]
    assert len(sampler) == 2

    shuffled = ScreenGroupedSampler(
        ("screen-a", "screen-b", "screen-a", "screen-c", "screen-b"),
        screens_per_batch=1,
        shuffle=True,
        seed=9,
    )
    first = list(shuffled)
    shuffled.set_epoch(1)
    second = list(shuffled)
    assert first != second
    assert sorted(index for batch in first for index in batch) == list(range(5))


def test_unsupervised_command_batch_has_every_required_mask() -> None:
    batch = CommandBatch.unsupervised(
        torch.tensor([[1, 2, 0], [1, 3, 4]]),
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([0, 1]),
    )

    assert batch.input_ids.shape == (2, 3)
    assert batch.reference_mask.shape == (2, 1)
    assert batch.parameter_mask.shape == (2, 9)
    assert batch.parameter_node_mask.shape == (2, 2)
    assert not batch.target_mask.any()
    assert not batch.action_mask.any()
    assert not batch.target_point_mask.any()
