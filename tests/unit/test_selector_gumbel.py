from __future__ import annotations

import torch

from screen2action.ssb.selector import annealed_temperature, straight_through_gumbel_mask


def test_gumbel_selector_is_hard_forward_and_has_gradients() -> None:
    logits = torch.zeros(8, requires_grad=True)
    generator = torch.Generator().manual_seed(4)
    mask = straight_through_gumbel_mask(
        logits,
        annealed_temperature(50, 100),
        mandatory_mask=torch.tensor([True, False, False, False, False, False, False, False]),
        generator=generator,
    )
    assert mask[0].item() == 1.0
    assert torch.all((mask.detach() == 0.0) | (mask.detach() == 1.0))
    mask.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
