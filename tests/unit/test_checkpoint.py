from __future__ import annotations

import torch

from screen2action.training.checkpoints import load_checkpoint, save_checkpoint
from screen2action.training.seed import seed_everything
from screen2action.training.tiny import TinyGroundingModel


def _step(model, optimizer, node_features, command_ids, command_mask, targets):
    optimizer.zero_grad(set_to_none=True)
    logits, points = model(node_features, command_ids, command_mask)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_checkpoint_resume_restores_model_optimizer_and_rng(tmp_path) -> None:
    seed_everything(11)
    model_a = TinyGroundingModel(vocab_size=12)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=0.01)
    features = torch.eye(4).unsqueeze(0).expand(4, -1, -1)
    commands = torch.tensor([[1, 2, 3] for _ in range(4)])
    mask = torch.ones_like(commands, dtype=torch.bool)
    targets = torch.tensor([0, 1, 2, 3])
    _step(model_a, optimizer_a, features, commands, mask, targets)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model_a, optimizer_a, epoch=1, step=1, config={"device": "cpu"})
    loss_a = _step(model_a, optimizer_a, features, commands, mask, targets)

    seed_everything(99)
    model_b = TinyGroundingModel(vocab_size=12)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.01)
    state = load_checkpoint(checkpoint, model_b, optimizer_b, device="cpu")
    loss_b = _step(model_b, optimizer_b, features, commands, mask, targets)
    assert state.epoch == 1 and state.step == 1
    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=1e-6)
    for left, right in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)
