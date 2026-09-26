from __future__ import annotations

import pytest
import torch

from screen2action.training.checkpoints import load_checkpoint, save_checkpoint
from screen2action.training.distributed import OptimizationOutputAdapter
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


def test_optimization_adapter_preserves_checkpoint_model_keys(tmp_path) -> None:
    model = TinyGroundingModel(vocab_size=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = tmp_path / "adapter.pt"
    save_checkpoint(checkpoint, OptimizationOutputAdapter(model), optimizer, epoch=0, step=0)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert set(state["model"]) == set(model.state_dict())
    load_checkpoint(checkpoint, OptimizationOutputAdapter(model), optimizer)


@pytest.mark.gpu
def test_cuda_checkpoint_restores_cpu_and_device_rng(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = torch.nn.Linear(2, 1).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters())
    seed_everything(31)
    checkpoint = tmp_path / "cuda.pt"
    save_checkpoint(checkpoint, model, optimizer, epoch=0, step=0)
    expected_cpu = torch.rand(3)
    expected_cuda = torch.rand(3, device="cuda")
    load_checkpoint(checkpoint, model, optimizer, device="cuda")
    assert torch.equal(torch.rand(3), expected_cpu)
    assert torch.equal(torch.rand(3, device="cuda"), expected_cuda)
