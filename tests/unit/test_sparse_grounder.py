from __future__ import annotations

import torch

from screen2action.models.sparse_grounder import SparseCandidateGrounder


def test_sparse_grounder_shapes_finite_and_gradients() -> None:
    torch.manual_seed(2)
    model = SparseCandidateGrounder(embedding_dim=16, heads=4, blocks=2)
    command = torch.randn(2, 5, 16, requires_grad=True)
    crops = torch.randn(2, 3, 4, 16, requires_grad=True)
    nodes = torch.randn(2, 3, 16, requires_grad=True)
    output = model(command, crops, nodes)
    assert output.candidate_logits.shape == (2, 3)
    assert output.point_local.shape == (2, 3, 2)
    assert torch.isfinite(output.candidate_logits).all()
    loss = output.candidate_logits.mean() + output.point_local.mean()
    loss.backward()
    assert command.grad is not None and torch.isfinite(command.grad).all()


def test_candidate_local_attention_has_no_direct_cross_candidate_path() -> None:
    torch.manual_seed(4)
    model = SparseCandidateGrounder(embedding_dim=16, heads=4, blocks=1)
    command = torch.randn(1, 4, 16)
    groups = torch.randn(1, 3, 5, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    baseline = model.local_candidate_update(command, groups, mask)
    changed = groups.clone()
    changed[:, 0] += 100.0
    modified = model.local_candidate_update(command, changed, mask)
    assert torch.allclose(baseline[:, 1:], modified[:, 1:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(baseline[:, 0], modified[:, 0])
