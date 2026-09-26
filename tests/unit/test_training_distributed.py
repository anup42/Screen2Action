from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
from torch import nn

from screen2action.training.distributed import initialize_distributed, wrap_ddp
from screen2action.training.engine import LossResult, Trainer


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _gloo_worker(rank: int, world_size: int, port: int, output: str) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    context = initialize_distributed("cpu")
    try:
        torch.manual_seed(5)
        model = wrap_ddp(nn.Linear(2, 1), context)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        value = torch.tensor([[float(rank), 1.0]])
        loss = model(value).sum()
        loss.backward()
        optimizer.step()
        checksum = sum(float(parameter.detach().sum()) for parameter in model.parameters())
        reduced = context.reduce_sum(torch.tensor(float(rank + 1)))
        Path(output, f"rank-{rank}.txt").write_text(
            f"{checksum:.8f},{float(reduced):.1f}",
            encoding="utf-8",
        )
        context.barrier()
        buffered = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))
        validation_model = wrap_ddp(buffered, context)
        buffered[0].running_mean.fill_(float(rank))
        trainer = Trainer(
            validation_model,
            torch.optim.SGD(validation_model.parameters(), lr=0.1),
            distributed=context,
        )
        validation = trainer.validate(
            [torch.ones(2, 2)] if rank == 0 else [],
            lambda module, batch, step: LossResult(module(batch).square().mean(), {}, 2),
        )
        assert validation.examples == 2
        assert torch.isfinite(torch.tensor(validation.mean_loss))
        assert torch.equal(buffered[0].running_mean, torch.zeros(2))
    finally:
        context.cleanup()


def test_two_process_cpu_ddp_smoke(tmp_path: Path) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Torch Gloo distributed support is unavailable")
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rows = [
        (tmp_path / f"rank-{rank}.txt").read_text(encoding="utf-8").split(",") for rank in range(2)
    ]
    assert rows[0][0] == rows[1][0]
    assert rows[0][1] == rows[1][1] == "3.0"
