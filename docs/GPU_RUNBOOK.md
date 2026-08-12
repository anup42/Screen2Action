# GPU runbook

Install a CUDA-compatible PyTorch build using the official PyTorch selector
before installing repository GPU extras. The repository intentionally does not
pin one CUDA wheel URL.

```text
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -e .[perception,data,train,export,dev]
screen2action doctor --device cuda --json
screen2action train probe-batch --device cuda
screen2action train smoke --device cuda
```

For distributed runs, launch the same internal entry point through torchrun:

```text
torchrun --standalone --nproc-per-node=2 -m screen2action.training.launch \
  --config configs/experiments/paper_reference_stage2.yaml
```

CUDA setup is conditional and uses `torch.device`/`.to(device)`. NCCL is the
preferred multi-GPU backend; Gloo is the CPU fallback. Validate global-batch
math, per-rank local batch, accumulation (including final partial flush),
sampler sharding, synchronized metrics, and rank-zero writes before scaling.
Record GPU names/capabilities, PyTorch/CUDA/NCCL versions, free/total VRAM,
resolved digests, command, and checkpoint in the handoff. A CPU-only host must
leave CUDA gates unverified, not failed or silently emulated.
