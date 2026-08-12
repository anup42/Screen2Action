# GPU runbook

Install a CUDA-compatible PyTorch build using the official PyTorch selector
before installing repository GPU extras. The repository intentionally does not
pin one CUDA wheel URL.

```text
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -e .[perception,data,train,export,dev]
screen2action doctor --device cuda --json
screen2action train smoke --device cuda
```

Resolve/fetch the model lock, build a frozen canonical data manifest, and
precompute perception before Stage 2. Probe each selected stage against those
real immutable inputs; the command performs an actual forward/backward pass
and recommends accumulation for a global batch of 256:

```text
screen2action train probe-batch \
  --stage stage2_selector_retrieval \
  --manifest ${SCREEN2ACTION_DATA_ROOT}/normalized/public-v1/manifests/training.json \
  --cache-manifest ${SCREEN2ACTION_CACHE_ROOT}/perception/<bundle>/manifest.json \
  --model-lock configs/models/lock.json \
  --config configs/experiments/stage2_selector_retrieval.yaml \
  --device cuda --max-batch-size 64 --json
```

For distributed runs, launch the same internal entry point through torchrun:

```text
torchrun --standalone --nproc-per-node=2 -m screen2action.training.launch \
  --stage stage2_selector_retrieval \
  --manifest ${SCREEN2ACTION_DATA_ROOT}/normalized/public-v1/manifests/training.json \
  --cache-manifest ${SCREEN2ACTION_CACHE_ROOT}/perception/<bundle>/manifest.json \
  --model-lock configs/models/lock.json \
  --config configs/experiments/stage2_selector_retrieval.yaml \
  --device cuda \
  --run-dir ${SCREEN2ACTION_RUN_ROOT}/stage2-selector
```

Initialize Stage 2 from the Stage 1 semantics artifact with
`--init-checkpoint`; use `--resume .../checkpoints/latest.pt` only to continue
the exact same stage lineage. Stage 1 detector and OCR configs invoke their
native Ultralytics/docTR trainers and retain separate checkpoint families.
Stage 3 keeps OCR frozen and, when enabled, alternates detector-native updates
without claiming gradients through NMS. Stage 4 requires a Stage 3
initialization or an exact Stage 4 resume.

After Stage 1 and Stage 3 produce digest-compatible artifacts, create the
profile-separated Accurate package. Export runs ONNXRuntime parity on CPU even
when training occurred on CUDA, so it may be moved to a CPU packaging host with
the same verified lock assets:

```text
screen2action export --profile accurate \
  --config configs/export/accurate.yaml \
  --checkpoint ${SCREEN2ACTION_RUN_ROOT}/stage3/checkpoints/best.pt \
  --visual-checkpoint ${SCREEN2ACTION_RUN_ROOT}/stage1-semantics/checkpoints/visual-model.pt \
  --model-lock configs/models/lock.json \
  --output ${SCREEN2ACTION_RUN_ROOT}/exports --json
```

Do not use `fast_roi` as an alias. It currently writes a separate unsupported
report until an ROIAlign crop-token implementation and distinct trained
checkpoint lineage exist. See `EXPORT_CONTRACTS.md`.

Perception precomputation is data-parallel without DDP collectives. Each
`torchrun` process loads the locked perception bundle on its `LOCAL_RANK`, and
stable screen-hash sharding is inferred from `RANK`/`WORLD_SIZE`:

```text
torchrun --standalone --nproc-per-node=2 -m screen2action \
  perception precompute \
  --manifest ${SCREEN2ACTION_DATA_ROOT}/normalized/public-v1/manifests/training.json \
  --model-lock configs/models/lock.json \
  --visual-checkpoint ${SCREEN2ACTION_RUN_ROOT}/stage1-semantics/checkpoints/visual-model.pt \
  --device cuda --batch-size 8 --json
```

Do not pass contradictory `--shard-index` or `--shard-count` values under
`torchrun`; the command rejects them. Independent CPU processes may use those
flags explicitly. The command's JSON identifies the bundle manifest for
`perception validate` and `perception stats`.

CUDA setup is conditional and uses `torch.device`/`.to(device)`. NCCL is the
preferred multi-GPU backend; Gloo is the CPU fallback. Validate global-batch
math, per-rank local batch, accumulation (including final partial flush),
sampler sharding, synchronized metrics, and rank-zero writes before scaling.
Record GPU names/capabilities, PyTorch/CUDA/NCCL versions, free/total VRAM,
resolved digests, command, and checkpoint in the handoff. A CPU-only host must
leave CUDA gates unverified, not failed or silently emulated.

The first external-machine command after installing dependencies is:

```text
screen2action doctor --device cuda --json
```
