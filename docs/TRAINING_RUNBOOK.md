# Training runbook

Every run starts from a validated resolved configuration and frozen data/model
digests. Run directories contain `resolved-config.yaml`, `run-manifest.json`,
JSONL metrics, optional TensorBoard events, and rank-zero checkpoints.

## Stages

1. Stage 1 trains perception/graph objectives for 12 epochs. Detector/icon/OCR
   submodes are explicit; mixing incompatible supervision is forbidden.
2. Stage 2 consumes perception cache only and trains selector, retrieval, and
   grounding for 8 epochs. A true positive is inserted with probability 0.5
   only during epochs 0 and 1.
3. Stage 3 jointly fine-tunes non-OCR paths for 6 epochs; OCR stays frozen.
4. Stage 4 is optional QAT and cannot be used to claim quantized accuracy
   without parity and evaluation artifacts.

Paper-reference defaults are AdamW, global command batch 256, 5% warm-up,
weight decay 0.05, gradient clipping 1.0, new-head LR `2e-4`, pretrained LR
`2e-5`, and objective weights documented in `docs/PAPER_SPEC.md`.

Before a long run, use `train probe-batch`, then `train smoke`. Resume must
restore model, optimizer, scheduler, scaler, RNG, sampler epoch/state, step,
best metric, config/model/data/cache digests, and Git revision. A digest
mismatch is an error unless an explicit, logged override permits it.

## Perception cache prerequisite

After Stage 1 semantics/graph training, precompute command-independent frame
state from the exact checkpoint that supplies the MobileNet custom heads:

```powershell
screen2action perception precompute `
  --manifest "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\manifests\training.json" `
  --model-lock configs/models/lock.json `
  --visual-checkpoint "$env:SCREEN2ACTION_RUN_ROOT\stage1_semantics_graph\best.pt" `
  --device cpu --max-screens 8 --batch-size 2 --json
```

`--max-screens` is a deterministic CPU smoke boundary. Omitting it processes
the full manifest. `--allow-untrained-heads` exists only for an explicitly
labeled, seeded plumbing smoke and is rejected by default; its cache is not a
trained Stage 2 input. Use the returned `cache_manifest_path` to run:

```powershell
screen2action perception validate <cache-manifest> --json
screen2action perception stats <cache-manifest> --json
```

Stage 2 must open the cache through `CachedPerceptionLoader`, provide the
expected model-lock/config/data-manifest digests, and refuse incomplete shard
runs. Training workers do not instantiate ScreenParser or CRNN.

## Objective and resume checks

The unified runner uses `training.loss_weights` (`candidate`, `point`, `action`,
`ui`, `survive`, `budget`) and `training.ui_contrastive_temperature`. The shipped
experiment profiles declare the paper defaults. UI loss averages full-screen
cosine and reranked cross-entropy (ADR-0027) so hard candidate selection does not
disconnect retrieval training. Logs include `ui_contrastive_loss`.

Action labels never influence inference candidate selection. Canonical action
points are screen-normalized; model supervision converts long-press and both
drag endpoints to their expanded candidate crops. Missing labels stay masked.

Resume an initialized run using its `latest.pt` and original run directory;
omit `--init-checkpoint` on resume. The original initialization digest is read
from the immutable run manifest and all current config/data/model/code checks
still apply. DDP diagnostics are detached only at the training wrapper boundary;
model-only checkpoints preserve the same keys as single-process models.

The correctness review changed the training objective and candidate selection.
Older weights remain topology-compatible, but an exact resume across changed
code is intentionally rejected. Start a new run with explicit model-only
initialization, and regenerate previous evaluation and calibration reports.
