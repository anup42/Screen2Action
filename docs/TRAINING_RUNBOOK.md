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
