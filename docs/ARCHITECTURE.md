# Architecture

Screen2Action is split at a structured semantic bottleneck (SSB). The frame
path is command-independent and cacheable; the command path consumes only the
cached frame representation, the current screenshot for candidate crops, and
command tokens.

```text
screenshot
  -> ScreenParser + CRNN + MobileNetV3 heads
  -> normalized typed nodes and relation graph
  -> two relation-aware graph layers
  -> learned retention -> deterministic BudgetSelect/closure -> SSB/cache

command + cached frame
  -> compact BERT command states
  -> cosine retrieval + relation reranking -> fixed K candidates
  -> expanded crops -> MobileViT-S spatial tokens
  -> two candidate-local sparse cross-attention blocks
  -> target/point/action/parameters/confidence
```

## Truth boundary

`docs/PAPER_SPEC.md` is the paper-fact contract. Public model choices,
thresholds, adapters, SSB grammar, weak references, and partition details are
reconstructions. They are configured, logged, and recorded in
`docs/DECISIONS.md`; they are not evidence of paper-exact reproduction.

## Core contracts

- Internal geometry is normalized `xyxy`; pixels are boundary-only.
- A frame's retained graph and serialization cannot depend on a command.
- Missing labels and modalities carry masks; absence is not a negative label.
- Production perception implements typed protocols and can be replaced with
  oracle/cached implementations without changing downstream schemas.
- Selection is stochastic/differentiable during training and deterministic,
  budget-bounded, and closure-safe at inference.
- Sparse grounding isolates each candidate group from other candidates.
- CPU is a supported execution device for all public APIs. CUDA accelerates
  optional work but never changes schemas.

## State and provenance

External assets live under environment roots. Registries are mutable intent;
lock files and manifests freeze resolved revisions and content digests.
Perception caches bind image, model bundle, preprocessing, taxonomy, and schema
digests. Checkpoints bind configuration, data manifest, Git revision, stage,
optimizer/scheduler/scaler/RNG/sampler state. `handoff.json` inventories all of
these without exposing credentials or machine-local paths.

## Export boundary

Host code retains image decoding, detector/OCR postprocessing, graph
construction, exact BudgetSelect/closure, crop extraction, cache management,
and final coordinate conversion. Exportable fixed-shape neural partitions are
perception, graph/retention, command retrieval/reranking, and candidate
grounding/action heads. A partition is supported only after PyTorch/export
parity is observed for that exact profile.
