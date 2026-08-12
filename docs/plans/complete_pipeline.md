# Complete production, data, and GPU-training pipeline

**Status:** ACTIVE
**Active milestone:** Completion workflow - final handoff, publication, and archive
**Started:** 2026-08-12
**Specification:** `Screen2Action_Remaining_Gaps_Codex_Prompt.md` supplied by
the user (external input; not copied into the repository)

## Purpose and observable outcome

Turn the verified CPU architecture scaffold into a production-oriented,
configuration-driven Screen2Action repository that can be transferred to a
GPU machine and resumed from repository documentation alone. A user must be
able to inspect the environment, resolve optional public model assets,
register or download licensed data, normalize/audit it, precompute
command-independent perception, run stage-aware training, evaluate and
calibrate, export supported subgraphs, and produce a reproducible handoff.

The implementation must not claim the paper's 80.2% ScreenSpot accuracy, 185M
parameter count, 190 MB disk size, or Galaxy latency without direct
measurement under the stated protocol.

## Truth hierarchy

1. `docs/PAPER_SPEC.md` for facts explicitly stated by the paper.
2. The architecture/equations and locked public reconstruction choices in the
   user-supplied remaining-gaps specification.
3. Existing deterministic invariants and public APIs where compatible.
4. Accepted decisions in `docs/DECISIONS.md`.
5. New, visibly labeled reconstruction decisions recorded during this plan.

## Non-negotiable constraints

- CPU is supported by every public API and default smoke test.
- No direct model `.cuda()` calls and no unconditional CUDA imports.
- Default tests require no network, model weights, datasets, credentials, or
  GPU.
- Heavy dependencies are lazy and optional.
- Internal boxes remain normalized `xyxy`.
- ScreenSpot is evaluation-only and protected by a hard leakage guard.
- Raw data, screenshots, weights, caches, checkpoints, secrets, and
  machine-specific paths are excluded from Git.
- Downloads are explicit, license-gated, resumable, revision/checksum aware.
- Missing supervision uses masks.
- The deterministic graph/SSB/knapsack/closure implementation remains intact.
- Commands provide help, structured errors, optional JSON, and dry-run where
  mutation/download is meaningful.

## Baseline observed before this plan

- Branch `main` exists but has no commits; the implementation is untracked.
- The CPU baseline was rerun on 2026-08-12: 41 tests passed with full offline
  formatting, lint, and type-check verification.
- The repository contains canonical schemas, deterministic SSB and selection,
  tiny neural modules, oracle perception, losses/metrics/calibration,
  checkpointing, and tiny ONNX parity.
- It lacks the production model registry, real perception, unified trainable
  model, canonical data lake/source adapters, scalable cache, full DDP stage
  runner, production evaluation reports, and complete handoff.
- Several ignored/local presentation and test-output directories are present
  outside the source scope; they must not be staged or modified by this plan.

## Milestones and acceptance evidence

### Milestone 0 - repository truth and resumable handoff (COMPLETE)

- [x] Add `.agent/PLANS.md` and this living ExecPlan.
- [x] Require the read order and ExecPlan discipline in `AGENTS.md`.
- [x] Audit `docs/TRACEABILITY.md` into CPU, production, training, and
      deployment dimensions.
- [x] Reconcile tiny ONNX completion with paper-reference export status.
- [x] Add architecture/repository/provenance/data/training/GPU/handoff docs.
- [x] Add `.env.example`, model registry, and source registry.
- [x] Implement `screen2action handoff snapshot` and generate `HANDOFF.md` plus
      `handoff.json` using relative/environment-root paths.
- [x] Focused tests and full offline verification pass.
- [x] Prepare a Git checkpoint containing only repository source/config/docs.

### Milestone 1 - typed configuration, CLI, and environment diagnostics (COMPLETE)

- [x] Stable schema-versioned YAML composition with environment expansion,
      repeatable dotted overrides, validation, and resolved snapshots.
- [x] One `screen2action` CLI and matching `python -m screen2action` entrypoint
      exposing every required command group.
- [x] `doctor --device cpu|cuda --json` reports environment, roots, optional
      features, assets/manifests/caches/checkpoints, warnings, and next command.
- [x] Package extras do not force a CUDA-specific PyTorch wheel.
- [x] Idempotent CPU/GPU bootstrap and smoke shell scripts.
- [x] CLI/config/doctor tests and full offline verification pass.

### Milestone 2 - model registry and reproducible assets (COMPLETE)

- [x] Registry covers ScreenParser, docTR CRNN, MobileNetV3-small, compact
      BERT, and MobileViT-S with licenses and I/O contracts.
- [x] Immutable lock entries record resolved revisions/files/SHA256 digests.
- [x] Resumable cache-root fetch and fully offline verify paths.
- [x] Mocked unit tests and an opt-in `network` metadata smoke test; GPU is not
      relevant to asset resolution/fetch.
- [x] Full offline verification and checkpoint prepared.

### Milestone 3 - production perception and taxonomy (COMPLETE)

- [x] Typed detector/OCR/text-detector protocols and normalized image boundary.
- [x] ScreenParser conversion/class mapping/diagnostics/root/container policy.
- [x] docTR CRNN crop normalization, aspect batching, ordering, diagnostics,
      and deterministic OCR association.
- [x] Shared MobileNetV3 icon/actionability/visual module with masked losses.
- [x] Inspect/build/freeze icon taxonomy workflow with immutable 87-class
      reconstruction manifest and review report.
- [x] Command-independent real/oracle/cached `FullScreenPerception`.
- [x] Atomic, sharded, content-addressed perception cache with retries.
- [x] Offline fake-backbone/cache tests; real-model execution remains an
      explicitly unverified external gate.

### Milestone 4 - unified trainable paper-reference model (COMPLETE)

- [x] `Screen2ActionModel` separates perceive/encode/ground/forward and permits
      caller-controlled train/eval/freezing.
- [x] Batched multi-screen/multi-command representation and grouped sampler.
- [x] 256-D multimodal node fusion with provenance/missing-modality masks.
- [x] Explicit paper-equation and CPU-reconstruction graph/reranker variants;
      edge-list/dense/batched parity.
- [x] Learned command-independent retention connected to Gumbel training and
      deterministic inference selection.
- [x] Lazy compact BERT and MobileViT adapters with fake-backbone tests.
- [x] MobileViT contract `[B*K, 144, 256]` and shape probe.
- [x] 2-D box encoding, candidate/crop masks, grouped sparse isolation.
- [x] Typed click/long-press/scroll/drag/confidence heads and masked losses.
- [x] Tiny cached forward/backward and model command-independence tests.

### Milestone 5 - canonical data lake and audits (COMPLETE)

- [x] Backward-compatible provenance-rich schemas and environment-root layout.
- [x] Source registry and fixture adapters for GUIAct, GUIEnv, AMEX,
      AndroidControl, WaveUI, RICO Semantics, and ScreenSpot.
- [x] Explicit license-gated resumable download/register-local boundaries.
- [x] App-disjoint split creation with alias/collision audit.
- [x] Scalable staged dedup grouping/survivor/removal audit.
- [x] Hard ScreenSpot train/calibration/model-selection leakage guard.
- [x] `public_weak_reference_v1` annotation with confidence/provenance.
- [x] Manifest/shard creation and digest enforcement.
- [x] CLI fixture workflow exercises register -> normalize -> validate -> split
      -> dedup -> manifest without copyrighted assets.

### Milestone 6 - resumable perception precompute (COMPLETE)

- [x] Unique-screen deterministic sharding, atomic writes, resume/retry,
      failure logs, compatibility digests, cache validation/stats.
- [x] CPU sample path and CUDA/multiprocess-ready path.
- [x] Stage-2 cached loader does not instantiate ScreenParser/docTR.
- [x] Resume/atomicity/command-independence tests.

### Milestone 7 - GPU/DDP stage training (COMPLETE)

- [x] Torchrun local-rank setup, Gloo/NCCL selection, DDP, samplers, cleanup,
      no-sync accumulation, all-reduced metrics, deterministic seeds.
- [x] Stage-aware freezing and Stage 1 submodes, Stage 2 schedule, Stage 3
      non-OCR joint path, optional Stage 4 quantization.
- [x] Correct global-batch accumulation including final partial flush.
- [x] Conditional BF16/FP16/scaler/clip, validation, latest/best rank-zero
      checkpoints, exact resume including sampler state.
- [x] JSONL/TensorBoard/run-manifest timing/throughput/VRAM/ETA reporting.
- [x] CPU smoke, batch probe, one-process and supported two-process tests.

### Milestone 8 - evaluation, calibration, sweeps, and reports (COMPLETE)

- [x] ScreenSpot evaluation-only adapter and official point-in-target path.
- [x] Direct cascade/subset/action/confidence/latency/memory/failure metrics.
- [x] Per-example CSV/Parquet, JSON metrics, Markdown report, and provenance.
- [x] Validation-only temperature/threshold artifacts tied to digests.
- [x] B/K, selector/loss/relation/model/quantization ablations from config.
- [x] Offline report/calibration/leakage tests.

### Milestone 9 - partitioned export/runtime (COMPLETE)

- [x] Explicit host operations and four logical export partitions.
- [x] Fixed-shape contracts and CPU PyTorch/export parity for supported
      paper-reference subgraphs.
- [x] Clear partition report for unsupported operations.
- [x] Accurate/Fast artifacts and metrics remain separate.
- [x] Export tests and handoff-ready documentation.

## Progress log

- **2026-08-12:** Read the complete 1,006-line task specification, inspected
  the uncommitted repository baseline, and created the required plan rules and
  active ExecPlan. No production-integration gate has yet been claimed.
- **2026-08-12:** Reverified the preserved CPU baseline (41 tests), split
  traceability by fidelity tier, corrected the ONNX contradiction, added the
  operational/provenance documents and registries, and implemented the
  portable human/machine handoff generator.
- **2026-08-12:** Milestone 0 focused handoff tests passed (2 tests); the full
  verifier passed with 43 tests. GitHub connector access confirmed admin/push
  permission to the user-supplied private repository. Milestone 1 is active.
- **2026-08-12:** Milestone 0 was pushed to `anup42/Screen2Action`; local and
  remote `main` matched at `e98440e59f1e2bc2a6554b198ff453ed1e272b5a`.
- **2026-08-12:** Added schema-versioned YAML composition/overrides/snapshots,
  all required CLI paths and help, a secret-safe CPU/CUDA doctor, feature
  extras, bootstrap/smoke scripts, and a CPU training/checkpoint/resume smoke.
  Milestone 1 focused tests (8) and the full suite (51) passed. Milestone 2 is
  active.
- **2026-08-12:** Milestone 1 and refreshed handoff were pushed; local and
  remote `main` matched at `c2c69047b92790fb8ef987e48ae39c5e1bdc981b`.
- **2026-08-12:** Implemented the five-role model registry/lock/fetch/verify
  system with lazy official providers, role-scoped license acknowledgements,
  atomic per-role lock finalization, and strict offline SHA256 verification.
  Official cards corrected ScreenParser to Apache-2.0 and MobileViT to its
  linked Apple license reference. Milestone 3 is active.
- **2026-08-12:** Added typed ScreenParser/docTR/shared-MobileNet adapters,
  deterministic OCR association, an explicit reviewed taxonomy workflow,
  command-independent real/oracle/cached orchestration, and an atomic sharded
  multi-worker cache. Sixteen focused tests passed, including a Windows lock
  race. Real public weights remain unfetched and therefore unclaimed.
- **2026-08-13:** Added the unified trainable model, padded multi-screen masks,
  grouped command sampler, 256-D paper profile, graph variants with batched
  parity, exact relation reranking, learned retention, offline compact-BERT and
  MobileViT adapters, sparse absolute-box conditioning, and typed action and
  confidence losses. Milestone 5 is active; public weights remain unexecuted.
- **2026-08-13:** Implemented canonical schema 2.0 and the environment-rooted
  data lake; license-gated immutable acquisition; fixture-backed adapters for
  all seven registered sources; app-disjoint split creation; indexed staged
  deduplication; weak-reference reconstruction; frozen digest-enforced
  manifests; and deterministic training shards. Eighteen focused offline tests
  pass. Full public downloads and source-quality audits remain external gates;
  Milestone 6 is active.
- **2026-08-13:** Added manifest-bound unique-screen precomputation with stable
  process/torchrun hash shards, atomic run progress, entry-level resume and
  retries, stale-artifact recovery, raw and final perception evidence,
  compatibility manifests, offline validation/stats, and a cache-only Stage 2
  loader. Real locked models and CUDA remain opt-in external gates; Milestone 7
  is active.
- **2026-08-13:** Implemented Gloo/NCCL `torchrun` stage training, exact
  mid-epoch multi-rank resume, final accumulation flush, precision and optimizer
  policies, structured manifests/metrics, native detector/OCR sub-stages,
  Stage 2 schedules, Stage 3 semantic/native boundaries, Stage 4 QAT/PTQ
  comparison, and a real batch probe. Nineteen focused tests pass, including a
  two-process CPU run and exact interrupted/resumed parity. CUDA, real public
  weights, and native Ultralytics/docTR runs remain external gates; Milestone 8
  is active.
- **2026-08-13:** Added manifest-bound direct cascade evaluation, official
  ScreenSpot point scoring and leakage enforcement, platform/target subsets,
  action/confidence/latency/memory/failure metrics, complete report packages,
  validation-only linked calibration, and the required B/K/ablation matrix.
  Training-loss/QAT jobs require distinct checkpoints and optional ROIAlign
  Fast remains explicitly unsupported. Eighteen focused tests pass; Milestone 9
  is active.
- **2026-08-13:** Implemented four fixed-shape ONNX partitions, vectorized
  dense graph/reranker equations with edge-path parity, atomic profile
  packages, explicit host ownership, and checkpoint/model-lock provenance.
  The complete tiny package test exported and validated all four partitions.
  An opt-in paper-shape graph/command ONNX parity test passed. Fast ROI emits a
  separate unsupported report and performs no Accurate substitution. Final
  full-suite verification and handoff refresh remain.
- **2026-08-13:** The Milestone 9 full offline gate passed: 177 Python files
  were formatted, Ruff and Mypy passed, and 137 tests passed with nine opt-in
  network/GPU/slow tests deselected. Implementation milestones 0-9 are
  complete; final publication, handoff snapshot, and source archive remain.

## Discoveries and surprises

- Git is initialized but `main` has no commit. Milestone checkpoints therefore
  need a carefully scoped first source commit rather than a diff against an
  existing baseline.
- The workspace contains local presentation/test-output directories with
  malformed flattened names and at least one inaccessible path. They are
  outside repository source scope and will be ignored, not cleaned up.
- The task prompt's baseline test count was stale: current baseline collection
  is 41 tests, all passing before Milestone 0 changes.
- A single dense relation ID per source/destination pair would have erased
  overlapping containment/proximity/ordinal edges. Export therefore uses a
  boolean relation axis and separately indexed geometry, with host-side typed
  edge capacity validation and parity for overlap and empty-graph cases.

## Decision log

- **D-001 (process reconstruction):** Follow milestone dependencies
  sequentially, while implementing independent files/tests in parallel where
  safe. One writer owns each file and the full verifier is the merge gate.
- **D-002 (Git checkpoint):** Because no initial commit exists, the first
  checkpoint will include the preserved CPU scaffold plus Milestone 0 source
  artifacts, excluding ignored/local outputs. Later checkpoints will be
  milestone-scoped.
- **D-003 (external gates):** Real model, public data, CUDA/DDP, and device
  checks are opt-in. Offline fake/fixture boundaries can be complete locally;
  real external gates remain explicitly unverified until run.
- **D-004 (publication):** Push scoped milestone checkpoints to the private
  `anup42/Screen2Action` repository. Because the remote is empty, establish
  `main` directly; no pull request has a meaningful base for the initial
  checkpoint. Exclude all local presentation/output directories.
- **D-005 (configuration):** Keep YAML as the only core runtime dependency.
  All heavy model/data/training/export imports remain lazy and feature-gated;
  no package metadata selects a CUDA wheel.
- **D-006 (asset lock):** Resolve mutable Hub/package intent first, then
  finalize local sizes/SHA256 after license-gated fetch. Preserve non-SPDX
  upstream license terms instead of normalizing them to a convenient license.
- **D-007 (perception truth boundary):** Preserve original detector semantics
  beside coarse SSB types. Label generated containers as reconstruction,
  propagate OCR to at most one enclosing control, and cache only
  command-independent frame state.
- **D-008 (trainable model boundary):** Keep model mode/freezing under caller
  control, compute frame retention before commands, name paper and CPU graph
  variants explicitly, and treat `grounding_correctness_v1` as a configurable
  reconstruction rather than a paper-defined confidence target.
- **D-009 (public data boundary):** Preserve ambiguous upstream license labels,
  reject records lacking safe semantics/app identity, split before synthetic
  augmentation, and treat weak references and dedup thresholds as versioned
  public reconstructions. Fixture success is not full-corpus validation.
- **D-010 (precompute coordination):** Use deterministic hash partitioning for
  independent model-owning processes rather than DDP collectives. Bind every
  cache to model/config/checkpoint/schema digests, reject command fields, and
  require complete validated shards in Stage 2. Short directory prefixes are
  collision-checked against full immutable digests for Windows portability.
- **D-011 (training boundary):** Use exact-resume DDP for downstream stages,
  native loss APIs for detector/OCR sub-stages, and separate model-only stage
  initialization. Do not claim gradients across discrete NMS or OCR decoding.
- **D-012 (quantization boundary):** Treat Stage 4 as per-channel weight QAT,
  training-only activation observation, and same-data FP/PTQ/QAT comparison;
  retain explicit unsupported operations and no unmeasured mobile claim.
- **D-013 (evaluation truth boundary):** Aggregate typed per-command cascade
  observations directly; fit calibration only on held-out non-ScreenSpot
  validation rows; and require distinct checkpoint lineages for training-loss
  and QAT ablations instead of relabeling a baseline result.
- **D-014 (export boundary):** Keep discrete geometry/selection/crop policies
  on the host; use fixed dense neural graph tensors with direct edge-equation
  parity; require per-profile ONNXRuntime parity and lineage reports; never
  substitute Accurate artifacts for unsupported Fast ROI.

## Validation ledger

- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, format/compile checks, and 41 offline tests passed in 8.9 seconds.
- **2026-08-12:** `python -m pytest tests/unit/test_handoff.py -q` - exit 0;
  2 focused tests passed.
- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, format/compile checks, and 43 offline tests passed in 13.2 seconds.
- **2026-08-12:** `python -m pytest tests/unit/test_config.py
  tests/unit/test_cli.py tests/unit/test_doctor.py
  tests/unit/test_training_smoke.py -q` - exit 0; 8 focused tests passed.
- **2026-08-12:** `screen2action doctor --device cpu --json` - exit 0; CPU
  PyTorch, root writability, RAM/disk, optional features, and unresolved model
  lock reported without absolute configured paths or credential values.
- **2026-08-12:** `screen2action config validate --config
  configs/model/tiny_cpu.yaml --set retrieval_top_k=5 --json` - exit 0;
  schema 1 validated with deterministic digest.
- **2026-08-12:** `screen2action train smoke --device cpu --steps 2 --json` -
  exit 0; finite forward/backward loss and exact checkpoint/resume parity.
- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, format/compile checks, and 51 offline tests passed in 9.5 seconds.
- **2026-08-12:** `python -m screen2action.tools.verify all` after final doctor
  package-version reporting - exit 0; 51 tests and all static checks passed in
  9.9 seconds.
- **2026-08-12:** `python -m pytest tests/unit/test_model_assets.py
  tests/unit/test_cli.py tests/unit/test_doctor.py -q` - exit 0; 8 focused
  tests passed.
- **2026-08-12:** `screen2action models list --json` and `screen2action models
  resolve-lock --dry-run --json` - exit 0; five required roles and a
  deterministic no-write plan were reported.
- **2026-08-12:** `python -m pytest
  tests/optional/test_model_assets_network.py -m network -q` - exit 1; the
  Hugging Face HTTPS connection was reset by the remote/network path before
  metadata resolution. No file was downloaded and no real-model gate is
  claimed.
- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, formatting/compilation, and 54 offline tests passed; one network test
  was deselected as intended.
- **2026-08-12:** focused Milestone 3 tests - exit 0; 16 detector/OCR/model,
  taxonomy, concurrent-cache, and full-perception tests passed.
- **2026-08-12:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, formatting/compilation, and 70 offline tests passed; three opt-in real
  model/network/GPU tests were deselected as intended.
- **2026-08-13:** `python -m screen2action.tools.verify all` - exit 0; Ruff,
  Mypy, formatting/compilation, and 87 offline tests passed; three opt-in
  network/GPU tests were deselected as intended.
- **2026-08-13:** focused Milestone 5 data tests - exit 0; 18 schema/adapter,
  immutable asset, archive safety, app-split, scalable dedup, manifest, shard,
  and end-to-end CLI fixture tests passed. The optional live revision test was
  not used as evidence for full data acquisition.
- **2026-08-13:** `python -m screen2action.tools.verify all` - exit 0; 146
  Python files formatted, Ruff and Mypy passed, and 105 offline tests passed
  with seven opt-in network/GPU tests deselected. Milestone 5 is fixture/offline
  complete; public-corpus execution remains unverified.
- **2026-08-13:** focused Milestone 6 tests - exit 0; 13 cache/orchestration,
  two-shard resume, transient retry, stale lock/temp recovery, raw-output,
  cache-only loader, CLI rank-discovery, and corruption-detection tests passed.
- **2026-08-13:** `python -m screen2action.tools.verify all` - exit 0; 150
  Python files formatted, Ruff and Mypy passed, and 110 offline tests passed
  with eight opt-in network/GPU tests deselected.
- **2026-08-13:** focused Milestone 7 training tests - exit 0; 19 engine,
  checkpoint, DDP, exact-resume, batch-probe, stage, semantic, and quantization
  tests passed. Ruff, Mypy, and formatting checks also passed.
- **2026-08-13:** `python -m screen2action.tools.verify all` - exit 0; 165
  Python files formatted, Ruff and Mypy passed, and 122 offline tests passed
  with eight opt-in network/GPU tests deselected.
- **2026-08-13:** focused Milestone 8 evaluation tests - exit 0; 18 direct
  evaluation, ScreenSpot protocol, calibration-lineage, report, sweep, CLI,
  and ablation-path tests passed. Ruff and Mypy also passed.
- **2026-08-13:** `python -m screen2action.tools.verify all` - exit 0; 172
  Python files formatted, Ruff and Mypy passed, and 128 offline tests passed
  with eight opt-in network/GPU tests deselected.
- **2026-08-13:** focused Milestone 9 tests - exit 0; 21 graph, reranker,
  visual-null, sparse-mask, semantics, and CLI tests passed; the two-test
  partition integration exported all four tiny subgraphs with CPU parity and
  confirmed Fast remains a separate unsupported package.
- **2026-08-13:** `python -m pytest tests/slow/test_paper_export_parity.py -m
  slow -q` - exit 0; paper-dimension graph/retention and six-layer
  command/retrieval/reranking ONNX parity passed on CPU with locally
  initialized architecture weights.
- **2026-08-13:** `python -m screen2action.tools.verify all` - exit 0; 177
  Python files formatted, Ruff and Mypy passed, and 137 offline tests passed
  with nine opt-in network/GPU/slow tests deselected.

## Remaining blockers and continuation

Current host is CPU-oriented and no public data/model license acceptance was
provided. Full downloads and GPU runs are intentionally excluded. The first
GPU-machine command is `screen2action doctor --device cuda --json`; subsequent
immutable-input commands are recorded in `docs/GPU_RUNBOOK.md`.

## Outcome summary

Implementation milestones 0-9 are complete under the documented offline and
fixture evidence boundaries. Final handoff publication and the review archive
are active; real public assets/data, CUDA training, Accurate locked export, and
mobile measurement remain external gates.
