# Architecture decisions

## ADR-0001: normalized `xyxy` coordinates

**Status:** accepted.

The internal representation is `(x1, y1, x2, y2)` with each coordinate in
`[0, 1]`. Pixel `(x, y, w, h)` forms are converted at dataset/runtime
boundaries. This makes intersection, containment, and point tests unambiguous.

## ADR-0002: deterministic reconstruction codec

**Status:** accepted for the CPU milestone.

The supplied paper specifies an SSB budget but not its grammar. The first codec
uses a versioned integer-token grammar with a four-token graph header/footer,
fixed 56-token node records before text, variable-length text IDs, and eight
fixed five-token relation slots per node. Each relation slot stores relation
type, direction, remapped destination, and one geometry bucket. Fixed slots
make per-node costs independent of which optional references survive selection.
This is a testable reconstruction, not the paper's hidden grammar.

## ADR-0003: hierarchy and relation construction

**Status:** accepted for the CPU milestone.

Containment uses at least 80% child-area coverage and selects the smallest
strictly larger qualifying parent. A full-screen synthetic root is inserted
when needed. Proximity keeps at most one nearest neighbor in each cardinal
direction. Row and column ordinal groups use interval-overlap components with
an overlap ratio of at least 0.5 and stable geometric sorting.

## ADR-0004: selector closure repair

**Status:** accepted for the CPU milestone.

BudgetSelect first solves an exact independent-item 0-1 dynamic program over
fixed node costs after mandatory nodes. It then inserts parent and active
relation-reference dependencies. If closure exceeds the budget, optional
reference nodes are removed in ascending confidence order and their pointers
are remapped to null. Mandatory nodes and mandatory ancestry are never removed;
an unsatisfiable budget raises an explicit error.

## ADR-0005: Stage 2 grounding stabilization

**Status:** accepted for the CPU training workflow.

The stage contract trains selector, retrieval, and grounding heads together
while freezing the heavy crop encoder. A ground-truth positive is force
inserted with probability 0.5 during the first two Stage 2 epochs and then the
insertion rate is zero. The helper is deterministic at the schedule level and
leaves the random sampling policy to the future data loader.

## ADR-0006: CPU tokenizer and learned nulls

**Status:** accepted for the CPU milestone.

The default tokenizer is a deterministic Unicode case-folded vocabulary built
only from training text. It is a replaceable boundary for the paper's
unspecified tokenizer. Missing text, icon, and visual features use learned
null vectors in `NodeEncoder`; zeros are not treated as observed features.

## ADR-0007: relation attention geometry and export parity

**Status:** accepted for the CPU milestone.

Relation geometry is the 12-value vector emitted by `ssb.relations`: center
offset, center distance, log size ratios, IoU, directional coverage, and four
cardinal direction indicators. The variable edge-list GAT is the training
implementation. A fixed dense relation-ID/geometry tensor is converted to the
same equation for export parity; the parity test compares both paths using the
same layer weights.

## ADR-0008: crop margin and tiny visual encoder

**Status:** accepted for the CPU milestone.

The 12% crop setting is applied on each side as a fraction of candidate width
and height, then clipped to the normalized screen. The CPU crop encoder is a
small convolutional token adapter that emits exactly the configured `P`
tokens. It preserves the MobileViT-S interface and shape contract without
claiming MobileViT-S pretrained accuracy.

## ADR-0009: sparse candidate conditioning

**Status:** accepted.

Command tokens cross-attend to the flattened valid candidate groups. Each
candidate group is then reshaped into its own batch item and attends only to
the shared command stream plus its own node/crop tokens. This avoids a dense
cross-candidate attention matrix; `test_sparse_grounder.py` fixes the local
isolation invariant.

## ADR-0010: CPU export path

**Status:** accepted for the CPU milestone.

ONNX export uses the legacy TorchScript exporter with fast-path Transformer
kernels disabled during tracing because the installed CPU Torch build does not
export its fused Transformer operator. ONNXRuntime parity is measured on the
same fixed-shape tiny inputs. This is an export contract, not a mobile backend
claim.

## ADR-0011: Auditable CPU runtime logging

**Status:** accepted.

Runtime diagnostics use opt-in JSON events containing dimensions, counts,
selected IDs and action metadata, never screenshot pixels or raw OCR text.
The pipeline accepts an injected logger, while callers explicitly configure
the JSON handler when they need persistent diagnostics. This keeps tests and
default library use quiet without losing structured observability.

## ADR-0012: Offline duplicate-audit thresholds

**Status:** accepted for the dataset framework.

Duplicate auditing reports exact hashes first, then average-hash matches within
four differing bits, then OCR-token Jaccard matches. If an offline embedding
similarity map is supplied, OCR matches also require similarity at least 0.98.
The thresholds live in `configs/data/dedup.yaml`; no embedding model runs in
the default CPU test path.

## ADR-0013: Public reconstruction model bundle

**Status:** accepted for production integration.

The default public adapters are ScreenParser YOLO11-L, docTR
`crnn_vgg16_bn`, a TorchVision MobileNetV3-small shared icon/actionability/node
feature backbone, compact BERT `L-6_H-256_A-4`, and timm MobileViT-S. The
detector starts at 1280-pixel long edge, confidence 0.10, and NMS IoU 0.10.
These choices and thresholds come from the supplied reconstruction contract,
not the paper. Registry intent is mutable; only a resolved lock plus local
digests identifies an experiment's exact bundle.

## ADR-0014: Environment-rooted state and generated handoff

**Status:** accepted.

Real data, weights/caches, and runs use `SCREEN2ACTION_DATA_ROOT`,
`SCREEN2ACTION_CACHE_ROOT`, and `SCREEN2ACTION_RUN_ROOT`. The repository keeps
registries and generated handoff metadata but no external bytes. Handoff paths
are relative or environment-rooted, and host reporting excludes usernames,
absolute paths, environment values, and credentials.

## ADR-0015: Fidelity-tier traceability

**Status:** accepted.

Requirements are tracked independently across CPU architecture, production
integration, paper-reference training, and deployment/export. Tiny-model
shape/parity or fixture success may complete the first dimension while the
other dimensions remain partial or not started. This prevents interface tests
from becoming production or reproduction claims.

## ADR-0016: YAML as the sole core configuration dependency

**Status:** accepted.

PyYAML is the only core runtime dependency because configuration files are an
existing public repository contract and every operational command must compose,
expand, validate, and snapshot them consistently. PyTorch and all model/data,
training, AndroidControl, and export packages remain feature extras with lazy
imports. CUDA PyTorch is installed separately using the official selector.

## ADR-0017: Two-phase immutable model lock

**Status:** accepted.

Registry resolution freezes upstream Hub commits or package/weight-enum
versions and expected files without claiming local availability. License-gated
fetch then records observed size and SHA256 after each role, atomically updating
the same lock so interrupted bundles resume safely. A lock is experiment-ready
only when every selected role is complete and offline verification passes.
The MobileViT weight card's non-SPDX `other` license is preserved as
`LicenseRef-Apple-ML-CVNets`, not normalized to Apache-2.0.

## ADR-0018: Auditable production-perception reconstruction

**Status:** accepted.

ScreenParser's 55 public classes map through
`screenparser_55_to_ssb_v1`; original IDs, names, detector features, and
letterbox metadata remain separate from the coarse SSB type. The detector
supplies only a synthetic full-screen root plus leaf evidence. Optional aligned
child containers are explicitly labeled `reconstruction_policy` with a policy
version and confidence; they are not represented as model hierarchy output.

docTR uses locked `crnn_vgg16_bn` state with recognition-only crops normalized
by `doctr_crnn_rgb_h32_aspect_v1`. OCR may label its source node and one
smallest enclosing control; conflicts use highest confidence then source ID.
The MobileNetV3-small reconstruction shares one backbone across 87 icon logits,
four independent actionability logits, and a 256-D projection. Its custom heads
remain untrained until a reviewed taxonomy and public data are available.

Perception is command-independent and cached by screenshot, model-bundle,
perception-config, and schema digests. Cache files are sharded and atomically
replaced under cross-process locks. NumPy and Pillow are explicit perception
extras because the public input boundary supports their arrays/images and the
public adapters require those conversions; they are not core dependencies.

## ADR-0019: Trainable paper-profile variants and confidence reconstruction

**Status:** accepted.

The unified model names graph implementations `paper_eq_v1` and
`cpu_reconstruction_v1`. The paper variant uses shared query/key projections,
relation-specific values, relation embeddings, full 12-value geometry, and a
softmax within each destination/relation neighborhood. Edge-list and padded
dense paths share parameters and must remain numerically equivalent. Relation
reranking consumes destination state, neighbor state, geometry, and relation
embedding; neighbor gates and relation weights are separately normalized.

Node retention is computed before commands. Training uses straight-through
Gumbel masks plus target/reference survival and expected-budget losses;
inference uses the existing exact independent knapsack followed by
deterministic closure repair. Closure repair is not claimed to be a globally
optimal dependency-constrained knapsack.

Short OCR text shares compact-BERT WordPiece embeddings by default. An
independent embedding is an explicit ablation and requires pre-tokenized node
text. MobileViT's native final feature map is adaptively pooled to 12x12 and
projected to 256 dimensions; this projection is a public reconstruction.

The paper does not specify confidence labels. `grounding_correctness_v1`
creates a detached label only when both selected candidate and predicted
screen point are correct. Warm-up, refresh interval, and weight are configured;
the loss can be disabled and existing held-out temperature scaling used alone.
Long-press duration remains executor-defined. Scroll deltas and drag duration
are bounded model outputs.

## ADR-0020: Canonical public-data reconstruction

**Status:** accepted for production integration.

Canonical schema `2.0` adds source/revision/license, content digest, raw and
canonical app identity, split origin, annotation confidence/masks, matching,
trace, and synthetic-parent provenance with defaults that continue to load
legacy fixture records. Images are deterministically converted to RGB PNG and
content-addressed by SHA256; source metadata streams into per-source Parquet
partitions. Optional deterministic tar shards contain one image per screen.

Raw revisions are immutable inventories gated by a recorded license
acknowledgement. Archive extraction rejects absolute/traversing paths. Official
immutable source commits are registry intent, while local file hashes are the
actual acquisition evidence. GUIAct/GUIEnv and ScreenSpot retain
`NOASSERTION` where upstream terms are conflicting or source-dependent rather
than inventing a convenient license.

App identity is canonicalized before a deterministic app-disjoint split, and
synthetic records inherit their parent's partition. Production duplicate
grouping uses exact SHA256, a BK-tree perceptual-hash index, OCR Jaccard, an
optional frozen embedding gate, union-find, and a deterministic survivor.
ScreenSpot and its exact/near matches are prohibited outside evaluation.

Because the paper does not disclose reference annotation,
`public_weak_reference_v1` uses source evidence, then deterministic geometry,
then high-confidence relation phrases. It writes confidence, method, and masks;
uncertain labels have no reference loss. These are public reconstruction
choices, not paper-specified preprocessing.

## ADR-0021: Manifest-bound perception precomputation

**Status:** accepted for production integration.

Perception work is partitioned by the first 64 bits of the canonical screenshot
SHA256 modulo worker count. `torchrun` ranks or explicitly launched processes
therefore require no coordination collective: each process owns a model and a
disjoint deterministic shard, while short-lived file locks protect shared
atomic cache writes. Full SHA256 identities remain inside manifests and cache
keys. Directory names use verified 80-bit prefixes to remain below legacy
Windows path limits; opening a colliding prefix fails the immutable full-digest
manifest check.

Each bundle binds the exact model-lock digest, perception-config digest,
canonical/cache schema versions, and trained visual-head checkpoint digest.
Custom icon/actionability heads are required by default. A deterministic seeded
untrained-head mode is allowed only for plumbing smoke tests and is labeled in
CLI output and cache identity. Raw detector, OCR, icon/actionability, optional
detector ROI evidence, and final canonical graph state are cached; command
inputs/results are forbidden.

Entry writes, run progress, and final shard status use atomic replacement.
Completed shards validate every entry digest before resuming. Failed screens
carry bounded error text and monotonically increasing attempts; abandoned
short-lived locks and old atomic-write temporaries have scoped cache-only
recovery. Stage 2 loads completed cache entries through a module that does not
import or instantiate ScreenParser or docTR.

## ADR-0022: Exact-resume distributed stage runner

**Status:** accepted for the public reconstruction.

Stage training is launched through `torchrun` and uses Gloo on CPU or NCCL on
CUDA. Each rank owns a deterministically seeded, same-screen batch shard. DDP
gradient synchronization is deferred with `no_sync` during accumulation, and
the last partial accumulation window is scaled and flushed instead of dropped.
Validation totals, loss components, and example counts are reduced directly;
rank zero alone writes manifests, metrics, and atomic latest/best checkpoints.

A resume checkpoint binds config, data manifest, model lock, perception cache,
and run-manifest digests. It records the next batch, optimizer/scheduler/scaler
state, sampler epoch, and every rank's Python, Torch, CUDA, and optional NumPy
RNG state. A mismatched lineage fails explicitly. Model-only initialization is
separate from exact resume so stage transitions cannot be mistaken for a
continuation of optimizer or sampler state.

## ADR-0023: Native perception training boundaries

**Status:** accepted for the public reconstruction.

The default Stage 1 path trains source-supervised MobileNetV3 icon and
actionability heads, node projection, and relation GAT from tight source crops.
Optional detector and OCR modes delegate to Ultralytics-native YOLO losses and
docTR-native CRNN CTC loss, respectively, and keep separate checkpoint
lineages. Stage 1 full is an orchestrator over those independently auditable
sub-stages.

Stage 2 consumes frozen command-independent perception caches. Stage 3 adds a
weighted differentiable semantic branch while OCR remains frozen. Optional
detector fine-tuning alternates a detector-native update branch with downstream
updates. No downstream gradient is claimed through detector proposal/NMS or
OCR text decoding. A trained Stage 1 visual-head artifact is loaded explicitly
by perception precomputation rather than inferred from a general checkpoint.

## ADR-0024: Bounded Stage 4 quantization reconstruction

**Status:** accepted for the optional CPU/GPU training interface.

Stage 4 inserts symmetric int8 fake quantization with per-output-channel
weights for supported linear and convolution layers. It observes activation
ranges on training-only representative batches, inventories unsupported
subgraphs, and compares FP32, weight-only PTQ, and fake-quant QAT on the same
checkpoint and validation batches. Multi-head attention and transformer
container internals remain explicit unsupported boundaries when direct module
replacement would alter their public parameter access contract.

The activation observations are evidence and calibration metadata, not a
claim that an integer activation runtime was exported. PTQ is labeled
weight-only, and no mobile latency/runtime claim is permitted until a named
backend and device have been measured.
