# Export and host-runtime contracts

`screen2action export` creates an immutable, profile-namespaced ONNX package.
Every emitted neural artifact is checked by ONNX, executed with ONNXRuntime on
CPU, and compared directly with the source PyTorch module. A failed partition
is recorded in `partition-report.json`; no alternate implementation is
silently substituted.

## Ownership boundary

The host retains image decoding and normalization, detector NMS, OCR decoding,
canonical graph construction, exact BudgetSelect and closure repair,
actionability filtering and fixed-K identity selection, crop expansion and
extraction, cache management, and candidate-local to screen-coordinate
conversion. Internal and boundary boxes remain normalized `xyxy`.

The four neural packages are:

1. MobileNetV3 icon logits, four independent actionability logits, and visual
   node features;
2. multimodal node fusion, fixed-grid relation GAT, and command-independent
   retention logits/scores;
3. command token/CLS encoding, cosine retrieval, and relation reranking;
4. candidate crop encoding, sparse grounding, typed action parameters, and
   confidence.

Graph inputs use fixed `[batch,relation,max_nodes,max_nodes]` boolean masks and
matching geometry grids with `[source,destination]` orientation. The relation
axis preserves multiple typed edges for the same node pair. The host rejects
graphs beyond `max_nodes` or `max_edges`; the neural graph never performs
data-dependent `nonzero` compaction. Exact knapsack and top-K node IDs remain
host operations.

## Profiles and evidence

- `tiny_cpu` exports all four reconstruction partitions when both a downstream
  checkpoint and a trained Stage-1 visual checkpoint are supplied. The
  complete package path is covered by an offline integration test.
- `accurate` builds the locked compact-BERT, MobileViT-S, and MobileNetV3
  modules and requires matching downstream, visual, and model-lock digests.
  It performs parity at export time. Public locked assets/checkpoints have not
  been supplied on this host, so no real Accurate package is claimed.
- `fast_roi` is a distinct namespace. Detector ROIAlign crop tokens and a
  separately trained Fast checkpoint topology are not implemented. The
  command emits an unsupported partition report and exits nonzero; it never
  relabels Accurate artifacts or metrics.

The opt-in slow test exports paper-dimension `d=256`, two-layer
`paper_eq_v1` graph/retention and six-layer command/retrieval/reranking
partitions with locally initialized weights. This proves shape/equation ONNX
parity, not parity for unavailable pretrained weights or a mobile backend.

## Commands

Dry-run the selected profile without loading a checkpoint:

```text
screen2action export --profile tiny_cpu --dry-run --json
screen2action export --profile accurate --dry-run --json
```

Export a tiny trained package:

```text
screen2action export --profile tiny_cpu \
  --checkpoint ${SCREEN2ACTION_RUN_ROOT}/stage3/checkpoints/best.pt \
  --visual-checkpoint ${SCREEN2ACTION_RUN_ROOT}/stage1/checkpoints/visual-model.pt \
  --output ${SCREEN2ACTION_RUN_ROOT}/exports --json
```

For Accurate, add `--model-lock configs/models/lock.json` and ensure
`SCREEN2ACTION_CACHE_ROOT` contains the verified lock assets. Output is written
to `<output>/<profile>/`; an existing profile directory is never overwritten.

Each package contains:

- one `.onnx` and `.parity.json` per successful partition;
- `shape-contract.json`;
- `host-runtime-contract.json`;
- `resolved-config.yaml`;
- `partition-report.json`, including SHA256, bytes, exact I/O shapes/dtypes,
  parameter count, tolerance, errors, code identity, checkpoint lineage, and
  unsupported reasons.

Validation layers:

```text
python -m pytest tests/integration/test_partitioned_export.py -q
python -m pytest tests/slow/test_paper_export_parity.py -m slow -q
```

ONNX is currently an interchange/runtime-validation target, not evidence of a
mobile deployment. Quantized accuracy, artifact size against the paper, and
Galaxy latency remain unmeasured.
