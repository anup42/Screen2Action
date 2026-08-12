# Dataset runbook

The canonical data pipeline is offline-capable after source registration. Put
all real data outside the checkout by setting `SCREEN2ACTION_DATA_ROOT`; raw
source revisions are immutable, normalized images are content-addressed once,
and queryable metadata is stored in versioned Parquet partitions.

```text
$SCREEN2ACTION_DATA_ROOT/
  licenses/<source>/<acknowledgement>.json
  raw/<source>/<revision>/
  interim/<source>/<revision>/
  normalized/<dataset-version>/
    images/<sha256-prefix>/<sha256>.png
    screens/<source>/<revision>.parquet
    elements/<source>/<revision>.parquet
    commands/<source>/<revision>.parquet
    references/*.parquet
    audits/
    manifests/
  shards/<dataset-version>/
```

No dataset bytes, screenshots, manifests containing machine-local paths, or
license tokens belong in Git.

## Install and inspect

Install the data extra in the active CPU or GPU environment, then inspect the
pinned source registry and a no-write download plan:

```powershell
python -m pip install -e ".[data]"
screen2action data sources --json
screen2action data download wave_ui --sample 8 --accept-license --dry-run --json
```

`--accept-license` records a source/revision acknowledgement under the data
root. It confirms that the operator reviewed the upstream terms; it does not
change or reinterpret them.

## Acquire or register a source

Prefer a small sample before a full source:

```powershell
screen2action data download wave_ui --sample 8 --accept-license --json
```

To use files acquired through an upstream-specific process, register the
directory instead. Registration copies it into the immutable raw layout,
records every file size and SHA256, and refuses later mutation:

```powershell
screen2action data register-local wave_ui D:\licensed\wave-ui `
  --revision 21dc5aa1b39c038aea30f4572d80edef592b7989 `
  --accept-license --json
```

Re-running either command for an already verified revision is idempotent.
Interrupted network files retain a `.partial` suffix and use HTTP range resume
where the provider supports it. ZIP CRCs and member paths are checked before
extraction.

## Source-specific boundaries

| Source | Operational boundary |
| --- | --- |
| GUIAct / GUIEnv | Official Hugging Face commits are pinned. The current Hub/repository license labels conflict, so the registry preserves `NOASSERTION` and requires explicit review. Unsupported action semantics are rejected. |
| AMEX | Start with the official sample. Full multipart acquisition follows the source's documented `zip --fix` repair, verifies the merged ZIP, then extracts safely. Info-ZIP `zip` must be available for this full path. |
| AndroidControl | Install `.[android-control]` only when parsing the official GZIP TFRecords. TensorFlow is not a core dependency. Episode, step, action, and official split identity are preserved. |
| WaveUI | Each row retains its upstream `source`; the compilation's per-source terms must be reviewed. Rows without safe box/instruction mapping are rejected. |
| RICO Semantics | The semantic JSON release does not provide the screenshots or package identity by itself. Co-register licensed RICO PNGs and a `ui-app-map.json` mapping `screen_id` to canonical package ID. Rows lacking app identity are rejected so app-disjoint splitting remains meaningful. |
| ScreenSpot | Register only the official benchmark archive after reviewing incorporated-source terms. It is forced to `test` and hard-blocked from training, augmentation, calibration, threshold tuning, and model selection. |

Exact registered revisions and notes live in `configs/data/sources.yaml`.

## Normalize and audit

Normalize each accepted source into one dataset version. Coordinates cross the
boundary once and are stored as normalized `xyxy`; real points remain distinct
from pseudo-points derived from box centers.

```powershell
screen2action data normalize wave_ui `
  --revision 21dc5aa1b39c038aea30f4572d80edef592b7989 `
  --dataset-version public-v1 --json
screen2action data validate "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1" --json
screen2action data stats "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1" --json
```

Normalization emits exact accepted/rejected counts and reason codes. Missing
labels remain masked; unsupported actions are never relabeled as clicks.

Build app-disjoint splits before synthetic augmentation. An alias file, when
provided, is a JSON object from raw aliases to canonical app IDs.

```powershell
screen2action data split `
  --dataset "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1" `
  --output "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\audits\splits.json" `
  --alias-map D:\licensed\app-aliases.json --seed 7 --json
```

Synthetic records inherit the parent split. The split audit fails on app-ID
collisions and exact screen hashes crossing partitions.

Run staged deduplication next. Production grouping uses exact SHA256, a BK-tree
perceptual-hash candidate index, OCR-token Jaccard, and an optional frozen
embedding cosine gate. Union-find groups and deterministic survivors are
recorded in a complete removal audit.

```powershell
screen2action data dedup `
  --dataset "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1" `
  --split-plan "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\audits\splits.json" `
  --output "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\audits\dedup.json" `
  --config configs/data/dedup.yaml --json
```

Any exact or near ScreenSpot match in a train-eligible split is a hard error.

## Weak references, taxonomy, and frozen manifest

`public_weak_reference_v1` applies source references first, then deterministic
geometry, then high-confidence command phrases. Uncertain cases stay masked
and are included in the review JSON rather than forced into training.

```powershell
screen2action data annotate-references `
  --dataset "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1" `
  --output "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\references\public_weak_reference_v1.parquet" `
  --minimum-confidence 0.8 --json
```

Before freezing a training manifest, build and human-review the icon taxonomy;
training requires an immutable accepted 87-class taxonomy manifest.

```powershell
screen2action taxonomy icons inspect --input D:\licensed\rico-label-counts.json --json
screen2action taxonomy icons build --input D:\licensed\rico-label-counts.json `
  --output D:\review\icon-taxonomy-proposal.json --count 87 --json
screen2action taxonomy icons freeze --input D:\review\icon-taxonomy-proposal.json `
  --output D:\review\icon-taxonomy-87.json --count 87 `
  --accept-reconstruction --reviewer <name> --json
screen2action data build-manifest `
  --dataset "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1" `
  --split-plan "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\audits\splits.json" `
  --dedup-audit "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\audits\dedup.json" `
  --taxonomy D:\review\icon-taxonomy-87.json `
  --output "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\manifests\training.json" `
  --json
screen2action data validate `
  "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\manifests\training.json" --json
```

Manifest validation recomputes all referenced metadata/image digests and
leakage invariants. Move the manifest with its declared dataset tree; editing
or replacing a referenced file invalidates it.

Optional deterministic WebDataset-style shards store one image per unique
screen plus screen/element/command metadata:

```powershell
screen2action data build-shards `
  --manifest "$env:SCREEN2ACTION_DATA_ROOT\normalized\public-v1\manifests\training.json" `
  --output "$env:SCREEN2ACTION_DATA_ROOT\shards\public-v1" `
  --max-samples 2048 --json
```

## Validation layers

The default suite uses metadata-only fixtures and generated one-pixel images;
it never downloads public content:

```powershell
python -m pytest tests/unit/test_public_data_adapters.py `
  tests/unit/test_data_assets.py tests/unit/test_data_split_and_dedup.py `
  tests/integration/test_public_data_pipeline.py -q
```

The revision-resolution smoke is opt-in and networked:

```powershell
python -m pytest tests/optional/test_data_sources_network.py -m network -q
```

Successful fixtures establish adapter and audit behavior only. They do not
establish public-data quality, full-source coverage, training performance, or
paper reproduction.
