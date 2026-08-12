# Dataset runbook

Set `SCREEN2ACTION_DATA_ROOT` outside the checkout for real data. Raw source
revisions are immutable. Normalized images are content-addressed once and
metadata is versioned separately.

```text
$SCREEN2ACTION_DATA_ROOT/
  raw/<source>/<revision>/
  interim/<source>/<revision>/
  normalized/<dataset-version>/
    images/<sha256-prefix>/<sha256>.png
    screens/ elements/ commands/ references/
    audits/ manifests/
  shards/<dataset-version>/
```

## Safe workflow

1. Inspect `screen2action data sources` and upstream terms.
2. Download with explicit `--accept-license`, a pinned revision, and preferably
   `--sample N`; or register immutable pre-downloaded files.
3. Normalize boundary coordinates once, preserving source IDs, masks,
   platform/app identity, episode/step identity, and rejected reason counts.
4. Create app-disjoint splits before augmentation.
5. Run exact/perceptual/OCR/optional-embedding dedup and enforce the ScreenSpot
   evaluation-only leakage guard.
6. Add only confident `public_weak_reference_v1` labels; uncertain examples
   retain masks and incur no reference loss.
7. Freeze a manifest, verify every referenced digest, then optionally build
   WebDataset-style shards.

Never force unsupported actions into click labels, infer a real click point
from a box without marking it pseudo-derived, or copy one screen image for
each command. AMEX multipart archives require verified assembly; AndroidControl
uses optional TFRecord tooling and preserves official split identity.
