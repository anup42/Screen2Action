# Evaluation and calibration runbook

Evaluation requires a frozen data manifest, compatible completed perception
cache, and checkpoint from the same lineage. ScreenSpot is test-only.

```text
screen2action evaluate \
  --config configs/experiments/screenspot_eval.yaml \
  --checkpoint ${SCREEN2ACTION_RUN_ROOT}/stage3-joint/checkpoints/best.pt \
  --manifest ${SCREEN2ACTION_DATA_ROOT}/normalized/public-v1/manifests/training.json \
  --cache-manifest ${SCREEN2ACTION_CACHE_ROOT}/perception/<bundle>/manifest.json \
  --model-lock configs/models/lock.json \
  --output ${SCREEN2ACTION_RUN_ROOT}/evaluation/screenspot \
  --device cuda --json
```

The output contains `metrics.json`, `predictions.csv`,
`predictions.parquet`, `report.md`, `resolved-config.json`, and
`provenance.json`. Point-in-target accuracy is computed directly per example;
the report does not multiply cascade stages.

Fit confidence only on a separate held-out validation evaluation from
non-ScreenSpot sources:

```text
screen2action evaluate \
  --config configs/experiments/public_smoke_real_models.yaml \
  --split val --checkpoint <same-checkpoint> --manifest <manifest> \
  --cache-manifest <cache-manifest> --output <validation-output> --json

screen2action calibrate \
  --predictions <validation-output>/predictions.parquet \
  --output <calibration-output> --target-risk 0.10 --json
```

The temperature and threshold-policy files are separate and bind the
checkpoint SHA256, validation-manifest digest, and prediction-file SHA256.
Apply them to test evaluation with `--calibration <calibration-output>`.

Run the required B/K and ablation matrix with:

```text
screen2action evaluate \
  --sweep --config configs/experiments/budget_k_sweep.yaml \
  --checkpoint <baseline-checkpoint> --manifest <manifest> \
  --cache-manifest <cache-manifest> --output <sweep-output> --json
```

Training-loss and QAT ablations require their own checkpoint paths under
`evaluation.checkpoint_variants`. Missing checkpoint lineages and the optional,
unimplemented ROIAlign Fast path are reported explicitly; baseline metrics are
not relabeled. Every completed variant retains its own complete artifact set.
