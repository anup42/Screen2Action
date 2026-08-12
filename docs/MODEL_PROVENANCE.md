# Model provenance

`configs/models/registry.yaml` names logical roles and mutable upstream
identifiers. It does not prove that a weight exists locally. A generated
`configs/models/lock.json` records immutable resolution evidence: revision or
weight-enum/package version, expected files, local SHA256, license ID/source
note, preprocessing/output contracts, and reconstruction caveats.

## Locked public reconstruction

| Role | Public default | Status before lock resolution |
| --- | --- | --- |
| UI detector | `docling-project/ScreenParser`, YOLO11-L | interface intent only |
| OCR | docTR `crnn_vgg16_bn`, recognition-only | interface intent only |
| Icon/action/node feature | TorchVision MobileNetV3-small | backbone intent; heads untrained |
| Command | `google/bert_uncased_L-6_H-256_A-4` | interface intent only |
| Crop | `timm/mobilevit_s.cvnets_in1k` | interface intent only |

No model is downloaded implicitly by import, test, `doctor`, or config
validation. Asset resolution is explicit and license-gated. `models verify`
must remain offline and compare every locked file digest. Weights remain under
`${SCREEN2ACTION_CACHE_ROOT}`, never under Git.

## Resolve, fetch, verify

```text
screen2action models list --json
screen2action models resolve-lock --lock configs/models/lock.json --json
screen2action models fetch --lock configs/models/lock.json \
  --accept-license all --json
screen2action models verify --lock configs/models/lock.json --json
```

Use `--accept-license <role-or-license-id>` instead of `all` when recording
individual review. Resolution pins Hugging Face commits and expected files, or
the exact package version plus TorchVision enum/docTR factory. Fetch checkpoints
the updated lock after each role, records local size and SHA256, and reuses
provider caches/partial downloads. A changed registry cannot silently replace
an existing lock. Offline verify imports no model framework and performs no
network call.

The current ScreenParser model card reports Apache-2.0, YOLO11-L, 55 classes,
1280-pixel training/inference, and a mutable `main` that moved to ScreenParse
v2 in May 2026; the resolver therefore records the exact Hub commit. The
compact BERT card reports Apache-2.0. The MobileViT card reports `other` and
links Apple's ml-cvnets license, represented as
`LicenseRef-Apple-ML-CVNets` rather than being mislabeled Apache-2.0.

Provider API references:

- <https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api>
- <https://huggingface.co/docs/huggingface_hub/en/guides/download>
- <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html>
- <https://mindee.github.io/doctr/latest/modules/models.html>

The tiny transformer/CNN modules used by offline tests are reconstruction test
models, not substitutes for public-model accuracy. A successful mocked or tiny
test establishes API/shape behavior only. Real-model smoke evidence is
reported separately with lock digest, dependency versions, device, and command.
