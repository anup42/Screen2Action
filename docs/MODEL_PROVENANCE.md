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

The tiny transformer/CNN modules used by offline tests are reconstruction test
models, not substitutes for public-model accuracy. A successful mocked or tiny
test establishes API/shape behavior only. Real-model smoke evidence is
reported separately with lock digest, dependency versions, device, and command.
