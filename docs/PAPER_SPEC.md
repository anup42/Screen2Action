# Paper specification

This document is the executable boundary between facts stated in the supplied
Screen2Action plan/paper and decisions made to make the system runnable. A
statement marked **paper** is a contract. A statement marked **reconstruction**
must not be used to claim exact numerical reproduction.

## Paper-specified pipeline

### Frame-level, command-independent path

1. YOLO11-L detects UI regions.
2. CRNN OCR recognizes visible text.
3. MobileNetV3-small classifies icon crops into 87 semantic classes.
4. Click, drag, scroll, and long-press actionability are predicted.
5. Typed nodes contain text, icon semantics, visual features, normalized
   geometry, hierarchy depth, actionability, and confidence.
6. Containment, proximity, and ordinal relations are constructed.
7. Two relation-aware graph layers are applied.
8. The graph is serialized under `B = 512` SSB tokens and cached per frame.

### Command-dependent path

1. A six-layer command transformer encodes up to 64 command tokens.
2. Cosine similarity retrieves node candidates.
3. Relation-aware reranking uses validation-selected `lambda = 0.30`.
4. The top `K = 8` actionable candidates are retained.
5. Candidate boxes are expanded by 12%, clipped, and resized to `192 x 192`.
6. Crops are encoded in one batched MobileViT-S pass.
7. Each candidate contributes `P = 144` visual tokens at `d = 256`.
8. Two sparse cross-attention blocks use four heads.
9. Candidate, point, action, action-parameter, and confidence outputs are
   produced.

### Training constants

| Item | Contract |
| --- | --- |
| Stage 1 | perception and graph, 12 epochs |
| Stage 2 | selector and retrieval, 8 epochs |
| Stage 3 | joint non-OCR fine-tuning, 6 epochs |
| Optimizer | AdamW |
| Global batch | 256 |
| New-head LR | `2e-4` |
| Pretrained LR | `2e-5` |
| Warm-up | 5% |
| Weight decay | 0.05 |
| Gradient clipping | 1.0 |
| Objective weights | candidate 1.0, point 1.0, action 0.4, UI contrastive 0.2, survival 0.5, budget 0.05 |
| InfoNCE temperature | 0.07 |
| Selector temperature | 1.0 to 0.1 over first 80% of selector training |

## Reference metrics

The following are diagnostic targets from the plan, not pass/fail criteria for
this reconstruction: proposal recall 98.7%, target survival 98.3%, relational
reference survival 95.2%, R@1/R@4/R@8 of 82.6/94.2/96.1%, conditional candidate
selection accuracy 86.7%, conditional point accuracy 96.4%, end-to-end point
accuracy 82.1%, and public ScreenSpot 80.2%.

## Frozen internal conventions

- Boxes are normalized `xyxy` tuples internally.
- Coordinates are clipped to `[0, 1]` at input boundaries.
- Coordinate quantization uses `round(clip(x, 0, 1) * 1023)`.
- Actionability is four independent labels, not a mutually exclusive class.
- Missing supervision is masked rather than converted into a negative label.

## Not specified by the paper

The detector resolution and thresholds, node taxonomy details, icon manifest,
tokenizer, SSB grammar, relation geometry vector, reference-label procedure,
confidence target, quantization schedule, and mobile backend remain open. The
current defaults are recorded in `docs/DECISIONS.md` and
`docs/OPEN_QUESTIONS.md`.

## CPU milestone implementation contract

The CPU milestone uses oracle/cached perception and keeps the interfaces for
the later YOLO11-L, CRNN, MobileNetV3-small, and MobileViT-S components. Its
tiny profile is `d=64`, one graph layer, two command layers, `K=4`, `P=16`,
64-pixel crops, and a reconstruction SSB budget of 256 so the
fixed test grammar can retain mandatory root/scroll nodes. The paper-reference
profile is separately exposed with `d=256`, two graph layers, six command
layers, `K=8`, `P=144`, `192 x 192` crops, and `B=512`.

The CPU runtime contract is:

```text
prepare_frame(screenshot, oracle_or_cached_nodes)
    -> graph + SSB + command-independent selection + cached embeddings
ground(frame_state, command)
    -> fixed-K retrieval + batched crops + sparse grounding + action/confidence
```

This contract is tested end to end without network access or raw datasets.
