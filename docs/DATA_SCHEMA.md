# Canonical data schema

The dataclasses in `src/screen2action/data/schema.py` are the canonical
boundary for adapters and runtime code.

## Screen and command

- `ScreenRecord`: screen ID, app ID, image path, dimensions, split, source, and
  annotated elements.
- `CommandRecord`: command text, action type, normalized target box/point,
  action parameters, relation type, and reference element IDs.

## Perceived node

`NodeRecord` stores node type, normalized box, text and token IDs, icon
probabilities, ROI features, four actionability logits, hierarchy metadata,
confidence values, and an optional non-serialized retention score.

Actionability labels are independent. Unknown labels are represented by masks
at training boundaries rather than as negative labels.

## Graph edge

`EdgeRecord` is directed and stores source/destination IDs, relation type,
direction, relative geometry, and an optional row/column axis for ordinal
relations. Graph validation rejects self-edges, missing endpoints, non-finite
geometry, and invalid relation/direction combinations.
