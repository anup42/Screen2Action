# Export contracts

Export status is profile-specific. The repository already exports a
fixed-shape **tiny CPU reconstruction model** to ONNX and observes CPU
ONNXRuntime numerical parity in `tests/unit/test_export.py`. That is completed
CPU architecture evidence only; it is not paper-reference or mobile evidence.

The production boundary has four logical neural partitions:

1. perception;
2. graph encoder and retention;
3. command retrieval and relation reranking;
4. candidate grounder and action heads.

Host code owns image decoding, detector/OCR postprocessing, graph construction,
exact BudgetSelect and closure repair, reference remapping, crop extraction,
cache management, and final coordinate conversion. Accurate and Fast variants
remain separate artifacts. Each supported partition requires explicit fixed
maximum tensor shapes and numerical parity against PyTorch for its exact
configuration/model-lock digest.

The paper-reference partitions, quantized accuracy, mobile backend integration,
artifact size, and Galaxy latency remain **not implemented or unverified**.
Until Milestone 9 supplies those artifacts and reports, the tiny ONNX test must
not be described as production neural export or on-device support.
