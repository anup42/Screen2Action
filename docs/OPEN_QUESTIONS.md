# Open questions

These questions prevent an exact reproduction claim. They are intentionally
kept visible so later experiments do not silently turn defaults into facts.

1. What detector input resolution, NMS threshold, and score threshold were
   used?
2. What exact UI node taxonomy and detector training labels were used?
3. What are the 87 icon class names, class sources, and unknown-class policy?
4. What tokenizer, vocabulary, special tokens, and vocabulary training corpus
   were used?
5. What exact SSB token grammar and per-field token costs were used?
6. What components make up `phi(b_i, b_j)` and the relation features?
7. How were reference nodes annotated for containment, proximity, and ordinal
   commands?
8. What target and calibration data define the confidence head?
9. What is the exact per-module width/depth split behind 185M parameters?
10. What quantization schedule and target mobile runtime produced the reported
    latency?
11. What are the exact ScreenSpot subset definitions for small and dense
    screens?
12. Which license governs GUIAct/GUIEnv while the official Hub cards and
    GUICourse repository advertise different identifiers?
13. What reviewed cross-source application alias map should be frozen for the
    full public corpus, especially where a source exposes only domain or task
    identity?
14. Which licensed RICO screenshot/package metadata release should be joined
    to RICO Semantics for production icon-taxonomy and app-disjoint use?
15. Did the paper jointly optimize detector-native losses with downstream
    losses, and if so what update ratio and optimizer state sharing were used?
16. Which exact CRNN alphabet, Unicode normalization, augmentation, and CTC
    blank-token convention were used for UI-specific OCR fine-tuning?
17. Which operations were quantized for the reported mobile model, including
    activation granularity, calibration observer, backend, and fallback policy?
18. What fixed maximum node/edge capacities and graph tensor layout did the
    paper's mobile runtime use, and which backend/operator versions were
    validated for each neural partition?

Until these are answered, reports must label results as reconstruction results.
