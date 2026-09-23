# Deep-model region witness

The saved depth-5, 33-output models differ on a valid input even though their held-out predictions differ only at rounding scale. This is a concrete function difference, not merely a loose all-bin bound. The same input also exposes variation between two runs of the per-output baseline.

| Comparison | Output | Reference raw margin | Candidate raw margin | Binary64 absolute difference |
|---|---:|---:|---:|---:|
| Per-output a versus output-batch a (stream) | 11 | -0.10114017896017452 | -0.14774935652925669 | 0.046609177569082172 |
| Per-output a versus per-output b (stream) | 11 | -0.10114017896017452 | -0.14774935652925666 | 0.046609177569082144 |

Encoded feature bins: `1, 1, 1, 20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 18, 1`.

Raw feature values are retained in both JSON files. Every value is finite and exactly representable as float32; numeric values lie within [-1,1], and the categorical value is 0. All features are nonmissing. Independent CPU encoding of those raw values reproduces the bins for every compared model; independent traversal then sums the serialized leaves in model order. The JSON retains contributing leaves, exact fraction sums, binary64 sums, model hashes and helper-source hashes. No GPU workload was used.

The cross-policy pair changes 7 of 99 tree structures/splits. Baseline a versus b changes 6. These observations establish variation in the stored deep model functions; they do not establish its cause or its frequency on other data. Repeat variation cannot turn a cross-policy failure into a pass. Do not describe all model changes as harmless rounding or claim global functional equivalence.

- [Cross-policy witness](deep-region-witness.json)
- [Baseline-repeat witness](baseline-repeat-region-witness.json)
- [Complete metric and region audit](summary.json)
