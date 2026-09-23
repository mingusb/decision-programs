# Resident campaign quality audit

Status: **regression**. All comparisons use zero allowance.

| Resident case | Matching hybrid | Quality gate | Differing prediction values | Max absolute difference | Classification decision changes | Changed split thresholds |
|---|---|---|---:|---:|---:|---:|
| scalar-stream8-a | scalar-hybrid-a | pass (0 regressed metrics) | 2910 | 3.3306690738754696e-16 | N/A | 0 |
| scalar-graph8-a | scalar-hybrid-a | regression (4 regressed metrics) | 2967 | 2.2204460492503131e-16 | N/A | 0 |
| scalar-graph4-a | scalar-hybrid-a | pass (0 regressed metrics) | 3666 | 2.2204460492503131e-16 | N/A | 0 |
| scalar-stream8-b | scalar-hybrid-b | pass (0 regressed metrics) | 3706 | 2.2204460492503131e-16 | N/A | 0 |
| scalar-graph8-b | scalar-hybrid-b | pass (0 regressed metrics) | 2732 | 2.2204460492503131e-16 | N/A | 0 |
| scalar-graph4-b | scalar-hybrid-b | regression (4 regressed metrics) | 3192 | 2.2204460492503131e-16 | N/A | 0 |
| binary-resident | binary-hybrid | pass (0 regressed metrics) | 5 | 1.1102230246251565e-16 | 0 | 0 |
| multiclass-resident | multiclass-hybrid | pass (0 regressed metrics) | 5054 | 2.2204460492503131e-16 | 0 | 0 |
| outputs129-graph | outputs129-hybrid | regression (39 regressed metrics) | 21016 | 3.8857805861880479e-16 | N/A | 0 |
| outputs1024-graph | outputs1024-hybrid | regression (278 regressed metrics) | 49388 | 3.3306690738754696e-16 | 0 | 77 |
| outputs4096-graph | outputs4096-hybrid | regression (619 regressed metrics) | 31950 | 2.7755575615628914e-16 | 0 | 166 |
| outputs1024-stream | outputs1024-hybrid | regression (274 regressed metrics) | 49382 | 3.3306690738754696e-16 | 0 | 68 |

Model against its own constant base scores:

| Case | Zero-allowance gate | Regressed metrics |
|---|---|---:|
| binary-hybrid | pass | 0 |
| binary-resident | pass | 0 |
| multiclass-hybrid | pass | 0 |
| multiclass-resident | pass | 0 |
| outputs1024-graph | pass | 0 |
| outputs1024-hybrid | pass | 0 |
| outputs1024-stream | pass | 0 |
| outputs129-graph | pass | 0 |
| outputs129-hybrid | pass | 0 |
| outputs4096-graph | pass | 0 |
| outputs4096-hybrid | pass | 0 |
| scalar-graph4-a | pass | 0 |
| scalar-graph4-b | pass | 0 |
| scalar-graph8-a | pass | 0 |
| scalar-graph8-b | pass | 0 |
| scalar-hybrid-a | pass | 0 |
| scalar-hybrid-b | pass | 0 |
| scalar-stream8-a | pass | 0 |
| scalar-stream8-b | pass | 0 |

Threshold counts above compare matching-feature split thresholds at matching structural paths. Feature, missing-direction, topology, leaf and base-score changes are retained separately in JSON.

- Zero allowance applies independently to every applicable metric, including every output; failures remain failures, even at the last decimal place.
- These deterministic synthetic held-out datasets are not evidence of NLP quality or accuracy on another dataset.
- 64 held-out rows in the 4096-output case remain 64 independent examples, not 262144 examples.
- Exact target bytes, row identity, settings, and source hashes are required for comparisons.
- Prediction comparisons use exact parsed binary64 values. Classification decision changes use >=0.5 or lowest-index argmax; regression has no classification threshold.
- Model node comparisons align structural paths, not incidental node-array ordering. Different split features are not counted as same-feature threshold changes.
- A pass on all reported metrics does not prove mathematical equivalence or preservation of quality on unobserved data.
