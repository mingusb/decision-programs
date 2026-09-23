# Optimized resident quality audit

Status: **regression**. Zero allowance is retained for every applicable metric.

## Optimized versus matching hybrid

| Candidate | Gate | Regressed metrics | Maximum prediction difference | Decision changes | All-bin raw-margin bound |
|---|---|---:|---:|---:|---:|
| optimized-scalar-stream-a | pass | 0 | 2.2204460492503131e-16 | N/A | 3.677613769070831e-16 |
| optimized-scalar-graph-a | pass | 0 | 2.2204460492503131e-16 | N/A | 3.8857805861880479e-16 |
| optimized-scalar-stream-b | pass | 0 | 2.2204460492503131e-16 | N/A | 4.0245584642661925e-16 |
| optimized-scalar-graph-b | pass | 0 | 2.2204460492503131e-16 | N/A | 3.8163916471489756e-16 |
| optimized-129-graph-batch0-a | regression | 38 | 4.9960036108132044e-16 | N/A | 6.3317406873153459e-16 |
| optimized-129-graph-batch16-a | regression | 30 | 4.9960036108132044e-16 | N/A | 6.609296443471635e-16 |
| optimized-129-graph-batch0-b | regression | 34 | 3.8857805861880479e-16 | N/A | 6.8695049648681561e-16 |
| optimized-129-graph-batch16-b | regression | 45 | 4.9960036108132044e-16 | N/A | 6.1929628092372013e-16 |
| optimized-129-graph-batch1 | regression | 44 | 5.8286708792820718e-16 | N/A | 6.591949208711867e-16 |
| optimized-129-stream-batch16 | regression | 39 | 4.4408920985006262e-16 | N/A | 5.9154070530809122e-16 |
| optimized-1024-graph-batch0-a | regression | 263 | 3.8857805861880479e-16 | 0 | 1.429412144204889e-15 |
| optimized-1024-graph-batch16-a | regression | 292 | 2.7755575615628914e-16 | 0 | 1.6358442378461291e-15 |
| optimized-1024-graph-batch0-b | regression | 277 | 2.7755575615628914e-16 | 0 | 2.0712598303163077e-15 |
| optimized-1024-graph-batch16-b | regression | 293 | 2.7755575615628914e-16 | 0 | 1.5439038936193583e-15 |
| optimized-1024-graph-batch1 | regression | 258 | 2.7755575615628914e-16 | 0 | 1.5404344466674047e-15 |
| optimized-1024-stream-batch16 | regression | 267 | 2.7755575615628914e-16 | 0 | 1.609823385706477e-15 |
| optimized-4096-graph-a | regression | 663 | 2.2204460492503131e-16 | 0 | 8.3266726846886741e-16 |
| optimized-4096-graph-b | regression | 638 | 2.2204460492503131e-16 | 0 | 1.0547118733938987e-15 |
| confirm-129-graph-batch16 | regression | 43 | 4.9960036108132044e-16 | N/A | 5.8286708792820718e-16 |

## Same-binary batch16 versus compact0

| Candidate | Gate | Regressed metrics | Maximum prediction difference | Decision changes | All-bin raw-margin bound |
|---|---|---:|---:|---:|---:|
| optimized-129-graph-batch16-a | regression | 21 | 3.8857805861880479e-16 | N/A | 5.6551985316843911e-16 |
| optimized-129-graph-batch16-b | regression | 43 | 5.2735593669694936e-16 | N/A | 6.106226635438361e-16 |
| optimized-1024-graph-batch16-a | regression | 260 | 3.8857805861880479e-16 | 0 | 1.56472057533108e-15 |
| optimized-1024-graph-batch16-b | regression | 240 | 3.3306690738754696e-16 | 0 | 1.1761425167122752e-15 |

Model-versus-base: 26/26 trained evidence cases passed.

Preparation-only exact feature metadata checks:

- optimized-quantize-1048576x8-radix8: pass against quantize-1048576x8-hybrid.
- optimized-quantize-1048576x8-radix4: pass against quantize-1048576x8-hybrid.
- optimized-quantize-16777216x1-radix8: pass against quantize-16777216x1-hybrid.
- optimized-quantize-16777216x1-radix4: pass against quantize-16777216x1-hybrid.

- Every applicable quality metric and every output receives zero allowance. Last-decimal failures remain failed.
- Preparation-only rounds0 cases are checked for evidence integrity and exact feature metadata, and excluded from learned-model promotion gates.
- The four batch16-versus-compact0 comparisons require identical executable hashes and all training/quantization settings except the intended export-batch setting.
- First-hybrid references for outputs129/1024 are retained historical executions; same-binary export comparisons isolate that change more closely.
- Synthetic held-out results do not establish NLP performance or generalization to other datasets. Large output counts do not increase independent held-out rows.
- All-bin region comparisons cover missing and all retained present bins. Their exact-real raw-margin bounds exclude additional runtime arithmetic rounding.
