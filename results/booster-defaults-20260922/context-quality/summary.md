# Default-context quality audit

Status: **regression**; allowance remains zero.

| Candidate | Strict gate | Regressed metrics | Maximum prediction difference | Decision changes |
|---|---|---:|---:|---:|
| context-scalar-candidate-a | pass | 0 | 2.2204460492503131e-16 | not applicable |
| context-scalar-candidate-b | pass | 0 | 2.2204460492503131e-16 | not applicable |
| context-129-candidate-a | regression | 29 | 4.9960036108132044e-16 | not applicable |
| context-129-candidate-b | regression | 28 | 4.9960036108132044e-16 | not applicable |
| context-1024-candidate-a | regression | 240 | 2.2204460492503131e-16 | 0 |
| context-1024-candidate-b | regression | 235 | 2.7755575615628914e-16 | 0 |
| context-4096-candidate-a | regression | 441 | 2.7755575615628914e-16 | 0 |
| context-4096-candidate-b | regression | 493 | 2.7755575615628914e-16 | 0 |
| context-multiclass17-candidate-a | pass | 0 | 8.3266726846886741e-17 | 0 |
| context-multiclass17-candidate-b | pass | 0 | 1.1102230246251565e-16 | 0 |
| context-fallback33-candidate-a | regression | 6 | 1.6653345369377348e-16 | not applicable |
| context-fallback33-candidate-b | regression | 7 | 1.6653345369377348e-16 | not applicable |

Maximum positive deterioration by metric unit:

- nats: 2.9999999999999999e-16
- proportion: 0
- squared_probability: 1.2e-16
- target_units: 2e-16

All 24 captures are checked against the same frozen executable, exact declared shapes/seeds/policies, serialized model dimensions, and the unchanged evaluator. Paired target hashes and feature metadata must match. These are held-out synthetic results; prior strict failures remain unchanged.
