# Synthetic level-batching audit

Status: **regression**. Every applicable quality metric retains zero allowance.

| Shape | Submission | Per-output training ms (a, b) | Output-batch training ms (a, b) | Median training speedup | Median total-training speedup |
|---|---|---|---|---:|---:|
| scalar | stream | 27.7675, 26.9767 | 28.0800, 26.3327 | 1.0061× | 1.0778× |
| scalar | graph | 29.0994, 28.9880 | 28.3150, 29.7558 | 1.0003× | 1.0288× |
| 33deep | stream | 54.4194, 53.7780 | 21.8914, 22.2027 | 2.4538× | 1.8675× |
| 33deep | graph | 47.3411, 50.4240 | 29.4878, 25.1110 | 1.7906× | 1.4348× |
| 129 | stream | 110.9638, 96.1173 | 30.5011, 41.3495 | 2.8821× | 2.3495× |
| 129 | graph | 79.3520, 82.4439 | 36.8354, 30.6484 | 2.3975× | 2.0334× |
| 1024 | stream | 639.3920, 570.4794 | 159.6217, 167.8718 | 3.6943× | 3.3230× |
| 1024 | graph | 490.9750, 461.8252 | 170.8938, 173.0886 | 2.7699× | 2.5113× |
| 4096 | stream | 992.7264, 1088.0745 | 278.9633, 291.8511 | 3.6453× | 3.1506× |
| 4096 | graph | 915.1256, 864.6359 | 258.0981, 274.9495 | 3.3388× | 2.8651× |
| multiclass17 | stream | 18.8021, 17.2511 | 6.2752, 5.7303 | 3.0031× | 1.9312× |
| multiclass17 | graph | 16.2886, 18.4477 | 10.5917, 8.3811 | 1.8308× | 1.3691× |
| fallback33 | stream | 30.9191, 29.9464 | 20.0572, 24.8521 | 1.3553× | 1.2679× |
| fallback33 | graph | 31.7671, 28.9233 | 23.6793, 21.8045 | 1.3343× | 1.2547× |

## Cross-policy gates

| Candidate | Gate | Regressed metrics | Max saved prediction delta | Decision changes | Max independent raw-margin delta | All-bin margin bound |
|---|---|---:|---:|---:|---:|---:|
| synthetic-scalar-stream-output-batch-a | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 5.4817261840867104e-16 |
| synthetic-scalar-stream-output-batch-b | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 4.3715031594615539e-16 |
| synthetic-scalar-graph-output-batch-a | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 8.1532003370909933e-16 |
| synthetic-scalar-graph-output-batch-b | pass | 0 | 3.3306690738754696e-16 | None | 3.3306690738754696e-16 | 1.0920084281274001e-15 |
| synthetic-33deep-stream-output-batch-a | regression | 1 | 2.7755575615628914e-16 | None | 2.7755575615628914e-16 | 0.088984272576618034 |
| synthetic-33deep-stream-output-batch-b | regression | 4 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 0.042375095007535883 |
| synthetic-33deep-graph-output-batch-a | regression | 2 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 0.08898427257661802 |
| synthetic-33deep-graph-output-batch-b | regression | 3 | 2.7755575615628914e-16 | None | 2.7755575615628914e-16 | 0.046609177569082297 |
| synthetic-129-stream-output-batch-a | regression | 34 | 4.9960036108132044e-16 | None | 4.9960036108132044e-16 | 1.0234868508263162e-15 |
| synthetic-129-stream-output-batch-b | regression | 27 | 5.2735593669694936e-16 | None | 5.2735593669694936e-16 | 6.106226635438361e-16 |
| synthetic-129-graph-output-batch-a | regression | 34 | 5.5511151231257827e-16 | None | 5.5511151231257827e-16 | 6.2450045135165055e-16 |
| synthetic-129-graph-output-batch-b | regression | 34 | 3.8857805861880479e-16 | None | 3.8857805861880479e-16 | 5.2974118147641747e-16 |
| synthetic-1024-stream-output-batch-a | regression | 259 | 2.7755575615628914e-16 | 0 | 1.2212453270876722e-15 | 1.3600232051658168e-15 |
| synthetic-1024-stream-output-batch-b | regression | 245 | 2.2204460492503131e-16 | 0 | 9.9920072216264089e-16 | 1.3877787807814457e-15 |
| synthetic-1024-graph-output-batch-a | regression | 243 | 3.3306690738754696e-16 | 0 | 1.3322676295501878e-15 | 1.2993078835066285e-15 |
| synthetic-1024-graph-output-batch-b | regression | 251 | 3.3306690738754696e-16 | 0 | 1.1102230246251565e-15 | 1.3322676295501878e-15 |
| synthetic-4096-stream-output-batch-a | regression | 527 | 2.2204460492503131e-16 | 0 | 8.8817841970012523e-16 | 8.1878948066105295e-16 |
| synthetic-4096-stream-output-batch-b | regression | 462 | 2.2204460492503131e-16 | 0 | 9.9920072216264089e-16 | 9.0205620750793969e-16 |
| synthetic-4096-graph-output-batch-a | regression | 494 | 2.2204460492503131e-16 | 0 | 7.7715611723760958e-16 | 8.0491169285323849e-16 |
| synthetic-4096-graph-output-batch-b | regression | 503 | 2.2204460492503131e-16 | 0 | 6.6613381477509392e-16 | 6.106226635438361e-16 |
| synthetic-multiclass17-stream-output-batch-a | pass | 0 | 8.3266726846886741e-17 | 0 | 6.6613381477509392e-16 | 1.609823385706477e-15 |
| synthetic-multiclass17-stream-output-batch-b | pass | 0 | 1.1102230246251565e-16 | 0 | 6.6613381477509392e-16 | 1.3045120539345589e-15 |
| synthetic-multiclass17-graph-output-batch-a | pass | 0 | 1.1102230246251565e-16 | 0 | 6.6613381477509392e-16 | 7.1123662515049091e-16 |
| synthetic-multiclass17-graph-output-batch-b | pass | 0 | 1.1102230246251565e-16 | 0 | 8.8817841970012523e-16 | 1.4155343563970746e-15 |
| synthetic-fallback33-stream-output-batch-a | regression | 1 | 1.1102230246251565e-16 | None | 1.1102230246251565e-16 | 2.0261570199409107e-15 |
| synthetic-fallback33-stream-output-batch-b | pass | 0 | 1.1102230246251565e-16 | None | 1.1102230246251565e-16 | 2.0261570199409107e-15 |
| synthetic-fallback33-graph-output-batch-a | pass | 0 | 9.9920072216264089e-16 | None | 9.9920072216264089e-16 | 1.0130785099704553e-15 |
| synthetic-fallback33-graph-output-batch-b | pass | 0 | 9.9920072216264089e-16 | None | 9.9920072216264089e-16 | 1.033895191682177e-15 |

## Within-policy repeat gates: a versus b

| Candidate | Gate | Regressed metrics | Max saved prediction delta | Decision changes | Max independent raw-margin delta | All-bin margin bound |
|---|---|---:|---:|---:|---:|---:|
| synthetic-scalar-stream-per-output-b | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 7.2858385991025898e-16 |
| synthetic-scalar-stream-output-batch-b | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 7.2164496600635175e-16 |
| synthetic-scalar-graph-per-output-b | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 1.0937431516033769e-15 |
| synthetic-scalar-graph-output-batch-b | pass | 0 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 7.1297134862646772e-16 |
| synthetic-33deep-stream-per-output-b | regression | 2 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 0.046609177569082262 |
| synthetic-33deep-stream-output-batch-b | regression | 4 | 1.6653345369377348e-16 | None | 1.6653345369377348e-16 | 2.1510571102112408e-16 |
| synthetic-33deep-graph-per-output-b | regression | 1 | 2.2204460492503131e-16 | None | 2.2204460492503131e-16 | 0.042375095007535842 |
| synthetic-33deep-graph-output-batch-b | pass | 0 | 1.1102230246251565e-16 | None | 1.1102230246251565e-16 | 1.5959455978986625e-16 |
| synthetic-129-stream-per-output-b | regression | 44 | 4.9960036108132044e-16 | None | 4.9960036108132044e-16 | 8.1185058675714572e-16 |
| synthetic-129-stream-output-batch-b | regression | 23 | 4.4408920985006262e-16 | None | 4.4408920985006262e-16 | 5.6551985316843911e-16 |
| synthetic-129-graph-per-output-b | regression | 26 | 5.0653925498522767e-16 | None | 5.0653925498522767e-16 | 7.7368667028565596e-16 |
| synthetic-129-graph-output-batch-b | regression | 21 | 5.5511151231257827e-16 | None | 5.5511151231257827e-16 | 5.9674487573602164e-16 |
| synthetic-1024-stream-per-output-b | regression | 221 | 2.2204460492503131e-16 | 0 | 8.8817841970012523e-16 | 1.5334955527634975e-15 |
| synthetic-1024-stream-output-batch-b | regression | 187 | 2.2204460492503131e-16 | 0 | 9.9920072216264089e-16 | 1.1657341758564144e-15 |
| synthetic-1024-graph-per-output-b | regression | 212 | 2.7755575615628914e-16 | 0 | 1.3322676295501878e-15 | 1.3877787807814457e-15 |
| synthetic-1024-graph-output-batch-b | regression | 179 | 2.7755575615628914e-16 | 0 | 1.1102230246251565e-15 | 1.1102230246251565e-15 |
| synthetic-4096-stream-per-output-b | regression | 454 | 2.2204460492503131e-16 | 0 | 8.3266726846886741e-16 | 8.0491169285323849e-16 |
| synthetic-4096-stream-output-batch-b | regression | 421 | 2.2204460492503131e-16 | 0 | 8.8817841970012523e-16 | 8.7430063189231078e-16 |
| synthetic-4096-graph-per-output-b | regression | 452 | 2.2204460492503131e-16 | 0 | 6.6613381477509392e-16 | 7.6327832942979512e-16 |
| synthetic-4096-graph-output-batch-b | regression | 410 | 2.2204460492503131e-16 | 0 | 6.106226635438361e-16 | 6.106226635438361e-16 |
| synthetic-multiclass17-stream-per-output-b | pass | 0 | 8.3266726846886741e-17 | 0 | 6.6613381477509392e-16 | 8.8817841970012523e-16 |
| synthetic-multiclass17-stream-output-batch-b | pass | 0 | 1.1102230246251565e-16 | 0 | 4.4408920985006262e-16 | 6.0715321659188248e-16 |
| synthetic-multiclass17-graph-per-output-b | pass | 0 | 8.3266726846886741e-17 | 0 | 4.4408920985006262e-16 | 1.1934897514720433e-15 |
| synthetic-multiclass17-graph-output-batch-b | pass | 0 | 1.1102230246251565e-16 | 0 | 4.4408920985006262e-16 | 6.106226635438361e-16 |
| synthetic-fallback33-stream-per-output-b | pass | 0 | 1.1102230246251565e-16 | None | 1.1102230246251565e-16 | 2.5673907444456745e-16 |
| synthetic-fallback33-stream-output-batch-b | pass | 0 | 1.1102230246251565e-16 | None | 1.1102230246251565e-16 | 8.3266726846886741e-17 |
| synthetic-fallback33-graph-per-output-b | regression | 3 | 1.9984014443252818e-15 | None | 1.9984014443252818e-15 | 2.0261570199409107e-15 |
| synthetic-fallback33-graph-output-batch-b | pass | 0 | 1.9428902930940239e-16 | None | 1.9428902930940239e-16 | 2.2898349882893854e-16 |

- Two uninstrumented samples per mode/execution/shape; medians are descriptive, not confidence intervals.
- Independent CPU model consistency uses the existing benchmark tolerance; every applicable quality metric retains zero allowance.
- Within-policy a-to-b repeat gates quantify run variability independently; they cannot change any cross-policy failure into a pass.
- All-bin raw-margin bounds compare exact-real serialized model functions and exclude runtime accumulation/transform rounding.
- These are synthetic generator-v2 fixtures, not a general speed/accuracy claim or real-data benchmark.
