# Completion inventory

This is a tracking inventory, not evidence that the entries are implemented.
Baseline: pre-greenfield-20260923; source and full observations preserved outside
the new production tree. New API and compact error/report records are permitted;
model/dataset/prediction formats and supported numerical behavior are retained.

| ID | Required contract | Evidence to complete |
|---|---|---|
| C1 | u8/u32 input, u32/u64 counts, overwrite/empty/alignment/nonalias and overflow | independent exact GPU oracle and boundary suite |
| C2 | all exercised count algorithms, 16 tuning policies, measured default keys | catalog coverage, full-operation comparison |
| C3 | resident reuse, changed input, clear modes and graph behavior | lifetime/replay checks and measured boundaries |
| D1 | exact distinct quantiles, radix4/8, complete categories | independent GPU sorted-distinct fixtures |
| D2 | missing/unseen/Inf/zero, feature-major u16, strict budgets | exact metadata/bins and failure tests |
| D3 | GHBDS001 import/export with validated bounds | literal bytes, roundtrip and malformed cases |
| M1 | immutable resident schema/model, structural/numeric validation | graph corruption, capacities, finite/bound cases |
| M2 | ordered FP64 raw prediction and objective transforms | bit comparisons, empty/tail, signed zero/nonfinite |
| M3 | GHBMODEL v1 and prediction bytes | literal fixtures and full roundtrip |
| T1 | weighted regression, binary/multilabel, coupled multiclass | initialization/derivative/loss/reference checks |
| T2 | GPU stats/splits/routing/frontiers/materialization/prediction updates | exhaustive small trees and end-to-end suite |
| T3 | root batching/count reuse, output tiles, global/shared/auto, split schedules | numerical schedule controls plus quality gates |
| T4 | order3/order4 clipped logistic, bounded workspace/exports | solver boundary/oracle and higher-order campaigns |
| Q1 | synthetic weighted RMSE/MAE/logloss/accuracy/Brier/tied-AUC per output/aggregate | pinned arithmetic-profile conformance |
| Q2 | real-data MSE/RMSE/MAE/R2, multiclass macroF1/Brier, binary F1/AP | frozen real-data protocol and predictions |
| Q3 | multilabel hamming/exact-match/micro+macroF1/AP/AUC/precision@1,3,5 | all outputs, eligibility denominators and tie rules |
| Q4 | signal confusion/recall/FPR/precision/F1 at .5 and validation-frozen threshold | exact selection constraint and held-out results |
| Q5 | zero-allowance gates including unavailable metrics | finite adjacent/extreme/signed-zero comparisons |
| P1 | GPU-generated cases, independent GPU oracles, sanitizer | every required case records completion |
| P2 | uninstrumented complete operations, paired raw timing/statistics | qualified timestamp protocol and archive controls |
| P3 | Nsight Systems/Compute, CUPTI, NVBit, CUDA-GDB, disassembly | actual activity per supported collector, failures kept |
| S1 | formatted lines/tokens by production/tests/tools, no dead code | same-capability baseline and final measurements |

No required entry is removed to improve source-size numbers. Earlier passes and
failures remain tied to their original sources. New code is not fastest until a
fair applicable measurement establishes that local result.
