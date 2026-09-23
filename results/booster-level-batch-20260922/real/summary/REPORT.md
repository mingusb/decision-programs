# Validation-selected real-data results

All training jobs used GPU backends. Times below include preparation/upload/binning and model fitting from raw host input, in fresh processes with CUDA context setup excluded. Three selected test repetitions are summarized by median and full range. Parameters were selected only on validation; test metrics did not select models.

## wine

| Implementation | Rounds / depth | Training ms (range) | Test loss | Public prediction ms | Prediction backend |
|---|---:|---:|---:|---:|---|
| custom-per-output | 75 / 5 | 50.777 (46.885–55.415) | 0.48661927 mse | 2.425 | cuda |
| custom-output-batch | 75 / 5 | 51.047 (49.259–67.154) | 0.48726513 mse | 4.473 | cuda |
| xgboost | 75 / 3 | 225.800 (209.550–261.225) | 0.50025767 mse | 81.706 | cuda |
| lightgbm | 75 / 5 | 1150.994 (1133.499–1176.330) | 0.4948043 mse | 0.894 | cpu-public-lightgbm-api |
| catboost | 75 / 5 | 883.819 (856.548–908.312) | 0.49038766 mse | 4.386 | cuda |

## magic

| Implementation | Rounds / depth | Training ms (range) | Test loss | Public prediction ms | Prediction backend |
|---|---:|---:|---:|---:|---|
| custom-per-output | 75 / 5 | 68.718 (63.724–77.693) | 0.31584596 log_loss | 2.857 | cuda |
| custom-output-batch | 75 / 5 | 62.824 (62.530–64.464) | 0.31581744 log_loss | 2.520 | cuda |
| xgboost | 75 / 5 | 295.222 (282.471–326.333) | 0.31658513 log_loss | 80.071 | cuda |
| lightgbm | 75 / 5 | 1142.454 (1052.221–1171.533) | 0.30926143 log_loss | 2.473 | cpu-public-lightgbm-api |
| catboost | 75 / 5 | 902.293 (873.462–940.143) | 0.33046883 log_loss | 5.390 | cuda |

## letter

| Implementation | Rounds / depth | Training ms (range) | Test loss | Public prediction ms | Prediction backend |
|---|---:|---:|---:|---:|---|
| custom-per-output | 75 / 5 | 1252.573 (1245.986–1294.860) | 0.32514239 log_loss | 29.603 | cuda |
| custom-output-batch | 75 / 5 | 483.807 (480.277–496.446) | 0.32606019 log_loss | 49.093 | cuda |
| xgboost | 75 / 5 | 3931.765 (3875.256–4016.873) | 0.32512334 log_loss | 156.130 | cuda |
| lightgbm | 75 / 5 | 36387.234 (36160.692–37967.121) | 0.16132678 log_loss | 87.602 | cpu-public-lightgbm-api |
| catboost | 75 / 5 | 1569.640 (1507.852–1817.104) | 0.60824129 log_loss | 4.528 | cpu-public-catboost-api-multioutput-gpu-evaluation-unsupported |

## delicious

| Implementation | Rounds / depth | Training ms (range) | Test loss | Public prediction ms | Prediction backend |
|---|---:|---:|---:|---:|---|
| custom-per-output | 10 / 3 | 49272.021 (49219.678–49837.402) | 0.067766926 log_loss | 163.614 | cuda |
| custom-output-batch | 10 / 3 | 28386.659 (28179.005–28638.069) | 0.067767558 log_loss | 169.580 | cuda |
| xgboost | 10 / 3 | 13343.327 (13234.378–13673.370) | 0.067763569 log_loss | 419.298 | cuda |
| lightgbm | 10 / 3 | 91009.547 (88072.563–92804.769) | 0.067123888 log_loss | 993.061 | cpu-public-lightgbm-api |
| catboost | 10 / 3 | 9878.300 (9840.633–10100.041) | 0.23482477 log_loss | 76.620 | cpu-public-catboost-api-multioutput-gpu-evaluation-unsupported |

Native algorithm and architecture differences remain: CatBoost has symmetric shared vector leaves, LightGBM independent labels use a complete serial binary-relevance loop, and binning/objective/regularization details differ. CPU reference inference entries do not rank GPU prediction kernels. See PROTOCOL.md and every raw metric in summary.json. Custom owned device payload, process RSS and any separate sampled device-wide GPU-memory observation have different scopes and must not be conflated.

These shallow, bounded pilot grids and four datasets do not establish saturated accuracy, large-language-model capability, or universal superiority.

## Provenance audit

Verified every requested case: 80 validation and 60 test. Checks passed for 280 captured fixture identities, 700 captured quality/metrics/prediction/stdout/stderr hashes, 680 frozen source hashes, 56 unchanged custom-binary identities and 60 cached-evaluator provenance hashes. Captured commands and quality-reference identities agree. Full case capture hashes and source identities are retained in JSON; immutable baseline-cache payloads have a separate audit.

## Matched custom implementation checks

The JSON preserves every matched validation configuration and each test repetition whose selected configurations match. Explicit higher/lower directions gate every emitted quality metric with zero allowance; metadata/counts are excluded. The unchanged evaluator emits aggregate metrics, not per-output metric arrays. A failed gate remains failed; it is not relabeled exact because a difference is small. Loss-only results remain separately recorded.

| Dataset | Compared pairs | Max absolute prediction difference | Topology/quantization changed pairs | Strict loss regressions | Any-metric regressions |
|---|---:|---:|---:|---:|---:|
| wine | 7 | 0.1794409 | 6 | 3 | 5 |
| magic | 7 | 0.23553372 | 5 | 1 | 3 |
| letter | 7 | 0.25709104 | 7 | 7 | 7 |
| delicious | 7 | 0.59643733 | 7 | 6 | 7 |

## Preserved capability failures

13 failed smoke jobs are retained verbatim in summary.json, including adapter/runtime setup failures and unsupported CatBoost multi-output GPU evaluation. Supported training remains on GPU; explicit public CPU reference prediction is labeled above.

## Observed validation tradeoffs

Each row below is nondominated among the four-config budgets on its dataset: no other observed point has both no-greater training time and no-greater validation loss, with at least one strict improvement. These are single validation-run timings, not repeated speed rankings. They do not change the validation-loss-only test selection. All dominated and nondominated observations remain in JSON.

| Dataset | Implementation | Rounds / depth | Validation loss | Observed training ms |
|---|---|---:|---:|---:|
| wine | custom-per-output | 25 / 3 | 0.58758085 mse | 20.605 |
| wine | custom-output-batch | 25 / 5 | 0.55491717 mse | 25.119 |
| wine | custom-per-output | 25 / 5 | 0.55473841 mse | 32.701 |
| wine | custom-output-batch | 75 / 3 | 0.55219668 mse | 40.346 |
| wine | custom-output-batch | 75 / 5 | 0.54503399 mse | 48.098 |
| magic | custom-per-output | 25 / 5 | 0.35057307 log_loss | 27.459 |
| magic | custom-per-output | 75 / 3 | 0.33268777 log_loss | 50.626 |
| magic | custom-output-batch | 75 / 5 | 0.30304742 log_loss | 61.399 |
| magic | lightgbm | 75 / 5 | 0.29573115 log_loss | 1183.641 |
| letter | custom-output-batch | 25 / 3 | 1.0715935 log_loss | 116.889 |
| letter | custom-output-batch | 25 / 5 | 0.69591134 log_loss | 170.596 |
| letter | custom-output-batch | 75 / 3 | 0.57543476 log_loss | 369.866 |
| letter | custom-output-batch | 75 / 5 | 0.3001281 log_loss | 464.368 |
| letter | custom-per-output | 75 / 5 | 0.29925936 log_loss | 1251.021 |
| letter | lightgbm | 75 / 3 | 0.26546298 log_loss | 9985.500 |
| letter | lightgbm | 75 / 5 | 0.14577429 log_loss | 44849.555 |
| delicious | catboost | 5 / 2 | 0.37803262 log_loss | 4592.625 |
| delicious | xgboost | 5 / 2 | 0.070901607 log_loss | 5002.416 |
| delicious | xgboost | 5 / 3 | 0.070200278 log_loss | 7120.247 |
| delicious | xgboost | 10 / 2 | 0.069193508 log_loss | 8934.600 |
| delicious | xgboost | 10 / 3 | 0.068261592 log_loss | 12616.239 |
| delicious | lightgbm | 10 / 3 | 0.067619725 log_loss | 89686.786 |
