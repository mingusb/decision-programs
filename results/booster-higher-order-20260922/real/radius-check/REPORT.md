# Validation clipping-radius check

Fixed caps 4 and 16 were added after the original cap-1 screen exposed a clipping confound. All twelve cells use the original training/validation fixtures and unchanged screen shapes. This check does not select a radius or inspect test results.

Completed job receipts: 12/12; passed jobs: 12; audit failures: 0. Complete metrics, operating points, timing scopes, memory payloads and strict comparison failures are retained in summary.json.

| Dataset | Order | Cap | Validation AP / macro AP | Log loss | Training wall ms | Prediction wall ms |
|---|---:|---:|---:|---:|---:|---:|
| magic | 2 | 1 | 0.928072445 | 0.413850096 | 24.729 | 1.921 |
| magic | 3 | 1 | 0.925749152 | 0.41521611 | 27.795 | 1.928 |
| magic | 4 | 1 | 0.926252595 | 0.414015489 | 34.366 | 2.195 |
| magic | 2 | 0 | 0.933283046 | 0.39791825 | 28.011 | 3.403 |
| delicious | 2 | 1 | 0.0708353944 | 0.074283665 | 7937.839 | 110.419 |
| delicious | 3 | 1 | 0.0722946844 | 0.0741933277 | 11746.504 | 92.009 |
| delicious | 4 | 1 | 0.0725886852 | 0.0741865225 | 15607.647 | 92.236 |
| delicious | 2 | 0 | 0.105561926 | 0.0709025567 | 7977.900 | 114.721 |
| magic | 2 | 4 | 0.933283046 | 0.39791825 | 26.513 | 2.026 |
| magic | 3 | 4 | 0.932140662 | 0.393207749 | 28.338 | 1.692 |
| magic | 4 | 4 | 0.934417583 | 0.383283967 | 33.562 | 2.005 |
| delicious | 3 | 4 | 0.0967630472 | 0.0723594774 | 11737.800 | 90.558 |
| delicious | 4 | 4 | 0.0951307354 | 0.0724836623 | 15798.461 | 98.234 |
| delicious | 2 | 4 | 0.0960659121 | 0.071759821 | 7995.296 | 103.083 |
| magic | 4 | 16 | 0.934417583 | 0.383283967 | 32.420 | 1.878 |
| magic | 2 | 16 | 0.933283046 | 0.39791825 | 30.571 | 1.887 |
| magic | 3 | 16 | 0.932140662 | 0.393207749 | 30.598 | 1.931 |
| delicious | 2 | 16 | 0.105541913 | 0.0709578081 | 7800.957 | 85.489 |
| delicious | 3 | 16 | 0.0963623709 | 0.0729075862 | 11916.765 | 106.814 |
| delicious | 4 | 16 | 0.0933043656 | 0.0727366881 | 15799.592 | 90.695 |

Cap zero denotes the original unclipped order-2 control. Radius comparisons are single-fit validation observations; unordered floating-point variation is not estimated here. Higher orders also change stable/unfloored derivatives. A matching cap does not isolate that change, and neither a quality win nor a default promotion follows from this check.
