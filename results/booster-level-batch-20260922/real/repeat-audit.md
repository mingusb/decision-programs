# Within-policy repeat audit

Within-policy differences quantify observed repeat variability. They do not excuse matched-policy regressions or establish a cause for those differences. Zero-allowance failures remain failed. In the unchanged gate function, baseline means the explicitly labeled reference run.

Reference is test repetition r0; candidate is r1 or r2 of the same policy, dataset, configuration and binary. These are not comparisons between policies. Regression predictions use target units; binary/multiclass predictions are probabilities.

| Dataset | Policy | Pairs | Maximum prediction difference | Changed-topology pairs | Strict loss failures | Any-metric failures |
|---|---|---:|---:|---:|---:|---:|
| wine | custom-per-output | 2 | 0.1194348411 | 2 | 1 | 2 |
| wine | custom-output-batch | 2 | 0.114436884 | 2 | 0 | 2 |
| magic | custom-per-output | 2 | 0.04095993703 | 2 | 0 | 0 |
| magic | custom-output-batch | 2 | 0.04095993703 | 2 | 0 | 0 |
| letter | custom-per-output | 2 | 0.103636381 | 2 | 1 | 2 |
| letter | custom-output-batch | 2 | 0.1534560067 | 2 | 0 | 0 |
| delicious | custom-per-output | 2 | 0.5058195835 | 2 | 2 | 2 |
| delicious | custom-output-batch | 2 | 0.2905407202 | 2 | 0 | 2 |

# Largest preserved matched-policy difference

Among 28 matched-policy comparisons, the largest difference is 0.59643733248038688 probability on delicious validation, zero-based row 2446, output 965 (TAG_wikipedia). The actual target is 1; per-output predicts 0.67117692629372083, and output-batch predicts 0.074739593813333921.

Reference case: `validation-delicious-g3-custom-per-output`. Candidate case: `validation-delicious-g3-custom-output-batch`. Training positives for this label: 109/10336. The decision changes at this witness: True. Across the entire pair, 25 decisions change across 23 rows. Fixture, model, prediction and capture hashes are retained in JSON.
