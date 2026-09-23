# Higher-order exploratory comparison

exploratory reused-test follow-up; no data-uncertainty intervals or claims of fastest/superior generalization

All main runs use leaf clipping 1, including the order-2 control. Selection maximizes validation AP (MAGIC) or macro AP (Delicious), then minimizes log loss and synchronized training wall time. Test thresholds are frozen from the selected validation model; test FPR is measured, not constrained retrospectively.

| Dataset | Order | Rounds / lambda | Train ms | Prediction ms | AP / macro AP | Log loss | Recall at validation-selected threshold | Measured test FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| magic | 2 | 75 / 1 | 69.788 | 3.908 | 0.94503624 | 0.32154834 | 0.60218978 | 0.073298429 |
| magic | 3 | 75 / 1 | 79.090 | 2.508 | 0.94387408 | 0.32274248 | 0.61476075 | 0.077038145 |
| magic | 4 | 75 / 1 | 95.074 | 4.190 | 0.94365517 | 0.3231262 | 0.61678832 | 0.075542259 |
| delicious | 2 | 10 / 10 | 28231.774 | 251.240 | 0.085340289 | 0.071180478 | 0.5231699 | 0.049893084 |
| delicious | 3 | 10 / 1 | 42426.012 | 235.609 | 0.086907747 | 0.07083958 | 0.52723771 | 0.050005113 |
| delicious | 4 | 10 / 1 | 58439.281 | 254.234 | 0.087296808 | 0.070833279 | 0.527686 | 0.050448996 |

Matched optimizer comparisons: 22; strict all-common-metric failures: 20; additional signal-metric failures: 21. All failures and metric deltas remain in JSON. Unmatched selected configurations are not treated as matched optimizer controls.

Coarse time-to-quality observations, complete metric values, threshold confusion counts, timing ranges, raw repeats and memory scopes remain in JSON. These are bounded tuning results on previously inspected held-out data, not fresh confirmation or saturated model-quality rankings.
