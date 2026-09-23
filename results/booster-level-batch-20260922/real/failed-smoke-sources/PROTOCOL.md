# Real-data comparison protocol, fixed before measurements

This is an initial four-dataset comparison, not evidence of universal superiority.
All GPU jobs run serially under the root agent. Profilers are excluded from timing
rankings. Framework algorithms are benchmark references only; the custom library
does not link to them. Offline dataset parsing, splitting, and binary fixture
writing are experiment setup, never a production training/preprocessing stage.

## Data and binary interface

Each fixture begins with the eight bytes `GHBDS001`, followed by six little-endian
uint32 values: version=1, rows, feature columns, target columns, objective
(0=squared error, 1=independent binary logistic, 2=multiclass softmax), classes.
The header is followed by row-major little-endian float32 features and targets.
Multiclass has one target column with class indices. Independent multilabel
targets have one binary column per label. There are no weights or categorical
feature declarations. Trailing bytes, invalid shapes, nonfinite targets, and
invalid label domains are rejected. Numerical feature NaNs represent missing;
infinities are rejected. Production custom training calls the existing CUDA
feature-fitting/binning path on these raw floats.

`ghb_real_bench --train FILE --evaluation FILE --output-dir NEW_DIRECTORY`
accepts rounds, depth, bins, learning-rate, l2, output-tile, tree-export-batch,
tree-execution, tree-build, histogram, and memory-budget controls. It writes
model.ghb, predictions.f64 (row-major probabilities or regression predictions),
and metrics.json. Evaluation never affects fitting. Loading and serialization
times are separate from the synchronized raw-host-input training wall time.
Prediction timing includes the public GPU prediction API's uploads, execution,
and returned host array; it is not a resident-kernel-only latency.

Initial datasets:

* White wine quality: 4,898 rows, 11 numeric features, scalar regression.
  Identical feature rows are kept in the same split to reduce duplicate leakage.
  Source: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality),
  Cortez et al., DOI 10.24432/C56S3T, CC BY 4.0.
* MAGIC: 19,020 rows, 10 numeric features, binary gamma/hadron labels.
  This is a published Monte Carlo physics benchmark, not an observational sample.
  Source: [UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope),
  Bock, DOI 10.24432/C52C8B, CC BY 4.0. Report ROC AUC in addition to log loss,
  because the source explicitly warns that accuracy alone is inadequate.
* Letter: 20,000 rows, 16 numeric features, 26 classes. Preserve the source's final
  4,000-row test set and split only its first 16,000 rows for validation.
  Source: [UCI Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition),
  Slate, DOI 10.24432/C5ZP40, CC BY 4.0.
* Delicious: 16,105 rows, 500 binary numeric features, all 983 independent labels.
  Preserve official train/test membership; reserve validation from train only.
  Source: [Mulan datasets](https://mulan.sourceforge.net/datasets-mlc.html),
  Tsoumakas, Katakis and Vlahavas, ECML/PKDD MMD'08. The archive provides data
  for research comparison but no explicit standalone license; preserve attribution
  and source archive, and do not imply a broader redistribution license.

The fixture manifest records source SHA256, exact row indices, deterministic split
seed, binary SHA256, shapes, and split target summaries. No scaling, imputation,
feature selection, learned encoding, feature replication, or label reduction is
performed. Sparse absent Delicious entries mean observed zero, not missing.

## Fairness and selection

Compare custom per-output and output-batch trainers, XGBoost CUDA histogram,
LightGBM CUDA, and CatBoost GPU. All consume identical float32 feature/target
values, split membership, no row/feature sampling, and the same requested maximum
depth, learning rate, round budget and nominal bin budget. These are native
framework comparisons, not claims of identical algorithms: quantile cut methods,
minimum-child rules, multiclass Hessians, regularization scaling, tree shape and
output sharing differ. CatBoost's native symmetric vector-leaf tree is explicitly
different from independent scalar trees; XGBoost's primary baseline explicitly
uses `one_output_per_tree`. A vector-leaf XGBoost result may be added as a separately
labeled architecture. LightGBM uses one GPU model per independent label because
its public objective contract does not expose native independent multi-label
training. Report the complete wrapper time and model count, never a per-label
extrapolation. Constant-label columns may use a documented constant predictor
without constructing an invalid classifier, with all such columns recorded.

For each dataset/implementation, run the same four predeclared configurations and
select the lowest validation MSE or log loss; ties select lower training wall time.
Test predictions are generated only for the selected configuration. Selection
uses no test scores. No early stopping or unequal tuning allowance. Repeat the
selected configuration three times in alternating implementation order; retain
every raw time and metric. Training uses only train, not a post-selection union.

Proposed initial scalar/multiclass grid: rounds {25,75}, depth {3,5}; multilabel
grid: rounds {5,10}, depth {2,3}; learning rate 0.1, l2=1, bins=32. These bounded
budgets establish working quality/speed comparisons; they are not saturated
accuracy or NLP language-model experiments. Record any final grid change before
the first timed job. Identical configurations apply to both custom modes.

## Timing, quality, and limits

Separate dataset loading, framework preparation, fit-only, complete raw-host-input
training (including framework preparation/upload/binning), model serialization,
and public prediction latency. CUDA synchronization brackets timing. A small
training warm-up may be performed and separately recorded for every implementation;
never charge it to one implementation alone or discard an undocumented first run.
Report process peak RSS; custom owned-device bytes are not directly comparable to
framework allocator/device peak memory. A sampled GPU-memory measurement may be
added uniformly by the serial runner, with its sampling interval and baseline.

Regression: MSE, RMSE, MAE, R2. Binary: natural-log loss, Brier score, ROC AUC,
average precision, accuracy and F1 at 0.5. Multiclass: log loss, accuracy, macro F1,
multiclass Brier. Multilabel: mean per-label log loss/Brier, micro/macro F1 at 0.5,
Hamming loss, exact-match accuracy, micro/macro AP, and precision@1/@3/@5; macro
AP reports the number of labels with test positives. CPU metric computation is
an explicit validation reference after timed GPU work. Report a training-mean
constant predictor for context, calculated from train targets only.

Cross-framework quality differences are model differences; numerical agreement
between custom modes remains a separate check. Historic zero-allowance failures
remain failed, even though the user has accepted observed rounding-scale variation
as non-blocking for default selection. Preserve adverse speed and quality results.

## Primary implementation references checked 2026-09-22

* [XGBoost GPU support](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
  and [multi-output](https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html):
  CUDA `hist`, scalar-per-output and vector-leaf modes; use installed 3.4.1.
* [LightGBM CUDA build](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-cuda-version)
  and [parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html):
  `device_type=cuda` is distinct from OpenCL `gpu`; source-build 4.7.0 for SM86.
  CUDA currently uses double-precision histogram arithmetic.
* [CatBoost multilabel objectives](https://catboost.ai/docs/en/concepts/loss-functions-multilabel-classification)
  documents GPU MultiLogloss; [multiregression](https://catboost.ai/docs/en/concepts/loss-functions-multiregression)
  documents GPU MultiRMSE but not MultiRMSEWithMissingValues. Use 1.2.10.

Each job records the package/build version and resolved parameters. Requested GPU
backends must be verified from configuration or explicit backend startup logs;
an error is recorded rather than silently substituting CPU training. LightGBM's
public Python prediction is a CPU path and must be labeled as such; its latency
does not rank GPU prediction kernels.
