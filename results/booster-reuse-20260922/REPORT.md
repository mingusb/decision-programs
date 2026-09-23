# Root count reuse, root split batching, and deeper histogram experiments

Implemented opt-in exact root-count reuse and batched root split searches in
the GPU-resident CUDA C++23 trainer. Added independent deeper-histogram batching
primitives and measured them before changing the trainer's retained tree state.
Original count kernels/policies/defaults and all prior finalized evidence remain
unchanged. No NVIDIA algorithm is linked into the production trainer.

## Contracts and implementation

[The pre-code decision](../../training/REUSE_BATCH_EXPERIMENT.md) defines the
input/output, numerical, memory and experimental contracts, with primary sources
and prior measured evidence. All candidates use owned algorithms. Production
statistics and split decisions remain on the GPU; CPU metric evaluation and
validation references are outside production compute and the timing boundary.

- `--root-counts reuse-global|reuse-shared`: compute immutable uint64 root counts
  once, seed each output's histogram with them, and accumulate only FP64 G/H.
  Counts include missing and zero-weight rows. This requires fixed bins and root
  row membership, as in the present no-sampling trainer.
- `--split-batch root`: search all roots in a tile together and consume cached
  winners during tree initialization. This removes per-tree root histogram copies
  and split launches. Candidate scratch is reused. The arithmetic per feature,
  tie rules and independent-tree architecture are retained.
- New deeper primitives batch independent per-output assignments with global or
  shared histograms, clear active ranges and preserve inactive capacity. They are
  not integrated into the multi-tree trainer. Their benchmark excludes the future
  cost of retaining and managing multiple tree states.

Both new trainer controls require `--root-histogram batched`. Existing defaults
remain per-tree roots, per-output counts, per-tree splits and block256 split
search. Multiclass retains its full frozen pre-round derivative snapshot.

Exact count reuse does not establish identical FP64 atomic ordering. Fixed-input
split tests check result fields bitwise; independent full-training comparisons
retain all strict quality failures.

## Complete training measurements

RTX A5000 Laptop, SM86, 48 SMs, 16 GiB, WSL2, CUDA 13.4.59; clocks were not locked.
All GPU jobs ran serially. The 61-case campaign and three primitive benchmarks
finished before CPU quality auditing and diagnostic profiling began. Timings used
for ranking have instrumentation off. Source data generation is identical to the
prior frozen benchmark; results include exact commands, binary hashes, telemetry,
saved models and held-out predictions.

The baseline here is the **previous combined policy**: batched roots plus warp32
split policy, with per-output count atomics and per-tree split searches. Thus
these ratios are incremental changes, not comparisons against NVIDIA histograms
or external boosting libraries.

The following table gives both total-training wall-time observations, in ms,
from forward/reverse policy sweeps. It includes preparation, count-cache setup,
graph construction, boosting and model export, input checks, structural model
validation and device cleanup. Held-out prediction and CPU reference validation are separate.
Raw training-only times and every configuration are in
[timing-summary.json](timing-summary.json) and [campaign.stdout](campaign.stdout).

| Workload | Baseline | Count reuse (global setup) | Batched splits | Both | Both + deeper shared |
|---|---:|---:|---:|---:|---:|
| Scalar, N65536/F32, 10 rounds, depth5, B64 | 44.51 / 47.48 | 41.10 / 41.93 | 47.39 / 44.35 | 39.45 / 46.90 | 40.96 / 43.91 |
| 129 regression outputs, N4096/F16, 3 rounds, depth2, B32 | 95.99 / 93.53 | 86.08 / 87.81 | 87.22 / 92.43 | 86.50 / 78.36 | 114.04 / 101.88 |
| 1024 binary outputs, N4096/F16, 2 rounds, depth2, B16 | 457.08 / 436.18 | 411.65 / 426.62 | 480.77 / 416.85 | 408.32 / 406.36 | 526.32 / 520.24 |
| 4096 binary outputs, N1024/F16, 1 round, depth2, B16 | 789.42 / 747.35 | 771.34 / 750.77 | 727.96 / 714.80 | 736.69 / 721.11 | 761.68 / 752.83 |

Paired total-time reductions for both changes were 9.9–16.2% at 129 outputs,
6.8–10.7% at 1024, and 3.5–6.7% at 4096. Two observations per policy are not a
confidence interval. Four baseline repeats (a/b/c/d) span 82.23–95.99 ms at
129 outputs, 417.97–457.08 ms at 1024 and 747.35–789.42 ms at 4096. In particular,
129-output baseline-c at 82.23 ms is below combined-a at 86.50 ms. Those additional
raw observations remain in the campaign; the percentage ranges above compare
the paired forward/reverse sweeps only. Launch/host overhead, unlocked clocks and order remain
limits; no tiny near-one result establishes a universal ranking.

Negative and confirmation observations matter:

- Split batching alone lost one 1024-output sweep (480.77 vs 457.08 ms), while
  winning the reverse sweep. Counts alone lost the reverse 4096-output sweep
  (750.77 vs 747.35 ms). Both changes were slower than splits alone in both
  4096-output sweeps.
- Shared count setup with per-tree splits took 79.28 / 85.36 ms at 129 outputs,
  but 816.88 / 761.80 ms at 4096, slower than the respective baselines there.
- Fresh seed 20260922604: 129-output baseline/both/deeper-shared took
  85.02 / 82.98 / 105.51 ms; 4096 outputs took 775.52 / 739.13 / 772.71 ms.
  These single-run checks do not establish statistical reproducibility.
- Seventeen-class multiclass, tile5: 25.24 / 24.04 / 26.99 ms for
  baseline/both/deeper-shared. The partial-tile and full-derivative semantics are
  also covered by integration tests.
- The original scalar policy took 46.34 / 45.35 ms. New scalar combined timings
  overlap that range. Scalar already used shared deeper histograms, so its
  `both` and `deep-shared` cases are identical configurations and expose timing
  variability rather than different algorithms.

## Memory and component boundaries

At 129 outputs the new combination adds 3,880 count-cache bytes and 11,520 bytes
of incremental split scratch, increasing declared training device payload from
7,769,956 to 7,785,356 bytes. At 1024/4096 outputs it adds 1,960 + 11,520 bytes.
The scalar case adds 15,912 + 48 bytes. These components are included in existing
totals, not extra unreported allocations. Requested device/histogram limits are
checked, including one-byte-below boundary failures in tests.

See [root/split primitive results](root-split-primitives.md) for complete root
setup/cache-write and all-output split boundaries, raw spreads and negative cases.
See [deeper results](deeper-results.md) for all 16 shapes in stream and graph
execution. Primitive ratios are not full-training speedups. Synthetic deeper
assignments do not reproduce all learned partition imbalance/correlation or
large-class multiclass derivative strides.

## Quality, sanitizer and profiler gates

All 11 CTest suites pass. The final deeper binary also passes a targeted rerun.
All ten Compute Sanitizer runs pass: memcheck/racecheck/synccheck for root
histograms, split search and deeper histograms, plus full-trainer memcheck.
Racecheck reports zero hazards, errors or warnings; the other runs report zero
errors. Raw receipts and [sanitizer-summary.json](sanitizer-summary.json) retain
the commands, binary identities and full coverage, including the 397-second
split-search race check. No timed-output validation claim is added to the count
microbenchmark beyond its separately checked algorithm tests.

The quality audit's outcome is **regression**, with no invalid-evidence errors:

- All 61 trained models pass against their corresponding untrained constant
  predictions on every applicable metric.
- 52 of 62 strict paired comparisons fail; 10 pass. All 18 directional repeated
  baseline comparisons fail. These are nine run pairs tested in both directions.
  The remaining implementation/policy comparisons fail 34 of 44 gates.
- Maximum metric deterioration is 4e-16 nats in an individual output's log loss.
  The maximum prediction difference is 6.38378239159465e-16; classification
  decisions do not change. The largest exact-real encoded-domain margin bound is
  1.5681900222830336e-15; this bound excludes runtime summation rounding.

See [QUALITY.md](QUALITY.md), the [raw quality summary](quality/summary.json),
and [metadata verification](metadata-verification.json). The independent verifier
checks 61 cases, all 123 base/paired gates, complete metric coverage, exact flags,
objectives, seeds, target/source hashes and one unchanged executable. It retains
the regression status. Previous frozen repeats also fail 13/16 directional gates;
[their report](previous-repeat-controls/REPORT.md) distinguishes that variability
from evidence about any particular new optimization.

Tiny differences and unchanged decisions do not turn a zero-allowance failure
into a pass. Count reuse is exact for counts; FP64 accumulation is unordered.
Defaults are not promoted and no universal accuracy or speed claim is made.

Nsight Systems and Compute captures are diagnostic only. Three Systems captures
match all 2,666 emitted stage samples to NVTX ranges and verify no tree-scope
host waits or D2H calls. [SYSTEMS.md](SYSTEMS.md) explains launch counts and
remaining work; [COMPUTE.md](COMPUTE.md) examines occupancy, atomic traffic and
shared CAS behavior. Their instrumented durations never enter the ranking tables.
The combined trace executes 6,755 device kernels versus 7,447 in the baseline:
692 fewer, after including the new batched split, count setup and seed kernels.
Export waits and bytes remain unchanged. Deeper global accumulation remains
33.3% of that trace's kernel-time sum, making further batching worth testing.

## Next experiment supported by this evidence

The deeper batched global primitive has the lowest observed median in all 14
multi-output shapes in both execution modes. The next architectural experiment
is to retain bounded per-output tree state and build a level across an output
tile, then compare complete training including added state storage, routing,
split decisions and exports. The current primitive measurements alone do not
establish that this integration will win. Scalar shared results also argue
against assuming one universal deeper policy.

Separately, any claim of zero quality loss needs a numerical contract and
evidence that passes the existing gate. Repeated baseline failures establish
current variability; they do not authorize an allowance or explain away every
changed result. No deterministic-reduction implementation or altered gate is
introduced in this experiment.

## Evidence and preservation

Trainer SHA-256:
`fdf60682669fa790dc411e67d9053c0e02e6d49de13b365debc41254ba0f02a1`.
The same binary produced all 61 campaign cases. `final-provenance/` retains
64 source/build artifacts, including tests and the pre-code decision. Build and
runtime receipts retain executable hashes. Original count artifacts and all
684 previous root/split artifacts pass their existing manifests; see
[preservation-check.json](preservation-check.json). The production archive has
no CUB or Thrust algorithm symbols. [Generator check](generator-check.json)
verifies the exact synthetic data generator section against the frozen source.
[The final evidence manifest](artifacts.sha256) seals raw observations, reports,
scripts and frozen sources; [seal verification](seal-verification.json) records
its verification while retaining the failed quality status.

Raw failed setup attempts remain retained: the first CMake configure preceded
creation of the new benchmark source. A follow-up CTest command used an incorrect
test-name regex and selected no tests; it is not counted as validation. The
corrected command uses `--no-tests=error` and passes the final deeper test binary.
