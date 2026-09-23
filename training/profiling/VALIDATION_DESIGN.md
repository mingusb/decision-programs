# Optional numerical and lifecycle validation

Selected before implementation, 2026-09-22. These are bounded diagnostic
executables calling the existing CUDA C++23 implementation. They are neither new
production algorithms nor performance benchmarks. All host calculations are
explicit validation references; production source, arithmetic and defaults remain
unchanged.

## Contracts and candidate choice

`ghb_validate_gpu --output NEW.json --mode all|stage|lifecycle` uses the existing
allocation-free asynchronous statistics/split APIs and the synchronous public
train/predict/export interfaces. Stage inputs have feature-major uint16 bin IDs,
FP64 row-major derivatives, uint64 exact counts, validated finite inputs and
nonaliasing device storage. Three logical outputs use a two-output allocation and
a one-output final tile. The bounded trace includes one root and a routed deeper
frontier. An exact dyadic fixture permits zero-allowance statistic checks; a
cancellation-sensitive fixture measures floating-order differences separately.
Every small-case candidate is independently enumerated from downloaded actual
statistics, with its identity, eligibility, gain and runner-up margin. Reference
candidate evaluation uses long double, not an assertion of arbitrary precision.
Actual GPU per-feature candidates and winners are retained separately. Host
oracle routing/tree construction is diagnostic only.

The numerical report (pages 7–18) favors staged checkpoints and decision margins
over blanket higher precision. The current implementation already uses FP64;
FP32-to-FP64 promotion therefore adds no diagnostic distinction. Full small-array
snapshots avoid hash-collision ambiguity and expose all candidates rather than
assuming the top few contain the reference winner. Public APIs do not expose
all per-threshold GPU scores: independently enumerated scores must never be
labeled GPU scores. No new custom reduction, approximate sum, CPU training path,
or numerical clamp is introduced.

Repeated graph runs compare exact integer fields with zero tolerance. Dyadic
statistic sums and deterministic stages fed identical snapshots also have exact
gates. Unordered FP64 atomic sums on adversarial inputs may differ; retain the
differences and a conservative finite-error bound without calling them exact
preservation or changing any quality acceptance threshold. A logical first
divergence is not a hardware timestamp or proof of instruction-level causality.
Instrumented observations are not production timing evidence.

Independent candidate checks cover every feature and the final winner. A clearly
inferior candidate, omitted clearly positive split, wrong eligibility, corrupted
leaf or wrong exact tie order fails. The actual GPU winner must equal the total
ordering of the actual GPU per-feature candidates, with zero allowance. Small
candidate intervals that overlap are explicitly `ambiguous_within_numerical_budget`,
not exact numerical rank preservation. Diagnostic gain budgets scale with the
square of the absolute input-gradient sum to account for cancellation; these
fixture-specific conservative budgets are not a general forward-error proof or
a predictive-quality allowance. `--self-test-reference` is a CPU-only check that
deliberately corrupts winners, gains, leaves and identities and requires rejection.

Lifecycle coverage repeatedly trains alternating bounded shapes/objectives,
predicts, serializes, reloads, verifies prediction identity and destroys owned
objects before the next case. Counts/tail storage are exact gates. CPU prediction
is only an independent validation reference. No public asynchronous trainer,
allocator, graph-update or workspace API is invented.

`ghb_validate_count --output NEW.json` uses the existing count library, two
nonblocking streams, independent graph instances, nonoverlapping buffers and
valid uint32 IDs. uint64 counts must equal independent host integer references
exactly. One case gives each graph independent scratch; another shares scratch
with an explicit CUDA event dependency and validates each result before any
overwriting reuse. Changed input with retained shape/pointers is legal graph
replay. This concurrency correctness stress is separate from serial performance
experiments. An explicit global-window policy exercises nonempty scratch; it
does not replace automatic production defaults.

## Validation plan and limits

Build in a separate diagnostic configuration; inspect that production sources
are unchanged. Root runs all GPU work serially, except for the explicitly
authorized independent-stream stress within the count executable. Run each
harness normally, then relevant Compute Sanitizer modes. Preserve complete JSON,
commands, binary identity, failures and collector logs. Output paths must not
exist. A failed exact gate causes nonzero exit and remains visible in the report.
Successful bounded cases do not establish universal correctness or numerical
quality. No held-out quality gate is relaxed, and no speed claim is made.

## Initcheck finding: compact Node export representation

Recorded before any corrective production edit. The expanded initcheck run in
`results/profiling-expansion-20260923/initcheck-all-booster/` failed with 13 host
API uninitialized-source reports. Compact exports copied 160, 96 and 160 bytes,
corresponding to 5, 3 and 5 live 32-byte Nodes. Every Node was reported, rather than
only unused capacity. The Node layout has five 4-byte fields, four alignment
bytes at offsets 20–23, then its double value at offset 24. Offline SASS for the
actual `batch_resident.cu` materializer confirms writes at offsets 0–15, 16–19
and 24–31, leaving that gap untouched. The host bytewise D2H copy includes it.

Both trainer setup paths already clear the allocated Node representation once
when fixed-capacity export is enabled. Compact export (`tree_export_batch_size=0`)
skips that initialization. Candidate correction: retain the existing one-time
`cudaMemsetAsync` for all export modes, in both `booster.cpp` and
`batch_training.inc`. Contract: every exported object byte is initialized;
named Node fields, topology, live arithmetic, serialized model fields, count
kernels and defaults remain unchanged. The clear precedes all tree construction
on its existing stream, outside graph replay, with no extra scratch or atomics.

Alternatives considered: a separate padding-only kernel adds implementation and
launch complexity; splitting each export into multiple member-only copies adds
per-export traffic submission; altering Node layout or constructors affects the
public representation and many call sites. Extending the existing initialization
is the smallest correction and leaves all compiled production CUDA instructions
unchanged. It adds one startup clear to compact training exports; its cost must
be measured rather than assumed zero. Full-capacity export already pays this cost.

Validation: retain the failed run; repeat compact/bounded export in per-output
and batched tree construction under initcheck, including graph replay and a
tail output. Compare matched serialized live model values/predictions and the
ordinary correctness suite, preserve the source diff, and verify production
device instruction bytes. Any end-to-end performance assessment must use
uninstrumented matched workloads. No sanitizer suppression is proposed.
