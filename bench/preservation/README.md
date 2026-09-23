# Same-process preservation diagnostic

This standalone harness investigates the two configurations flagged by the
large-bin implementation's old/new measurements. It compares **two specific
archived source revisions**, not whichever production source happens to be in
the current working tree:

- Old: `results/a5000-custom-only/source.tar.gz`.
- New: `results/a5000-large-bins/final-source.tar.gz`.

The source preparer validates archive and individual file hashes before staging
both revisions in the build directory. Complete custom libraries compile under
distinct `gh_old` and `gh_new` namespaces. Equivalent adapters store prepared
configurations outside timing and call each revision's full `histogram` entry
point during stream execution. Both libraries share one dynamic CUDA runtime.
No NVIDIA reference histogram is linked. Production sources, automatic defaults,
and the preserved benchmark binaries are unchanged by this diagnostic.

```bash
cmake -S bench/preservation -B build/preservation-paired -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build/preservation-paired -j 3

# CPU-only schedule inspection:
python3 tools/run_paired_preservation.py --dry-run

# Serial GPU campaign; refuses changed/partial evidence:
python3 tools/run_paired_preservation.py

# CPU-only audit of a completed campaign:
python3 tools/run_paired_preservation.py --audit
```

The primary campaign freezes 24 invocations: two cases, two data seeds, two
execution-order seeds, and three comparisons (old/new, old/old, new/new). The
second order seed reverses the old/new slot mapping. Each invocation uses 32
balanced ABBA/BAAB quartets, 32 complete histogram operations per timed position,
and 200 ms of alternating warmup. All timing positions are retained.

| Case | Workload | Explicit configuration |
|---|---|---|
| `single` | 1,048,576 identical u32 values, 256 u32 counts | shared policy 6, 192 blocks, warm graph, kernel clearing |
| `stream4096` | 1,048,576 uniform shuffled u32 values, 4,096 u32 counts | shared policy 10, 48 blocks, warm stream, runtime clearing |

Both slots use the same device input/output addresses and nonblocking stream.
Every measured position checks the output against independent CPU counts and
checks output guards. The driver context is checked after each position; context,
stream, and backend function addresses are recorded. The old/old and new/new
controls bind both slots to exactly the same backend function, with independent
captured graph instances where applicable.

GPU event times include clearing and counting. Graph event nodes are captured
inside the batch. Stream event timing spans the full batch of API submissions,
matching the earlier timing method. CPU submission and enqueue-through-completion
durations are recorded separately and must not be added to GPU event durations.
One untimed launch precedes each measurement; pending warmup work can contribute
to the host completion interval.

This diagnostic changes executable layout and CUDA registration by rebuilding
both libraries into one process. Per-position verification also adds a device-to-
host copy and synchronization **outside** the timed interval, unlike the original
benchmark. These changes are controlled across slots but prevent treating the
new absolute timings as replacements for historical benchmark results. GPU
clocks remain unlocked. The single-valued data seeds generate identical inputs.

Analysis uses quartets as matched clusters and reports process/order variation
alongside same-backend controls. Near-unity ratios or overlap with control noise
do not prove zero performance loss. This harness does not tune or promote a
production default.

## Matched synchronization experiment

The new experiment builds into **`build/preservation-sync`** and records results
in a separate directory. Do not rebuild the frozen `preservation-paired` binary
or rerun its historical campaign with the changed harness source. Existing
legacy behavior and CSV columns remain the default (`--sync-mode legacy`);
legacy source hashes intentionally no longer match the current edited source.

```bash
cmake -S bench/preservation -B build/preservation-sync -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build/preservation-sync -j 2

# CPU-only bounded schedule inspection and evidence rejection tests:
python3 tools/run_sync_preservation.py --dry-run
python3 tests/test_sync_preservation.py

# Root runs these GPU invocations serially, after smoke and sanitizer checks:
python3 tools/run_sync_preservation.py

# CPU-only strict audit; preserves the original raw measurements:
python3 tools/run_sync_preservation.py --audit
```

`--sync-mode position` and `--sync-mode quartet` use the same four timing event
pairs and four captured graphs (for the graph case): slot A's first/second
occurrences have IDs 0/1, and slot B's have IDs 2/3. Each measured quartet uses
every pair once. Untimed graph warmup replays precede measured replays of the
same occurrence; only the final measured timestamps are read. A slot's second
measured occurrence cannot overwrite its first occurrence's events.

Both modes allocate four equally sized **pinned** host output snapshots before
measurement. A complete output-and-canary copy is enqueued after each timing end
and before the common device output is overwritten. Position mode synchronizes,
checks that snapshot, and reads its events after each position. Quartet mode
queues all four positions and copies before one stream synchronization, then
reads all four event pairs and validates all four snapshots against independent
CPU counts. No event read or host synchronization occurs between positions in
quartet mode. The same CUDA context is checked with each snapshot validation.

The explicit matched-mode CSV extends the legacy fields with `sync_mode`,
`snapshot_storage`, `timing_pair`, `quartet_host_us`, and `position_host_scope`:

- `event_us`: complete histogram batch event duration divided by batch. Snapshot
  copies and CPU checking are outside this interval.
- `submit_us`: timed batch submission duration divided by batch, excluding the
  snapshot-copy submission.
- `total_host_us`: position mode's timed batch enqueue through completion of its
  pinned snapshot copy, divided by batch. It includes waiting for preceding
  untimed work and the snapshot copy. It is **blank** in quartet mode because no
  per-position host completion is observed. Its scope differs from legacy's
  completion field, which ends before output copying.
- `quartet_host_us`: microseconds from before the first untimed launch until
  synchronization completes the fourth snapshot copy, with no batch division.
  The same value appears on all four rows and represents **one** observation.
  Position mode includes the first three positions' CPU checking and event
  reads between submissions, while both modes exclude checking after the fourth
  snapshot completion. This measures the host-visible effect of the entire
  synchronization/checking policy; it is not pure GPU service time.

The frozen initial screen has 64 invocations: two cases × two AA bindings
(`old-old`, `new-new`) × four batch sizes (32, 64, 128, 256) × two matched modes
× two process repetitions. Each invocation retains all 32 balanced ABBA/BAAB
quartets (128 timing positions), with 200 ms alternating warmup. Modes for the
same case/binding/batch/repetition are adjacent processes with identical input
and order seeds; their order reverses on the second repetition. Pairs are
shuffled using a fixed seed. The complete screen has 2,048 quartet clusters and
8,192 raw positions.

The runner freezes executable, archived sources and source provenance, harness,
runner/audit dependencies, build files and static libraries. It records exact
commands, exit status, before/after GPU identity and telemetry, raw CSV and
logs, and per-invocation hash receipts. It rejects partial or altered evidence
and never overwrites raw measurements. Its audit keeps **case, AA binding,
batch, and synchronization mode separate**, leaving two process repetitions per
stratum. It counts quartet completion once, excludes unobserved position
completion from summaries, and retains every AA quartet ratio. This is a
bounded descriptive screen, not proof of a precision plateau or zero loss.
Choose any later old/new confirmation campaign separately from these screening
results and record that decision before collecting confirmation data.
