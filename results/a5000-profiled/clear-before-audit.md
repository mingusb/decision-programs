# Output-clear graph topology audit

We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.

The fresh node trace supports testing an explicit CUDA clear kernel: the repeated
memset/count graph spends most of its measured node span between activities,
while CUB and shared-partial graphs have short kernel-to-kernel transitions.
The trace identifies a scheduling/topology symptom, not its driver or hardware
cause. In particular, it does not establish that the memset uses a copy engine.

Evidence: [timeline-clear-before.sqlite](timeline-clear-before.sqlite), its
[command](timeline-clear-before.command.json), and
[profiler/benchmark log](timeline-clear-before.log). SQLite SHA256:
`e4ac0a1b436c4951ddf856e010da80924a939c4b48c2662400b3238409084b14`.

The invocation uses N=1,048,576, u8 input, 256 bins, u32 output, uniform shuffled
data, warm graph mode, seed 12345, protocol 3, five sample rounds, and 20 complete
operations per graph. Variants are `cub:2:192`, `shared:7:96`, and
`shared_partial:7:96`. Both custom variants use vector4 loads, 256 threads, eight
items per thread, one shared replica, and native local counters.

## Matching and measured-graph results

`CUPTI_ACTIVITY_KIND_RUNTIME.nameId` joins `StringIds.id` to identify
`cudaGraphLaunch` calls. Each candidate NVTX timing range contains exactly two
calls on its host thread: warmup followed by measurement. Taking the second call
selects the measured graph. Its `correlationId` matches the graph's kernel and
memset activities. This excludes the two graph-validation launches per candidate
and every timing-round warmup. Kernel names come from
`CUPTI_ACTIVITY_KIND_KERNEL.demangledName -> StringIds.id`; memory operations come
from `CUPTI_ACTIVITY_KIND_MEMSET`.

Each selected graph has 40 activities: 20 complete operations. Node span is the
last activity's end minus the first activity's start. Active time sums those
activity durations; gap time sums the intervals between adjacent activities.
These intervals contain no overlap in this trace. They exclude graph boundary
events and are not interchangeable with the benchmark's CUDA-event intervals.

| Family | Measured graph correlation IDs | Median node span, µs | Median active time, µs | Median gap time, µs |
|---|---|---:|---:|---:|
| CUB | 941, 957, 969, 985, 989 | 140.420 | 131.556 | 8.736 |
| Shared atomic | 945, 953, 973, 981, 997 | 321.738 | 110.021 | 211.941 |
| Shared partial | 949, 961, 965, 977, 993 | 155.557 | 146.661 | 8.928 |

Columns are independently computed medians and need not add exactly. The shared
graph node spans range from 297.706 to 407.884 µs. CUB spans range from 139.844 to
140.997 µs; partial spans range from 155.461 to 155.589 µs. The corresponding
profiled benchmark medians, including its event boundary, are 16.2608, 7.2624,
and 8.0384 µs per operation for shared, CUB, and partial respectively. These are
diagnostic profiled measurements, not unprofiled performance rankings.

Pooling only transitions within the five measured graphs gives:

| Transition | Count | Minimum gap, µs | Median gap, µs | Maximum gap, µs |
|---|---:|---:|---:|---:|
| CUB init → sweep | 100 | 0.064 | 0.096 | 0.097 |
| CUB sweep → next init | 95 | 0.064 | 0.096 | 3.072 |
| 1,024-byte memset → shared counting | 100 | 1.664 | 1.776 | 55.426 |
| Shared counting → next memset | 95 | 1.248 | 1.441 | 42.466 |
| Partial counting → reduction | 100 | 0.064 | 0.096 | 0.096 |
| Reduction → next partial counting | 95 | 0.064 | 0.096 | 3.072 |

The median individual shared counting kernel is 4.385 µs, and the median memset
activity is 1.152 µs. The partial counting and reduction kernels have medians
3.680 and 3.648 µs. CUB initialization and sweep medians are 1.056 and 5.504 µs.
The direct shared counting kernel is slightly slower than its partial-writing
counterpart, but the large complete-operation penalty comes from the intervening
activity gaps. It cannot be explained by summing the shared graph's active work.

The export has 72 CUDA event records, all with device timestamp zero. Therefore
it does not independently reproduce the internal event timing boundaries.
Its memset activity table also has no execution-engine identifier. Tracing may
affect scheduling; these findings justify an unprofiled controlled ablation,
not a claim of a diagnosed WSL or copy-engine defect.

The older cold single-operation graph trace at B=4,096/u64 had 32,768-byte memset
activities lasting 1.60–2.14 µs and memset→shared gaps of only 0.256–0.288 µs.
That earlier observation did not establish a general handoff penalty. The fresh
trace is the directly relevant repeated warm topology; multiple parameters
differ between those two traces, so their contrast does not isolate one cause.

## Source topology and minimal ablation

The original custom atomic-merge and bit-plane operations use
`cudaMemsetAsync(output)` followed by a counting kernel. Shared partial histograms
overwrite scratch and then overwrite output in their reduction, requiring no
output memset. Installed CUB 3.4.2 explicitly launches `DeviceHistogramInitKernel`
and `DeviceHistogramSweepKernel`; the initializer zeros output bins in
`include/cccl/cub/device/dispatch/kernels/kernel_histogram.cuh`. This matches the
activity names in the trace.

The archived `build/profiled-loads` binaries and source preserve the old protocol-3
implementation. The bounded source change replaces only the nonempty custom
counting-path output memset with a typed, bounds-checked, grid-stride CUDA zero
kernel. It writes exactly the dense output, uses its original stream, and adds no
allocation or attribute query. Empty input, CUB, the NVIDIA sample, partial
histograms, and every counting kernel remain unchanged.

Compare those old/new binaries with identical workload, candidates, input seed,
warmup, sample count, batch size and protocol 3. Include CUB and shared partials as
unchanged controls; repeat batch 1/5/20 before accepting a stable improvement.
Profile the new warm batch with node tracing to check whether activity gaps
contract while counting-kernel times remain similar. An improvement would support
the explicit-kernel initialization choice for this stack; it would not by itself
identify the internal execution engine or establish performance on other drivers.

## Reproduce the SQLite analysis

Run from the repository root. The script opens SQLite read-only and prints both
per-measured-graph records and the aggregate tables above.

```bash
python3 -B - <<'PY'
import collections
import hashlib
from pathlib import Path
import sqlite3
import statistics

path = Path('results/a5000-profiled/timeline-clear-before.sqlite')
db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
db.row_factory = sqlite3.Row
print('sha256', hashlib.sha256(path.read_bytes()).hexdigest())
launches = list(db.execute('''
    SELECT r.*, s.value AS api
    FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
    JOIN StringIds AS s ON s.id = r.nameId
    WHERE s.value LIKE 'cudaGraphLaunch%'
'''))
activities = collections.defaultdict(list)
for row in db.execute('''
    SELECT k.*, s.value AS kernel_name
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    JOIN StringIds AS s ON s.id = k.demangledName
'''):
    name = row['kernel_name']
    for tag in ('DeviceHistogramInitKernel', 'DeviceHistogramSweepKernel',
                'shared_histogram_loaded', 'reduce_partials'):
        if tag in name:
            name = tag
            break
    activities[row['correlationId']].append((row['start'], row['end'], name))
for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET'):
    name = 'memset' + str(row['bytes'])
    activities[row['correlationId']].append((row['start'], row['end'], name))

graphs = collections.defaultdict(list)
gaps = collections.defaultdict(list)
durations = collections.defaultdict(list)
for nvtx in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
    calls = [call for call in launches
             if call['globalTid'] == nvtx['globalTid']
             and nvtx['start'] <= call['start'] <= nvtx['end']]
    if not calls:
        continue
    assert len(calls) == 2, (nvtx['text'], len(calls))
    measured = max(calls, key=lambda call: call['start'])
    events = sorted(activities[measured['correlationId']])
    assert len(events) == 40
    family = nvtx['text'].split(':')[0]
    span = (events[-1][1] - events[0][0]) / 1000
    active = sum(end - start for start, end, _ in events) / 1000
    graphs[family].append((span, active, span - active))
    print('graph', family, measured['correlationId'], span, active, span - active)
    for before, after in zip(events, events[1:]):
        gap = (after[0] - before[1]) / 1000
        assert gap >= 0
        gaps[(family, before[2], after[2])].append(gap)
    for start, end, name in events:
        durations[(family, name)].append((end - start) / 1000)

for family, rows in graphs.items():
    print('graph medians', family, len(rows),
          [statistics.median(row[i] for row in rows) for i in range(3)])
for kind, table in (('transition', gaps), ('activity', durations)):
    for key, values in table.items():
        print(kind, key, 'count/min/median/max', len(values),
              min(values), statistics.median(values), max(values))
print('nonzero device event timestamps', db.execute('''
    SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT WHERE timestamp != 0
''').fetchone()[0])
PY
```
