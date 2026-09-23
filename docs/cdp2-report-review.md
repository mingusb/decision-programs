# CDP2 research report: evidence review

Reviewed 2026-09-23 without building, querying the GPU or executing workloads.
Only existing receipts, tool `--version`/`--help`, installed documentation and
current primary web sources were inspected.

Report: `/mnt/c/Users/refle/Documents/Profiling and Validating CUDA Dynamic
Parallelism 2 on Ampere under WSL2.pdf`, 30 pages, SHA-256
`5b8ff58cbd8537307faffd37771f0c03045624c6c0ce212d09738ca8380b0f2e`.
Reviewed text: `/tmp/gh-cdp2-research.txt`, SHA-256
`f483975a73ebd640e9f5bd30f8c7dad104a1e4467e4466731e4c3d69f3e94b5d`.

The report's main recommendation is sound: rank complete uninstrumented
operations, profile actual leaf kernels through ordinary launches, and test CDP2
ordering independently. Its blanket Nsight Systems claim and Nsight Compute
version pairing need correction. Tool activity must remain distinguished from
child attribution and comprehensive correctness coverage.

## Corrections and unresolved conflicts

**Nsight Systems does record our host parent.** Installed version is
`2026.3.2.313-263238521929v0`. The retained
[SQLite export](../observations/diagnostics/nsys-observe-1/export.sqlite) has one
`CUPTI_ACTIVITY_KIND_KERNEL` row named `gh::test::run()`, with start 899533856 and
end 899655171. Therefore the report's pages 1–2 claim that no kernel trace is
usable on this Ampere CDP application is too broad. The
[Systems release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#cuda-trace-issues)
still contain that broad limitation; it conflicts with the observed parent row.
The more precise [CUPTI CDP2 limitation](https://docs.nvidia.com/cupti/release-notes/release-notes.html#known-issues)
allows host-parent tracing and excludes device-launched children. Retain the
parent record, do not infer a child timeline, and do not interpret its timestamp
span as the entire operation without a separate timing contract.

**NCU 2026.3 already supports CUDA 13.4.** The installed executable
`/opt/nvidia/nsight-compute/2026.3.0/ncu --version` reports
`2026.3.0.0 (build 38525999) (public-release)`, not Update 1. Both its bundled
`docs/ReleaseNotes/index.html` and NVIDIA's
[release history](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)
pair 2026.3 with CUDA 13.4; Update 1 pairs with CUDA 13.4 Update 1. Thus the
report's suggested base-version incompatibility is unsupported. An update may
still fix specific issues, but is not a prerequisite inferred from this pairing.
Our [third NCU receipt](../observations/diagnostics/ncu-observe-3/receipt.json)
contains a parent record with `launch__uses_cdp=1` and
`sm__ctas_launched.sum=2822`. These are call-tree metrics, not parent-only counts.
Earlier failures remain: receipt 1 missed the `.ncu-repz` extension; receipt 2
did not recognize the wide CSV layout. Neither demonstrates CUDA incompatibility.

**NVBit's driver ceiling is still documented.** Installed NVBit 1.8 README line
44 and the [current upstream README](https://github.com/NVlabs/NVBit#requirements)
both state driver <=575.xx; the [1.8 release](https://github.com/NVlabs/NVBit/releases/tag/v1.8)
also records bundled CUDA 13.2 headers. No inspected primary source removes the
ceiling, so calling it a corrected/stale requirement would overstate evidence.
Conversely, it does not justify claiming our actual attempt failed: the
[NVBit receipt](../observations/diagnostics/nvbit-count-observe-1/receipt.json)
records 27,020 instructions under `gh::test::run()` and successful completion on
the reported 616.92 stack. This establishes limited local operation beyond the
stated ceiling, not supported WSL/CDP2-child coverage.

## Findings supported by our evidence

- CUPTI trace reports one host kernel; legacy range injection reports one range
  and 2,823 CTAs. PC decoding has two PC records, both for the parent symbol,
  23,683 total samples and 23,681 non-user-kernel samples. These are actual
  [retained activities](../observations/diagnostics), with no child-specific
  coverage proof. Parent metrics include descendant work under CUPTI's contract.
- Initcheck, racecheck and synccheck explicitly refused CDP in the
  [retained runs](../observations/sanitizer). This is also an explicit
  [current release-note limitation](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations),
  not merely an unexplained local discrepancy. Memcheck does not validate
  device-side CUDA API return errors; our immediate launch-status checks matter.
- A clean memcheck attempt does not prove a particular child was inspected.
  Require a deliberately faulty child to be named in a negative control, while
  preserving actual clean production results and known internal-attribute errors.
  The report's `--report-api-errors all` is a valid installed option, but expands
  scope beyond our explicit-only application-error run; keep attempts separate.
- [CDP2 tail visibility](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
  supports our poison/guard/oracle continuations. Forcing CDP1 would change the
  executor. Existing raw `%globaltimer` spans remain the versioned benchmark
  contract; the report's CUDA-event/host-wall fields cannot rename those spans.

## Ranked next experiments

1. **Finish the eight-case uninstrumented benchmark.** Preserve every sample and
   independent check, empty/A–A controls and frozen/fresh p15 comparisons. Diagnose
   control drift before ranking. This directly answers the current performance
   question without depending on child profiler visibility.
2. **Prove checker coverage, then profile identical leaf code.** A bounded faulty
   CDP child must produce a named memcheck finding. For the other three checkers,
   use GPU fixtures/oracles and ordinary host launches of the actual production
   leaf object, with checker-specific negative controls. Current translation units
   mix leaves and device dispatch: isolate without copying arithmetic, exclude
   CDP launch paths from that executable, and compare final SASS identities.
   Collect NCU leaf resources/traffic/stalls afterward; these timings do not rank
   production CDP operations.
3. **Measure the launch floor and tail ordering.** Compare no-child, one-thread
   empty, geometry-matched empty and real-child controls with identical GPU
   decisions. Record enqueue returns, final epochs and sparse device markers.
   Query/set/read back pending-launch limits through runtime setup before launches;
   sweep only after a queue/resource hypothesis. Do not subtract these timings
   as if runtime, overlap and leaf costs were necessarily additive.
4. **Prove debugger/NVBit child attribution.** Use one known parent→child→tail
   probe. Require a CUDA-GDB child stop plus
   [launch ancestry](https://docs.nvidia.com/cuda/cuda-gdb/index.html#info-cuda-launch-children),
   or an injected child-only counter/sentinel for NVBit. Existing parent receipts
   do not satisfy either criterion.
5. **Then test alternative executors.** Batch, fuse or use persistent execution
   only where the complete-operation evidence identifies launch overhead. Check
   current graph/CDP composition restrictions before implementation. Preserve
   numerical order/quality gates and account for queue/scratch/residency costs;
   the report supplies candidates, not a local speed ranking.

No profiler upgrade, application rewrite or failed-gate waiver follows from this
review. Rolling online manuals and installed releases are identified separately;
the retained local artifacts control claims about what actually happened here.
