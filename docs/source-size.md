# Source inventory and counting contract

This is a source-only audit, not evidence of equal capabilities or a completed
replacement. Do not publish a reduction percentage until the applicable entries
in `CAPABILITIES.md` have matching correctness, numerical, quality, resource and
performance evidence. Passing the nine initial integrated GPU suites is useful
component evidence; it does not establish that equivalence.

`tools/source_size.py` reads both trees without modifying, compiling, importing
their application modules, or executing GPU code. The original tree is
`/home/b/gpu_histogram-archive-20260923`. Its default fresh tree is this checkout.
The script includes itself. The same normalization and token rules apply to both.

```sh
python3 tools/source_size.py --self-test
python3 tools/source_size.py --output /absolute/new/source-size.json
```

Output files must be new; previous receipts are never overwritten. Each receipt
contains original/normalized file SHA-256, raw bytes, exact tool hash, Python and
formatter versions, scope, per-file counts and category/language totals. A file
changing during a tree's scan is an error. Later edits require a new receipt.

## Exact definitions

- **Formatted noncomment LOC** counts normalized physical lines containing at
  least one non-whitespace character. Braces, directives and literal-content
  lines count. C/C++/CUDA comments are lexically replaced with whitespace before
  clang-format 21; literals containing comment markers remain intact. The fixed
  LLVM-derived style has width 100, indentation 2, no include sorting, and no
  one-line functions/if/loop/block compression. The complete style is in the
  receipt. Includes/macros/templates remain unexpanded and uninstantiated.
- Python uses the recorded interpreter's `ast.parse`/`ast.unparse` normalization.
  Comments are absent; docstrings remain string literals and count. There is no
  claim that Python and C++ LOC measure identical semantic work. Python 3.14.4
  was used for the initial receipt.
- CMake uses a specified mechanical normal form: discard comments, space lexical
  tokens, end each outer parenthesized command on its own line, and wrap at 100
  columns with a two-space continuation. Quoted/bracket arguments are indivisible.
  JSON registries use two-space JSON formatting. These rules normalize counting;
  they do not rewrite the actual sources or purport to be build-system parsers.
- **Lexical tokens** count the normalized unexpanded source. C-family/registry
  tokens are identifiers, preprocessing-style numbers, quoted/character/raw
  literals (including attached suffixes), longest listed operators, or one other
  punctuation character. CUDA `<<<` and `>>>` each count as one. Include header
  names in angle brackets are tokenized, not collapsed. Python uses `tokenize`,
  excluding encoding/end, comments, indentation and newline tokens. This is a
  documented source-token measure, not compiler semantic tokens or model tokens.
- Literal spelling is also reported separately as original UTF-8 bytes,
  physical spelling lines, and `\w+|[^\w\s]` word/punctuation matches. These are
  transparency counters, not decoded literal values. Thus a code generator's
  large string template is visible even when ordinary lexical counting treats it
  as one token. Generated maintained source, generators, registries, `.inc`,
  `.def`, `.inl`, `.ipp` and `.tpp` files all count when present in scope. Identical
  files are listed and counted separately; no deduplication hides maintained text.

The development self-test checks raw strings/comment markers, continued comments,
digit separators, Python docstrings, CMake quoted/bracket text and forced expanded
C++ formatting. It passed. It is development source analysis, not a CPU
application/test oracle. Unknown file extensions inside source roots appear in
`unclassified_files_requiring_review` and make the audit exit nonzero.

## Scope and initial receipt

Production roots are `include`, `src`, `training/include`, `training/src`.
Tests are `tests`, `training/tests`, including their shared drivers. Tooling is
`tools`, `bench`, `support`, `cmake`, `training/tools`, `training/bench`,
`training/profiling`, plus top-level/training CMake files. `support` is tooling:
the archived common/profiling headers serve benchmarks/checks, not library code.
The archived `training/src/batch_training.inc`, counting default/tuning tables,
benchmark reference adapters, `bench/preservation/prepare_sources.py`, and all
selected campaign/evaluation scripts are included.

Fourteen top-level Python files in
`results/booster-level-batch-20260922/real` are an explicit separate
**capability-references** category, including `evaluate.py`, `evaluate_cached.py`,
fixtures, framework, campaign, summaries and their checks. They are not silently
counted as missing fresh production LOC. Other results, snapshot copies, build
outputs, vendor/third-party source and binaries, documentation, caches and
datasets are outside the declared scope. Excluded directory names and exact
selected paths are recorded. Broader historical experiment scripts remain
evidence; claiming their capabilities replaced requires adding their scope or
mapping them explicitly, not assuming they disappeared.

Initial final receipt: `/tmp/gh-source-size-final.json`, SHA-256
`b266bab56b8f6d0c044cd7b4027041ec2cf8123c9bf4871c165d8585a2a46a8c`.
`/tmp/gh-source-size-final.stdout` and `.stderr` retain the invocation result;
exit status was zero, with no unclassified source-root files. The earlier
`/tmp/gh-source-size-initial.*` receipt remains preserved; it originally classified
two benchmark support headers as production, and the final receipt corrects that
classification. Root should preserve these receipts with the other observations.

| Category | Baseline files | Baseline LOC | Baseline tokens | Fresh files | Fresh LOC | Fresh tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production | 39 | 8,241 | 79,914 | 21 | 5,320 | 54,455 |
| Tests/drivers | 30 | 12,675 | 139,586 | 12 | 3,501 | 32,809 |
| Tooling/build | 40 | 9,722 | 102,280 | 4 | 548 | 6,266 |
| Capability references | 14 | 1,343 | 19,959 | 0 | 0 | 0 |

These are absolute counts of the receipt's source snapshots. Work continued
afterward, including benchmark/quality runners, the enlarged metric-test arena,
NVTX bootstrap ranges and additional trace checks. This table is not a claim
about the later HEAD or a matched-capability improvement; rerun after freezing it.

## Bounded source and capability audit

A declaration/reference scan plus targeted reads found no confirmed uncalled
production helper in the current library. Every inspected helper had a source
caller or deliberately exercised public/test path. This is not a whole-program
dead-code proof: overloads, runtime branches, templates and variant coverage need
compiled call/code-generation evidence. Public result fields are not dead merely
because their consumer is a test/report. For example, tuning samples and selected
policies are checked, and memory/capacity fields are exercised as budget reports.
No production deletion was made during this audit.

The largest original training features were compared against `booster.hpp`,
`booster.cpp`, `batch_training.inc`, the resident/root/deeper/split modules and
their current replacements:

| Original contract / inventory | Fresh source status | Evidence or capability still needed |
| --- | --- | --- |
| Exact fitting, complete categories, weighted regression/binary/multilabel/coupled multiclass, regularization and loss history; D1–D3, T1 | `data` and `learn` entrypoints and independent component checks exist | Full original workload/shape and frozen numeric/quality comparisons, including all large output domains |
| Root batching/count reuse, output tiles, shared/global/automatic histograms, three split schedules, batched tree construction; T2–T3 | Explicit fresh policies and shared implementation exist | Actual operation campaigns and all policy/default numerical/performance gates; a compact common body does not inherit archived timings |
| Orders 3/4, clipping and bounded derivative/frontier storage; T4 | Fresh higher-order solver and capacity checks exist | Full validation-grid selection, matched order-2 zero gates, repeated held-out signal evaluation and time-to-quality protocol from `higher_order_campaign.py` |
| Bounded incremental compact-tree/pinned-batch export | New training retains the complete forest resident, with separate GPU model serialization | Demonstrate accepted large-forest capacity and complete export-inclusive behavior; old pinned-batch behavior is not reproduced by the resident API alone |
| Stream/direct-leaf graph execution and prediction policy comparisons; C3/T3/M2/P2 | CDP executor; count host graph-of-CDP replay; resident inference | Explicit topology/capacity differences and complete setup + Q calls + teardown comparisons. Do not relabel CDP spans as old graph/CUDA-event measurements |
| Synthetic/real metrics and validation-frozen signal selection; Q1–Q5 | Both profiles, gates and signal primitives have component suites | Frozen prediction-file conformance, NumPy reduction dispatch and host/libdevice transcendental agreement; no tolerance waiver. Full campaign integration is separate from metric-unit success |
| Diagnostics, recording and benchmark orchestration; P1–P3 | GPU stamps/statistics, collectors and root-added NVTX bootstrap range exist; actual operation runners are being implemented | Per-operation ranking evidence, full comparison controls, and actual tool activity. CDP-unsupported init/race/sync sanitizer modes require separately exercised direct-leaf routes, not an unsupported result counted as clean |

Removing host training/preprocessing/prediction computation and replacing old
container/CLI contracts with the accepted simpler resident API are authorized
design changes. They are not missing mathematical features. Conversely, preserved
file formats, all requested learning/counting capabilities, resource limits,
quality criteria and deliberately exercised comparisons remain acceptance work.
The current completion inventory explicitly remains unproven until those gates
are closed; source-size totals must not become a substitute for that evidence.
