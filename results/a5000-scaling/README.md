# Larger histogram workloads

This campaign extends the previous 16,777,216-input / 16,384-bin performance range. It measures the existing custom CUDA histogram implementations and NVIDIA's histogram as a benchmark reference. It does not change the production library or its defaults.

The corrected campaign is under [complete-catalog/](complete-catalog/). An interrupted preliminary run remains in this directory, with its original runner and [reason for replacement](PRELIMINARY.md). Its measurements are not pooled into the corrected search, selection, or confirmation results.

## Results

All 29 workloads completed on the RTX A5000 Laptop GPU with NVIDIA driver 597.06. The [full audited report](complete-catalog/scaling-analysis.md) covers 174 invocations, 3,452 candidate measurements and 22,698 raw timing samples. Every candidate passed exact CPU-reference correctness checks before timing. The selected custom configurations beat NVIDIA's histogram in all 58 fresh confirmation comparisons, by **3.699–39.636×**.

These ranges contain the two confirmation-seed medians; they are not confidence intervals. Each timed operation includes initialization, counting and any final reduction. Input generation, allocation, host/device transfers and graph creation are excluded.

| Input elements | Bins | Selected custom time | NVIDIA time / custom time |
|---:|---:|---:|---:|
| 16,777,216 | 16,384 | 0.190–0.191 ms | 31.08–31.65× |
| 16,777,216 | 24,576 | 0.192 ms | 37.00–37.70× |
| 134,217,728 | 24,576 | 1.466–1.478 ms | 39.29–39.64× |
| 536,870,912 | 16,384 | 6.063–6.082 ms | 32.31–32.39× |
| 1,073,741,824 | 16,384 | 11.964–12.021 ms | 32.72–32.89× |
| 1,073,741,824 | 65,536 | 40.977–41.060 ms | 13.99–14.02× |
| 16,777,216 | 1,048,576 | 4.553–4.581 ms | 3.70–3.73× |

The largest input is 64 times the previous performance-test ceiling; the largest bin count is also 64 times the previous ceiling. These maxima were tested separately: the billion-input cases have 16,384 or 65,536 bins, and the million-bin case has 16,777,216 inputs. All results here use uniform shuffled u32 bin IDs, u64 outputs and warm graph execution.

The sharpest implementation boundary occurs at **24,576 → 24,577 bins**: with 16,777,216 inputs, custom time rises from **192.256 to 472.832 microseconds**, about **2.46×**. Our current shared-memory policies stop at 96 KiB, which holds 24,576 four-byte local counters. The next bin count uses a global-atomic kernel. This is the current kernel catalog's capacity, not a measured physical hardware maximum. Above this boundary, the best measured families are global or warp-based counting. At the billion-input scale, logical input throughput is about 357–359 GB/s for 16,384 bins and 105 GB/s for 65,536 bins; these figures are input bytes divided by complete-operation time, not measured DRAM bandwidth.

**Production defaults are unchanged.** Against the existing default measured in the same invocation, selected configurations recorded 39 wins, 12 losses and 7 equal medians. Six equalities reuse the identical configuration; the seventh is an equal recorded median for distinct configurations. Sixteen workloads improve on both seeds, two regress on both, seven reverse direction, three select the default itself, and one loses on one seed and ties on the other. Marginal improvements and validation winners that regress on confirmation are not evidence for safe default promotion. The clearest larger-bin candidate for a separate default-promotion study is 16,777,216 inputs with 24,575–24,576 bins: shared policy 15 with 24 blocks is about 1.19× as fast as the generic default on both confirmation seeds. Other distributions and execution modes would still need validation.

Clocks were unlocked, and material variation remains. The unchanged default at 268,435,456 inputs / 16,384 bins measured 4.108 and 3.198 ms across confirmation seeds. Both are retained; telemetry snapshots do not establish a cause. This experiment neither proves a universal optimum nor resolves the earlier strict zero-loss acceptance question for a different implementation campaign.

The next performance investigation should profile the larger-bin kernels, especially the 24,577-bin transition and million-bin workload, to explain the lost throughput before changing the algorithm. NVIDIA remains a benchmark reference only.

[Throughput chart (PNG)](complete-catalog/figures/throughput.png) · [SVG](complete-catalog/figures/throughput.svg) · [Speedup and bin-boundary chart (PNG)](complete-catalog/figures/speedup-by-bins.png) · [SVG](complete-catalog/figures/speedup-by-bins.svg). The [machine-readable audit](complete-catalog/scaling-analysis.json) preserves raw samples, exact commands, hashes and telemetry for every measurement.

## Scope

- Input sizes 2^24, 2^25, 2^26, 2^27 and 2^28, each with 16,384, 24,576, 32,768 and 65,536 bins.
- Additional 2^24-input cases with 24,575, 24,577, 131,072, 262,144 and 1,048,576 bins.
- Additional 2^29- and 2^30-input cases with 16,384 and 65,536 bins.
- All 29 workloads use uniform shuffled u32 bin IDs, u64 outputs, warm graph execution, and complete output initialization plus counting. "Warm" means repeated buffers; large inputs do not fit in the GPU's last-level cache.

The search includes every distinct currently supported kernel policy for these workloads at grids of 24, 48, 96, 192, 384, 768 and 1,536 blocks. Global and warp-aggregated policies 0–4 are distinct; policy 5 duplicates policy 2 for these kernels. At these bin counts, only shared policies 14 and 15 with u32 local counts fit, and only through 24,576 bins. All four shared update/reduction families are included where valid. Larger grids extend the prior 384-block search boundary. No new compile-time kernel policies are added.

## Measurement and selection

The same preserved benchmark binary is used throughout, SHA256 `d7568b5c93b80fbe39002c82866d08f2c75683b92ae99869d920c9896c481de7`. Each case has an automatic-default check, a search, validation on two new seeds, a frozen custom selection, and confirmation on two further seeds. Every candidate is checked against exact CPU counts before timing, including repeated output overwrite and graph replay.

Protocol 3 measures four complete operations per timing sample, after 200 ms requested warmup. Search uses five samples; validation eleven; confirmation twenty-one. The three-sample automatic invocation identifies the concrete production default. That default is then measured in the same validation/confirmation invocations as the candidates, so its initial timing is not used as an unmatched performance baseline. Batch four differs from earlier batch-32 campaigns; earlier absolute timings are not treated as matched comparisons.

NVIDIA remains a measurement reference and cannot be selected. The frozen choice is the custom validation winner; its subsequent confirmation can expose a selection that did not improve on the default. All such results must remain in the report. No universal optimum, zero-regression proof, or applicability to other input distributions is implied.

GPU commands run serially. CSVs, raw timing samples, exact commands, executable hashes, GPU UUID/driver identity and before/after telemetry are retained. The manifest declares the candidate set and all seeds before measurement. CPU-only analysis verifies those artifacts and reproduces the selection rules.

```bash
python3 results/a5000-scaling/run_scaling.py --output-root results/a5000-scaling/complete-catalog
python3 results/a5000-scaling/analyze_scaling.py --output-root results/a5000-scaling/complete-catalog
```

The runner reuses compatible completed records and refuses incompatible or partially written records. Analysis requires complete records unless `--available-only` is explicitly supplied. The final audit passed all 29 cases; partial reports are retained as historical snapshots and do not replace the final report.

Charts are generated from the complete audited JSON with `plot_scaling.py`. Rendering used Python 3.14 and Matplotlib 3.11.2 in an isolated temporary environment, after GPU measurements finished:

```bash
python3 -m venv /tmp/histogram-scaling-plot-env
/tmp/histogram-scaling-plot-env/bin/python -m pip install matplotlib==3.11.2
/tmp/histogram-scaling-plot-env/bin/python results/a5000-scaling/plot_scaling.py
```
