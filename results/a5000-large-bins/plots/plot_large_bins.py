#!/usr/bin/env python3
"""Render the audited final five-shape comparison without querying a GPU.

Usage from the repository root:
  /tmp/histogram-scaling-plot-env/bin/python \
      results/a5000-large-bins/plots/plot_large_bins.py

Bars show the median of two fresh-seed native-time / selected-time ratios.
Whiskers span the two ratios; they are not confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator


HERE = Path(__file__).resolve().parent
EXPECTED_BINS = (24577, 32768, 65536, 262144, 1048576)
COLORS = {"shared_overflow": "#316DAA", "narrow_global": "#25846E"}


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    payload = path.read_bytes()
    report = json.loads(payload)
    if report["status"] != "complete" or len(report["cases"]) != 5:
        raise ValueError("Expected the complete, audited five-case final report")
    rows = []
    for case in sorted(report["cases"], key=lambda c: c["workload"]["bins"]):
        workload = case["workload"]
        expected = {
            "n": 16777216,
            "input": "u32",
            "counter": "u64",
            "distribution": "uniform",
            "order": "shuffled",
            "cache": "warm",
            "launch": "graph",
            "warmup_ms": 200,
        }
        if case["status"] != "complete" or any(workload[k] != v for k, v in expected.items()):
            raise ValueError(f"Unexpected workload/status: {case['case']}")
        confirmations = case["confirmations"]
        if len(confirmations) != 2 or len({c["seed"] for c in confirmations}) != 2:
            raise ValueError("Expected two distinct confirmation seeds")
        ratios = [c["native_median_us"] / c["chosen_median_us"] for c in confirmations]
        summary = {"min": min(ratios), "median": median(ratios), "max": max(ratios)}
        for key, value in summary.items():
            if not math.isclose(value, case["summary"]["native_over_chosen"][key], rel_tol=1e-12):
                raise ValueError(f"Ratio summary mismatch: {case['case']}/{key}")
        chosen = case["selection"]["chosen"]
        if chosen["algorithm"] == "shared_overflow" and chosen["local_counter"] == "u32":
            family = "shared_overflow"
        elif chosen["algorithm"] == "global" and chosen["local_counter"] == "u32":
            family = "narrow_global"
        else:
            raise ValueError(f"Unexpected selected family: {chosen}")
        rows.append({
            "bins": workload["bins"],
            "family": family,
            "chosen_variant": case["selection"]["chosen_variant"],
            "native_variant": case["selection"]["native_comparator_variant"],
            "confirmation_seeds": [c["seed"] for c in confirmations],
            "native_over_selected_by_seed": ratios,
            "summary": summary,
        })
    if tuple(row["bins"] for row in rows) != EXPECTED_BINS:
        raise ValueError("Unexpected bin counts")
    provenance = {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "metric": "native_median_us / selected_median_us within each confirmation invocation",
        "bar": "median of the two seed ratios",
        "whisker": "min/max of the two seed ratios; not a confidence interval",
        "production_defaults_promoted": False,
    }
    return provenance, rows


def render(rows: list[dict], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": "#333D48",
        "text.color": "#202B36",
        "xtick.color": "#536170",
        "ytick.color": "#333D48",
        "svg.fonttype": "none",
        "svg.hashsalt": "large-bin-final-confirmation",
    })
    fig, ax = plt.subplots(figsize=(11.4, 6.9), facecolor="white")
    fig.subplots_adjust(left=0.155, right=0.945, top=0.745, bottom=0.245)
    fig.text(0.055, 0.94, "Large-bin gains over native counters", fontsize=21, weight="bold")
    fig.text(0.055, 0.895, "16,777,216 u32 input values → u64 output · uniform shuffled data · warm graph", fontsize=11.5)
    fig.text(0.055, 0.86, "RTX A5000 Laptop · driver 597.06 · batch 4 · 200 ms warmup", fontsize=10.5, color="#536170")
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["shared_overflow"], label="Shared prefix + global overflow"),
            Patch(facecolor=COLORS["narrow_global"], label="Narrow global + u64 widening"),
        ],
        loc="upper left", bbox_to_anchor=(0.049, 0.829), frameon=False,
        ncol=2, fontsize=10.5, columnspacing=2.4, handlelength=1.3,
    )

    for y, row in enumerate(rows):
        stat = row["summary"]
        center = stat["median"]
        ax.barh(y, center, height=0.54, color=COLORS[row["family"]], zorder=2)
        ax.errorbar(
            center, y,
            xerr=[[center - stat["min"]], [stat["max"] - center]],
            fmt="none", color="#182A3B", capsize=5, capthick=1.3, elinewidth=1.5, zorder=4,
        )
        if math.isclose(stat["min"], stat["max"], rel_tol=1e-9):
            label = f"{stat['min']:.3f}×"
        else:
            label = f"{stat['min']:.3f}–{stat['max']:.3f}×"
        ax.text(stat["max"] + 0.055, y, label, va="center", fontsize=11)

    ax.set_yticks(range(len(rows)), [f"{row['bins']:,}" for row in rows])
    ax.invert_yaxis()
    ax.set_ylabel("Bins", labelpad=15, fontsize=11)
    ax.set_xlabel("Speedup over frozen native control  (native time / selected time)", labelpad=12)
    ax.set_xlim(0, 2.95)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.grid(axis="x", color="#E3E8EC", linewidth=0.8, zorder=0)
    ax.axvline(1.0, color="#74808A", linestyle=(0, (3, 3)), linewidth=1.0, zorder=3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    fig.text(0.055, 0.116, "Bars: median of two fresh-seed ratios. Whiskers and labels: min–max; not confidence intervals.", fontsize=10, color="#536170")
    fig.text(0.055, 0.08, "Controls were frozen before search; each ratio uses the same final executable and invocation.", fontsize=10, color="#536170")
    fig.text(0.055, 0.044, "Bounded uniform-workload result · clocks unlocked · automatic production defaults unchanged", fontsize=10, color="#536170")

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "native-speedup.png", dpi=180, facecolor="white", metadata={"Software": "plot_large_bins.py"})
    fig.savefig(output / "native-speedup.svg", facecolor="white", metadata={"Date": None, "Creator": "plot_large_bins.py"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=HERE.parent / "overflow-evaluation" / "overflow-analysis.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    provenance, rows = load_rows(args.analysis)
    render(rows, args.output_dir)
    plotted_data = {**provenance, "rows": rows}
    (args.output_dir / "plotted-data.json").write_text(json.dumps(plotted_data, indent=2) + "\n")
    print(f"Wrote native-speedup.png, native-speedup.svg, and plotted-data.json in {args.output_dir}")


if __name__ == "__main__":
    main()
