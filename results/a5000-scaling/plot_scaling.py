#!/usr/bin/env python3
"""Plot audited scaling confirmation results; no CUDA or GPU access.

Requires matplotlib. Example, after the complete analyzer report exists:
  python plot_scaling.py complete-catalog/scaling-analysis.json

Writes throughput.svg/.png and speedup-by-bins.svg/.png next to the report,
inside figures/ unless --output-dir is supplied. Only final, complete reports
are accepted. Whiskers show the two confirmation-seed medians, not confidence
intervals; centers are their median. NVIDIA histogram is a benchmark reference only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parent
BASE_BINS = (16384, 24576, 32768, 65536)
BOUNDARY_BINS = (24575, 24576, 24577)
SLICE_N = 1 << 24
CUSTOM = "#27658B"
REFERENCE = "#777F88"
TEXT = "#26323B"


def load_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != 1
            or report.get("kind") != "audited_scaling_search_and_heldout_confirmation"
            or report.get("status") != "complete"
            or report.get("selected_cases") is not None):
        raise ValueError("plotting requires the complete, unfiltered analyzer report")
    manifest = report["manifest"]["recorded"]
    protocol = manifest["protocol"]
    if (protocol["batch"] != 4 or protocol["timing_protocol"] != 3
            or protocol["warmup_ms"] != 200):
        raise ValueError("unexpected measurement protocol; chart labels need review")
    seeds = manifest["seeds"]["confirmation"]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("expected two distinct confirmation seeds")
    cases = report["cases"]
    if (len(cases) != len(manifest["cases"])
            or len(cases) != report["case_count"]
            or len(cases) != report["completed_case_count"]
            or {case["case"] for case in cases} != {case["name"] for case in manifest["cases"]}):
        raise ValueError("report coverage differs from the recorded full campaign")
    indexed = {}
    for case in cases:
        if case["status"] != "complete":
            raise ValueError("unfinished case in a complete report")
        workload = case["workload"]
        expected = {"input": "u32", "counter": "u64", "distribution": "uniform",
                    "order": "shuffled", "cache": "warm", "launch": "graph"}
        if any(workload[key] != value for key, value in expected.items()):
            raise ValueError("workload differs from the chart labels")
        key = (workload["n"], workload["bins"])
        if key in indexed:
            raise ValueError("duplicate workload in analyzer report")
        confirmation = case["confirmations"]
        if len(confirmation) != 2 or {row["seed"] for row in confirmation} != set(seeds):
            raise ValueError("case does not contain both confirmation seeds")
        for row in confirmation:
            for field in ("chosen_median_us", "reference_median_us", "reference_over_chosen"):
                if not math.isfinite(row[field]) or row[field] <= 0:
                    raise ValueError("nonpositive or nonfinite confirmation measurement")
            if not math.isclose(row["reference_over_chosen"],
                                row["reference_median_us"] / row["chosen_median_us"],
                                rel_tol=1e-9):
                raise ValueError("confirmation speedup does not match its paired latencies")
        indexed[key] = case
    if any((SLICE_N, bins) not in indexed for bins in (*BASE_BINS, *BOUNDARY_BINS)):
        raise ValueError("report lacks a required base or catalog-boundary workload")
    gpus = report["environment"]["recorded"]["gpus"]
    if len(gpus) != 1:
        raise ValueError("expected one recorded GPU for these charts")
    gpu = gpus[0]
    label = (f"{gpu['name']} · driver {gpu['driver_version']}\n"
             "Uniform shuffled u32 input → u64 counts · warm graph · batch 4 · 200 ms warmup")
    return indexed, label


def interval(values):
    center = statistics.median(values)
    return center, center - min(values), max(values) - center


def draw_ranges(axis, xs, values, *, color, label=None, marker="o", linestyle="-"):
    ranges = [interval(value) for value in values]
    return axis.errorbar(xs, [value[0] for value in ranges],
                         yerr=[[value[1] for value in ranges], [value[2] for value in ranges]],
                         color=color, label=label, marker=marker, linestyle=linestyle,
                         linewidth=1.65, markersize=4.6, capsize=3.2, capthick=1.1,
                         elinewidth=1.2, zorder=3)


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#C8CDD2")
    axis.grid(axis="y", color="#E6E9EC", linewidth=.8, zorder=0)
    axis.tick_params(axis="both", length=3, color="#A6AFB7")
    axis.set_axisbelow(True)


def save_figure(figure, directory, name):
    for extension in ("svg", "png"):
        path = directory / f"{name}.{extension}"
        figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        print(path)


def throughput_figure(plt, ticker, cases, subtitle, directory):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), sharey=True)
    figure.subplots_adjust(left=.08, right=.98, bottom=.16, top=.79,
                           hspace=.38, wspace=.17)
    figure.suptitle("Histogram throughput as input grows", x=.08, y=.985,
                   ha="left", fontsize=19, fontweight="semibold")
    figure.text(.08, .936, subtitle, va="top", fontsize=10.5, linespacing=1.6)
    maximum = 0.0
    handles = None
    for bins, axis in zip(BASE_BINS, axes.flat):
        group = sorted((n, case) for (n, b), case in cases.items() if b == bins)
        xs = [n for n, _ in group]
        for field, color, label, marker, linestyle in (
                ("chosen_median_us", CUSTOM, "Selected custom kernel", "o", "-"),
                ("reference_median_us", REFERENCE, "NVIDIA histogram", "s", "--")):
            # GB/s is decimal input bytes per second, including output initialization
            # and merge time in the measured complete histogram operation.
            values = [[n * 4 / (row[field] * 1000) for row in case["confirmations"]]
                      for n, case in group]
            maximum = max(maximum, *(max(value) for value in values))
            draw_ranges(axis, xs, values, color=color, label=label,
                        marker=marker, linestyle=linestyle)
        style_axis(axis)
        axis.set_title(f"{bins:,} bins", loc="left", fontsize=12, fontweight="semibold", pad=10)
        axis.set_xscale("log", base=2)
        axis.set_xticks(xs, [f"{n // (1 << 20):,}" for n in xs])
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.set_xlabel("Input count / 2²⁰", labelpad=7)
        axis.margins(x=.07)
        handles = axis.get_legend_handles_labels()
    for axis in axes[:, 0]:
        axis.set_ylabel("Input GB/s", labelpad=9)
    axes[0, 0].set_ylim(0, maximum * 1.12)
    figure.legend(*handles, loc="upper left", bbox_to_anchor=(.075, .853),
                  ncol=2, frameon=False, handlelength=2.4, columnspacing=2.4)
    figure.text(.08, .071,
                "Points: median of two fresh confirmation-seed medians. Whiskers: seed range, not a confidence interval.\n"
                "Custom policy selected separately per shape before confirmation; clocks unlocked. 1 GB = 10⁹ bytes.",
                fontsize=9, color="#59656F", linespacing=1.6)
    save_figure(figure, directory, "throughput")
    plt.close(figure)


def speedup_figure(plt, ticker, cases, subtitle, directory):
    group = sorted((bins, case) for (n, bins), case in cases.items() if n == SLICE_N)
    bins = [bins for bins, _ in group]
    ratios = [[row["reference_over_chosen"] for row in case["confirmations"]]
              for _, case in group]
    maximum = max(1.0, *(max(value) for value in ratios))
    figure, (axis, closeup) = plt.subplots(1, 2, figsize=(12, 8), sharey=True,
                                         gridspec_kw={"width_ratios": [1.9, 1]})
    figure.subplots_adjust(left=.08, right=.98, bottom=.34, top=.73, wspace=.20)
    figure.suptitle("Custom histogram speedup over NVIDIA histogram", x=.08, y=.985,
                   ha="left", fontsize=19, fontweight="semibold")
    figure.text(.08, .921, subtitle, va="top", fontsize=10.5, linespacing=1.6)
    draw_ranges(axis, bins, ratios, color=CUSTOM)
    axis.set_xscale("log", base=2)
    wide_ticks = [value for value in bins if value not in BOUNDARY_BINS]
    axis.set_xticks(wide_ticks, [f"{value:,}" for value in wide_ticks], rotation=30, ha="right")
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.set_title(f"{SLICE_N:,} input values", loc="left", fontsize=12, pad=12)
    axis.set_xlabel("Bin count (log scale)", labelpad=7)
    axis.set_ylabel("NVIDIA histogram time / custom time", labelpad=9)
    axis.axvline(24576, color="#98A5AF", linestyle=":", linewidth=1.1, zorder=1)
    boundary_values = [[row["reference_over_chosen"] for row in cases[(SLICE_N, value)]["confirmations"]]
                       for value in BOUNDARY_BINS]
    draw_ranges(closeup, list(range(3)), boundary_values, color=CUSTOM)
    closeup.set_xticks(range(3), [f"{value:,}" for value in BOUNDARY_BINS])
    closeup.set_xlim(-.35, 2.35)
    closeup.set_xlabel("Bin count (one-bin steps)", labelpad=7)
    closeup.set_title("Current catalog boundary", loc="left", fontsize=12, pad=12)
    closeup.axvline(1, color="#98A5AF", linestyle=":", linewidth=1.1, zorder=1)
    for item in (axis, closeup):
        style_axis(item)
        item.axhline(1, color=REFERENCE, linestyle="--", linewidth=1.1, zorder=2)
        item.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}×"))
        item.set_ylim(0, maximum * 1.15)
    figure.text(.08, .08,
                "Above 1×: custom is faster. Points and whiskers summarize paired ratios across two fresh seeds; clocks unlocked.\n"
                "24,576 bins × 4-byte local counts = 96 KiB: current kernel catalog capacity, not a measured hardware maximum.\n"
                "Policies were fixed before confirmation. NVIDIA histogram is used only as a benchmark reference.",
                fontsize=9, color="#59656F", linespacing=1.6)
    save_figure(figure, directory, "speedup-by-bins")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path,
                        default=BASE / "complete-catalog/scaling-analysis.json")
    parser.add_argument("--output-dir", type=Path, help="defaults to REPORT_DIRECTORY/figures")
    options = parser.parse_args(argv)
    try:
        cases, subtitle = load_report(options.report)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        directory = options.output_dir or options.report.parent / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "text.color": TEXT, "axes.labelcolor": TEXT,
                             "xtick.color": TEXT, "ytick.color": TEXT,
                             "svg.fonttype": "none", "axes.unicode_minus": False,
                             "savefig.transparent": False}):
            throughput_figure(plt, ticker, cases, subtitle, directory)
            speedup_figure(plt, ticker, cases, subtitle, directory)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
