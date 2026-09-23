#!/usr/bin/env python3
"""Render every process-level GPU-event comparison, including same-backend controls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
source = BASE / "analysis.json"
report = json.loads(source.read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
plotted = []
for axis, case, title in zip(axes, ("single", "stream4096"),
                             ("Single-valued input · graph", "4,096-bin uniform input · stream")):
    for y, group, label in ((2, "real", "New / old"), (1, "old-old", "Old / old control"),
                            (0, "new-new", "New / new control")):
        summary = next(s for s in report["summaries"] if s["case"] == case and s["comparison"] == group)
        selected = [p for p in report["processes"] if p["case"] == case and
                    (p["comparison"] in ("old-new", "new-old") if group == "real" else p["comparison"] == group)]
        selected.sort(key=lambda p: (p["data_seed"], p["order_seed"]))
        values = [p["metrics"]["event_us"]["quartet_ratios"]["geometric_mean"] for p in selected]
        changes = [100 * (value - 1) for value in values]
        mean = summary["metrics"]["event_us"]["process_geometric_mean_ratios"]["geometric_mean"]
        color = "#2865a4" if group == "real" else "#788490"
        axis.plot([min(changes), max(changes)], [y, y], color=color, alpha=.45, linewidth=2)
        axis.scatter(changes, [y + offset for offset in (-.10, -.035, .035, .10)],
                     s=48, color=color, zorder=3)
        axis.scatter([100 * (mean - 1)], [y], marker="D", color="#182630", s=42, zorder=4)
        axis.text(6.1, y, f"{100 * (mean - 1):+.2f}%", va="center", fontsize=10)
        plotted.append({"case": case, "comparison": group, "process_ratios": values,
                        "geometric_mean": mean})
    axis.set_title(title, loc="left", fontsize=13, pad=17)
    axis.set_yticks([2, 1, 0], ["New / old", "Old / old control", "New / new control"])
    axis.tick_params(axis="y", length=0, pad=10)
    axis.axvline(0, color="#34495e", linewidth=1, linestyle="--")
    axis.set_xlim(-8, 8)
    axis.set_ylim(-.45, 2.45)
    axis.set_xticks([-8, -4, 0, 4, 8], ["−8%", "−4%", "0", "+4%", "+8%"])
    axis.grid(axis="x", color="#e5eaf0", linewidth=.7)
    axis.set_axisbelow(True)
    axis.set_xlabel("Change in event time · positive means slower")
fig.suptitle("Same-process old/new comparison and controls", x=.04, ha="left", fontsize=21, fontweight="bold")
fig.text(.04, .90, "RTX A5000 Laptop · 1,048,576 u32 inputs / u32 counts · 32 operations per batch · four processes per row",
         color="#52616f", fontsize=10)
fig.subplots_adjust(top=.76, bottom=.24, left=.145, right=.96, wspace=.59)
fig.text(.04, .12, "Dots: each process's geometric mean across 32 matched quartets. Diamonds and labels: aggregate geometric mean.", fontsize=10)
fig.text(.04, .075, "Lines span observed processes, not confidence intervals. Controls compare two slots running the same backend.", fontsize=10)
fig.text(.04, .03, "Archived code rebuilt in one process; clocks unlocked. These measurements do not establish zero performance loss.", fontsize=10)
for extension in ("png", "svg"):
    fig.savefig(BASE / f"paired-comparison.{extension}", dpi=160, facecolor="white")
(BASE / "plotted-data.json").write_text(json.dumps({
    "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "data": plotted,
}, indent=2) + "\n")
