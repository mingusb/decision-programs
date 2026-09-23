#!/usr/bin/env python3
"""Extract existing Nsight diagnostic reports without launching GPU work."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    profiles = []
    for path in sorted(BASE.glob("profile-*.details.txt")):
        name = path.name.removesuffix(".details.txt")
        command_path = BASE / (name + ".command.json")
        command = json.loads(command_path.read_text())
        text = path.read_text()
        sections = {}
        section = None
        for line in text.splitlines():
            if line.strip().startswith("Section: "):
                section = line.strip().removeprefix("Section: ")
                sections[section] = {}
                continue
            cells = re.split(r"\s{2,}", line.strip())
            if section and 2 <= len(cells) <= 3 and re.fullmatch(r"-?[\d.]+", cells[-1]):
                sections[section][cells[0]] = {"value": float(cells[-1]), "unit": cells[1] if len(cells) == 3 else None}
        def metric(section, key):
            return sections[section][key]["value"]
        shared = re.search(r"total of ([\d,]+) excessive wavefronts \(([\d.]+)% of\s+(?:the\s+)?total ([\d,]+) wavefronts\)", text)
        long_scoreboard = re.search(r"spends ([\d.]+) cycles being stalled waiting for a scoreboard dependency", text)
        profile = {"name": name, "details": path.name, "details_sha256": sha256(path),
                   "command_json": command_path.name, "command_json_sha256": sha256(command_path),
                   "command": command, "sections": sections,
                   "duration_us": metric("GPU Speed Of Light Throughput", "Duration"),
                   "dram_peak_percent": metric("GPU Speed Of Light Throughput", "DRAM Throughput"),
                   "registers_per_thread": metric("Launch Statistics", "Registers Per Thread"),
                   "dynamic_shared_Kbyte": metric("Launch Statistics", "Dynamic Shared Memory Per Block"),
                   "achieved_occupancy_percent": metric("Occupancy", "Achieved Occupancy"),
                   "no_eligible_percent": metric("Scheduler Statistics", "No Eligible"),
                   "local_spilling_requests": metric("Memory Workload Analysis", "Local Memory Spilling Requests"),
                   "shared_spilling_requests": metric("Memory Workload Analysis", "Shared Memory Spilling Requests"),
                   "unlocked_frequency_warning": "without fixed GPU frequencies" in text}
        if shared:
            profile["shared_excessive_wavefronts"] = int(shared[1].replace(",", ""))
            profile["shared_excessive_wavefronts_percent"] = float(shared[2])
            profile["shared_total_wavefronts"] = int(shared[3].replace(",", ""))
        if long_scoreboard:
            profile["long_scoreboard_cycles_per_issued_instruction"] = float(long_scoreboard[1])
        for option, value in (("--replay-mode", "kernel"), ("--cache-control", "all"), ("--clock-control", "none"), ("--launch-count", "1")):
            assert command[command.index(option) + 1] == value
        profiles.append(profile)
    result = {"schema": 1, "profile_count": len(profiles), "profiles": profiles,
              "scope": "Fresh session diagnostics; individual cache-flushed counting kernels, unlocked clocks. "
                       "These are not complete-operation timings or a matched historical before/after comparison.",
              "metadata_limit": "Profile command files record commands only; no per-command driver/hash snapshots. "
                                "Artifact hashes are computed by this CPU-only analyzer."}
    lines = ["# Fresh Nsight Compute diagnostics", "",
             "These captures diagnose individual counting kernels in the resumed session. They use kernel replay, "
             "cache-control all and unlocked clocks. They exclude output clearing and finalization costs outside the "
             "selected counting kernel, so they cannot replace unprofiled complete-operation comparisons. "
             "Historical captures came from another driver session and are not a controlled before/after baseline.", "",
             "| Capture | Counting kernel, µs | DRAM peak | Registers/thread | Dynamic shared, KiB | Achieved occupancy | No eligible warp |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for p in profiles:
        # Nsight's Kbyte values are rounded decimal kilobytes; use exact launch
        # shape/bins for interpretation, while preserving its original units here.
        lines.append(f"| {p['name']} | {p['duration_us']:.2f} | {p['dram_peak_percent']:.2f}% | "
                     f"{p['registers_per_thread']:g} | ≈{p['dynamic_shared_Kbyte'] * 1000 / 1024:.2f} | "
                     f"{p['achieved_occupancy_percent']:.2f}% | {p['no_eligible_percent']:.2f}% |")
    lines += ["", "The byte capture (`shared:11:48`, N1M/u8/B256) uses 1,024-thread blocks and approximately "
              "1 KiB of dynamic shared memory. Its 48.45% DRAM throughput and 78.76% cycles without an eligible warp "
              "leave latency and scheduling as measurable limitations in this cache-flushed capture; this is not proof "
              "that a particular change would improve the warm-graph operation.", "",
              "The opt-in capture (`shared:14:48:u32`, N16M/u32/B16384 with u64 output) uses 64 KiB of dynamic "
              "shared memory. Shared-memory capacity limits it to one 256-thread block per SM, yet it reaches "
              "90.60% of peak DRAM throughput. Increasing occupancy alone is therefore not an established improvement.", "",
              "Both reports record zero local- and shared-memory spilling requests. Nsight reports excessive shared "
              "wavefronts (68% in the byte capture and 70% in the opt-in capture), but those aggregate counts do not "
              "establish that all of the random histogram update cost is avoidable. Suggested Nsight speedups are "
              "diagnostic estimates, not measured algorithm gains.", "",
              "[Extracted metrics, artifact hashes and exact commands](profile-summary.json). The original "
              "`.details.txt` and `.ncu-repz` files preserve the full evidence. Command records contain no per-profile "
              "driver/hash snapshots; this summary does not manufacture that provenance.", ""]
    (BASE / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (BASE / "profile-summary.md").write_text("\n".join(lines))
    print(f"Wrote diagnostics for {len(profiles)} captures")


if __name__ == "__main__":
    main()
