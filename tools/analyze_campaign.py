#!/usr/bin/env python3
"""Summarize observed histogram portfolios without running any GPU workloads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from autotune import BASELINES, CONFIG, ENVIRONMENT, WORKLOAD, TuningError, config_key, extract, parse_csv
from campaign import variants


class AnalysisError(Exception):
    pass


def expand_inputs(arguments: list[str]) -> list[Path]:
    paths = []
    for argument in arguments:
        path = Path(argument).expanduser().resolve()
        if path.is_dir():
            aggregate = path / "measurements.csv"
            found = [aggregate] if aggregate.is_file() else sorted(path.glob("cell-*.csv"))
            if not found:
                raise AnalysisError(f"no measurements.csv or cell-*.csv files in {path}")
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise AnalysisError(f"input does not exist: {path}")
    return list(dict.fromkeys(paths))


def cell_key(row: dict) -> tuple:
    # A different seed, sampling protocol, or eviction extent is a different
    # observation. Never select across repeats and call the result one cell.
    return tuple(row[key] for key in WORKLOAD + ("seed", "samples", "batch", "eviction_bytes"))


def describe(row: dict) -> dict:
    return {**extract(row, CONFIG), "median_us": row["median_us"]}


def ratio_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "minimum": min(values), "maximum": max(values),
        "median": statistics.median(values),
        "geometric_mean": math.exp(statistics.fmean(math.log(value) for value in values)),
        "faster": sum(value > 1 for value in values),
        "at_least_1_05x": sum(value >= 1.05 for value in values),
        "slower": sum(value < 1 for value in values),
        "at_least_1_05x_slower": sum(value <= 1 / 1.05 for value in values),
    }


def load(paths: list[Path]) -> tuple[list[dict], list[dict], dict, list[dict]]:
    seen = {}
    sources = []
    manifests = {}
    hardware = None
    for path in paths:
        content = path.read_text(encoding="utf-8")
        _, rows = parse_csv(content)
        manifest_path = path.parent / "manifest.json"
        manifest = None
        if manifest_path.is_file():
            if manifest_path not in manifests:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema") != 1:
                    raise AnalysisError(f"unsupported campaign manifest schema: {manifest_path}")
                manifests[manifest_path] = manifest
            manifest = manifests[manifest_path]
        sources.append({"path": str(path), "sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "rows": len(rows), "manifest": str(manifest_path) if manifest else None})
        declared_cells = ({tuple(item[key] for key in WORKLOAD) for item in manifest["cells"]}
                          if manifest else None)
        for row in rows:
            environment = {key: value for key, value in extract(row, ENVIRONMENT).items()
                           if key != "eviction_bytes"}
            if hardware is None:
                hardware = environment
            elif hardware != environment:
                raise AnalysisError("GPU/runtime/CUB metadata differs; analyze environments separately")
            if manifest:
                if tuple(row[key] for key in WORKLOAD) not in declared_cells:
                    raise AnalysisError(f"row is outside declared campaign cells: {path}")
                if any(row[key] != manifest[key] for key in ("seed", "samples", "batch")):
                    raise AnalysisError(f"measurement protocol differs from manifest: {path}")
            identity = cell_key(row), config_key(row)
            if identity in seen:
                if seen[identity]["csv_row"] != row["csv_row"]:
                    raise AnalysisError("duplicate cell/config has different measurements; analyze repeated campaigns separately")
                if manifest:
                    seen[identity]["manifests"].add(manifest_path)
                continue  # Aggregate plus original cell file: exactly identical evidence.
            row["source"] = str(path)
            row["manifests"] = {manifest_path} if manifest else set()
            seen[identity] = row
    rows = list(seen.values())
    groups = defaultdict(list)
    for row in rows:
        groups[cell_key(row)].append(row)
    for group in groups.values():
        associated = set().union(*(row["manifests"] for row in group))
        for manifest_path in associated:
            manifest = manifests[manifest_path]
            expected = set()
            for variant in variants(extract(group[0], WORKLOAD), manifest["sm_count"]).split(","):
                algorithm, tuning, blocks, local, clear = variant.split(":")
                expected.add((algorithm, int(tuning), int(blocks), local, clear))
            if {config_key(row) for row in group} != expected:
                raise AnalysisError(f"cell candidate set differs from predeclared portfolio: {manifest_path}")
    manifest_summaries = []
    for path, manifest in manifests.items():
        observed = {cell_key(row) for row in rows if path in row["manifests"]}
        manifest_summaries.append({"path": str(path), "binary_sha256": manifest["sha256"],
                                   "planned_cells": len(manifest["cells"]),
                                   "observed_cells": len(observed),
                                   "complete": len(observed) == len(manifest["cells"])})
    if len({manifest["binary_sha256"] for manifest in manifest_summaries}) > 1:
        raise AnalysisError("campaign binary hashes differ; analyze builds separately")
    return rows, sources, hardware or {}, manifest_summaries


def summarize_group(cells: list[dict]) -> dict:
    return {
        "cells": len(cells),
        "candidate_measurements": sum(cell["candidate_count"] for cell in cells),
        "sample_reference_cells": sum(cell["sample"] is not None for cell in cells),
        "best_custom_algorithms": dict(Counter(cell["custom_best"]["algorithm"] for cell in cells
                                                if cell["custom_best"] is not None)),
        "best_portfolio_algorithms": dict(Counter(cell["portfolio_best"]["algorithm"] for cell in cells)),
        **{key: ratio_stats([cell[key] for cell in cells if cell[key] is not None])
           for key in ("custom_over_cub", "custom_over_reference", "portfolio_over_cub", "portfolio_over_reference")},
    }


def analyze(paths: list[Path]) -> dict:
    rows, sources, environment, manifests = load(paths)
    grouped = defaultdict(list)
    for row in rows:
        grouped[cell_key(row)].append(row)
    cells = []
    pairs = []
    unpaired_native = unpaired_u32 = 0
    for index, key in enumerate(sorted(grouped)):
        group = grouped[key]
        first = group[0]
        cub = [row for row in group if row["algorithm"] == "cub"]
        sample = [row for row in group if row["algorithm"] == "nvidia_sample256"]
        if len(cub) != 1 or len(sample) > 1:
            raise AnalysisError("each cell needs exactly one CUB measurement and at most one sample reference")
        references = cub + sample
        reference = min(references, key=lambda row: row["median_us"])
        custom = [row for row in group if row["algorithm"] not in BASELINES]
        best_custom = min(custom, key=lambda row: row["median_us"]) if custom else None
        best = min(group, key=lambda row: row["median_us"])
        cell = {
            "cell_id": index, "workload": extract(first, WORKLOAD),
            "seed": first["seed"], "samples": first["samples"], "batch": first["batch"],
            "eviction_bytes": first["eviction_bytes"], "candidate_count": len(group),
            "custom_candidate_count": len(custom), "source": first["source"],
            "predeclared_manifest_verified": all(row["manifests"] for row in group),
            "cub": describe(cub[0]), "sample": describe(sample[0]) if sample else None,
            "strongest_reference": describe(reference),
            "custom_best": describe(best_custom) if best_custom else None, "portfolio_best": describe(best),
            "custom_over_cub": cub[0]["median_us"] / best_custom["median_us"] if best_custom else None,
            "custom_over_reference": reference["median_us"] / best_custom["median_us"] if best_custom else None,
            "portfolio_over_cub": cub[0]["median_us"] / best["median_us"],
            "portfolio_over_reference": reference["median_us"] / best["median_us"],
        }
        cells.append(cell)
        if first["counter"] != "u64":
            continue
        by_policy = defaultdict(dict)
        for row in custom:
            if row["algorithm"] not in ("shared", "shared_rle", "shared_warp", "shared_partial"):
                continue
            # Scratch size deliberately differs between local widths. Every
            # other compile-time/runtime launch parameter must match exactly.
            policy = tuple(row[key] for key in ("algorithm", "tuning", "threads", "items", "replicas", "blocks", "clear_policy"))
            by_policy[policy][row["local_counter"]] = row
        for matching in by_policy.values():
            if set(matching) != {"native", "u32"}:
                unpaired_native += "native" in matching
                unpaired_u32 += "u32" in matching
                continue
            native, narrow = matching["native"], matching["u32"]
            pairs.append({"cell_id": index, "workload": cell["workload"], "seed": cell["seed"],
                          "native": describe(native), "u32": describe(narrow),
                          "native_over_u32": native["median_us"] / narrow["median_us"]})
    strata = defaultdict(list)
    for cell in cells:
        strata[(cell["workload"]["cache"], cell["workload"]["launch"])].append(cell)
    pair_strata = defaultdict(list)
    for pair in pairs:
        pair_strata[(pair["workload"]["cache"], pair["workload"]["launch"], pair["native"]["algorithm"])].append(pair["native_over_u32"])
    return {
        "schema": 1,
        "interpretation": "Best observed per-cell portfolios are ex-post oracles over measured candidates, not deployable selectors. "
                          "Ratios use within-cell median times. Aggregate medians/geometric means weight cells equally; "
                          "they are descriptive and do not establish uncertainty or generalization.",
        "ratio_direction": "Reference time / candidate time; greater than one favors the candidate. Local-width ratios are native time / local-u32 time.",
        "portfolio_definition": "All measured candidates, including CUB and the NVIDIA sample where present. Best custom excludes both references. "
                                "Strongest reference is the faster measured CUB/sample median within that cell.",
        "cell_id_definition": "Analysis IDs follow sorted workload/protocol order; they are not campaign cell-NNN filename indices.",
        "environment": environment, "sources": sources, "manifests": manifests,
        "overall": summarize_group(cells),
        "by_cache_launch": [{"cache": cache, "launch": launch, **summarize_group(group)}
                            for (cache, launch), group in sorted(strata.items())],
        "local_counter_comparison": {
            "overall": ratio_stats([pair["native_over_u32"] for pair in pairs]),
            "unmatched_native_rows": unpaired_native, "unmatched_u32_rows": unpaired_u32,
            "by_cache_launch_algorithm": [{"cache": cache, "launch": launch, "algorithm": algorithm,
                                            **ratio_stats(values)}
                                           for (cache, launch, algorithm), values in sorted(pair_strata.items())],
            "pairs": pairs,
        },
        "cells": cells,
    }


def ratio(value: float | None) -> str:
    return f"{value:.3f}x" if value is not None else "—"


def candidate_label(candidate: dict | None) -> str:
    if candidate is None:
        return "—"
    return f"{candidate['algorithm']} t{candidate['tuning']} g{candidate['blocks']} {candidate['local_counter']}"


def markdown(report: dict, table_limit: int) -> str:
    environment = report["environment"]
    out = ["# Observed histogram campaign", "", report["interpretation"], "", report["portfolio_definition"], "",
           f"Analyzed **{report['overall']['cells']} cells** and **{report['overall']['candidate_measurements']} candidate measurements**.", "",
           f"GPU: {environment['gpu']} (SM{environment['sm']}); runtime API {environment['runtime']}; "
           f"driver API {environment['driver_api']}; CUB {environment['cub_version']}. Driver API is a compatibility version, not the installed driver build.", ""]
    if report["manifests"]:
        for manifest in report["manifests"]:
            state = "complete" if manifest["complete"] else "partial"
            out.append(f"Campaign coverage: {manifest['observed_cells']}/{manifest['planned_cells']} cells ({state}); binary `{manifest['binary_sha256']}`.")
    if any(not cell["predeclared_manifest_verified"] for cell in report["cells"]):
        out.append("Some input has no verified campaign manifest; those results describe only the supplied observed portfolio.")
    out += ["", "## Comparisons by cache and launch", "", "Ratios above one favor the custom/portfolio result. The portfolio includes references, so its speedup cannot be below one by construction.", "",
            "| Cache / launch | Cells | Custom ≥1.05x CUB | Custom ≥1.05x strongest ref | Custom/ref median | Custom/ref geomean | Portfolio/ref geomean |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    groups = [{"cache": "all", "launch": "all", **report["overall"]}] + report["by_cache_launch"]
    for group in groups:
        custom = group["custom_over_reference"]
        out.append(f"| {group['cache']} / {group['launch']} | {group['cells']} | "
                   f"{group['custom_over_cub'].get('at_least_1_05x', 0)} | {custom.get('at_least_1_05x', 0)} | "
                   f"{ratio(custom.get('median'))} | {ratio(custom.get('geometric_mean'))} | "
                   f"{ratio(group['portfolio_over_reference'].get('geometric_mean'))} |")
    custom_cells = [cell for cell in report["cells"] if cell["custom_best"]]
    for heading, reverse in (("Largest observed custom/reference ratios", True),
                             ("Smallest observed custom/reference ratios", False)):
        selection = sorted(custom_cells, key=lambda cell: cell["custom_over_reference"], reverse=reverse)
        if table_limit:
            selection = selection[:table_limit]
        out += ["", f"## {heading}", "", "These rows are selected by the observed ratio; they are not an independent validation set. "
                "Analysis IDs are not campaign filenames.", "",
                "| Analysis ID | N | Bins | Input/count | Distribution/order | Cache/launch | Best custom | Custom µs | CUB µs | Strongest reference | Ref µs | Custom/CUB | Custom/ref |",
                "|---:|---:|---:|---|---|---|---|---:|---:|---|---:|---:|---:|"]
        for cell in selection:
            w, best, ref = cell["workload"], cell["custom_best"], cell["strongest_reference"]
            out.append(f"| {cell['cell_id']} | {w['n']} | {w['bins']} | {w['input']}/{w['counter']} | "
                       f"{w['distribution']}/{w['order']} | {w['cache']}/{w['launch']} | {candidate_label(best)} | "
                       f"{best['median_us']:.3f} | {cell['cub']['median_us']:.3f} | {ref['algorithm']} | {ref['median_us']:.3f} | "
                       f"{ratio(cell['custom_over_cub'])} | {ratio(cell['custom_over_reference'])} |")
    local = report["local_counter_comparison"]
    out += ["", "## Matched local counter widths for u64 outputs", "",
            "Only exact algorithm/tuning/threads/items/replicas/grid matches are paired; scratch width may differ. A ratio above one favors local u32.", "",
            f"Matched pairs: {local['overall']['count']}. Unmatched native rows: {local['unmatched_native_rows']}; unmatched u32 rows: {local['unmatched_u32_rows']}.", "",
            "| Cache/launch | Algorithm | Pairs | Median native/u32 | Geomean native/u32 | Min | Max | u32 ≥1.05x faster |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in local["by_cache_launch_algorithm"]:
        out.append(f"| {group['cache']}/{group['launch']} | {group['algorithm']} | {group['count']} | "
                   f"{ratio(group['median'])} | {ratio(group['geometric_mean'])} | {ratio(group['minimum'])} | "
                   f"{ratio(group['maximum'])} | {group['at_least_1_05x']} |")
    out += ["", "JSON output retains every cell, matched pair, source file hash, and measurement protocol. "
            "Reference scope, candidate coverage, cache eviction, launch mode, and host submission effects remain part of the measurement contract.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="measurements.csv, cell CSV files, or campaign directories")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--table-limit", type=int, default=12, help="rows per selected-cell table; zero lists all")
    parser.add_argument("--output", help="new output file; default stdout (existing files are never overwritten)")
    args = parser.parse_args()
    try:
        if args.table_limit < 0:
            raise AnalysisError("table limit must be nonnegative")
        report = analyze(expand_inputs(args.inputs))
        content = json.dumps(report, indent=2, allow_nan=False) + "\n" if args.format == "json" else markdown(report, args.table_limit)
        if args.output:
            with Path(args.output).expanduser().open("x", encoding="utf-8") as output:
                output.write(content)
        else:
            print(content, end="")
        return 0
    except (AnalysisError, TuningError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"analyze_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
