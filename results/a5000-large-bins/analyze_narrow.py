#!/usr/bin/env python3
"""Independently audit narrow-global measurements using the frozen source archive.

CPU-only: reads local artifacts, loads the archived CSV parser, and never calls
the benchmark, recorder, CUDA, nvidia-smi, or the current production sources.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
import types

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA = BASE / "narrow-evaluation"
ARCHIVE_SHA = "ebb2fb891735cf219dc17656d7ac7f4fc1e767b8da6340533f6b2b99ba83ae68"
BINARY_SHA = "bcf042dee54cd18408e36f5852e756e88348dd5aa3d3083e145a834a41e9f648"
PROTOCOL = {"timing_protocol": 3, "batch": 4, "warmup_ms": 200, "default_samples": 3,
            "search_samples": 5, "validation_samples": 11, "confirmation_samples": 21}
SEEDS = {"search": 49979687, "validation": [67867967, 86028121],
         "confirmation": [104395303, 122949829]}
GRIDS = [24, 48, 96, 192, 384, 768, 1536]
REFERENCE = "cub:2:192:native:kernel"
POLICIES = [(128, 4, 1), (256, 4, 1), (256, 8, 1), (256, 16, 1), (128, 8, 4)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    return {"path": str(path.resolve()), "sha256": digest(path.read_bytes())}


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def variant(row):
    return ":".join(str(row[key]) for key in
                    ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


def expected_cases():
    variants = [f"{a}:{p}:{g}:{local}:kernel" for a in ("global", "warp")
                for p in range(5) for g in GRIDS for local in ("native", "u32")] + [REFERENCE]
    return [{"name": f"n{1 << 24}-b{bins}", "workload": {
                "n": 1 << 24, "bins": bins, "input": "u32", "counter": "u64",
                "distribution": "uniform", "order": "shuffled", "cache": "warm",
                "launch": "graph", "warmup_ms": 200}, "search_variants": variants}
            for bins in (24577, 65536, 262144, 1048576)]


def audit_provenance(manifest, environment):
    archive = BASE / "narrow-source.tar.gz"
    source_manifest_path = BASE / "narrow-source-manifest.json"
    source_manifest = read_json(source_manifest_path)
    require(digest(archive.read_bytes()) == ARCHIVE_SHA == source_manifest["archive_sha256"],
            "frozen source archive hash differs")
    files = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            require(member.isfile() and member.name not in files, "unexpected/duplicate archive member")
            files[member.name] = bundle.extractfile(member).read()
    require(set(files) == {item["path"] for item in source_manifest["files"]}
            and len(files) == len(source_manifest["files"]), "archive coverage differs from source manifest")
    for item in source_manifest["files"]:
        require(digest(files[item["path"]]) == item["sha256"], "archived source hash mismatch: " + item["path"])
    for item in [*manifest["sources"], manifest["parser"]]:
        relative = str(Path(item["path"]).relative_to(ROOT))
        require(relative in files and digest(files[relative]) == item["sha256"],
                "campaign-declared source differs from archive: " + relative)
    for name in ("runner", "recorder"):
        item = manifest[name]
        require(file_record(Path(item["path"])) == item, name + " differs from its frozen hash")
    binary = manifest["binary"]
    require(binary["sha256"] == BINARY_SHA and file_record(Path(binary["path"])) == binary,
            "preserved narrow executable differs from campaign binary")
    require(environment["schema"] == 1 and environment["binary"] == binary["path"]
            and environment["binary_sha256"] == BINARY_SHA and environment["session_started_utc"]
            and environment["gpus"] and isinstance(environment["uname"], dict), "invalid session environment")
    require(manifest["schema"] == 1 and manifest["kind"] == "narrow_global_four_shape_evaluation"
            and manifest["protocol"] == PROTOCOL and manifest["seeds"] == SEEDS
            and manifest["grids"] == GRIDS and manifest["cases"] == expected_cases(),
            "experiment matrix, protocol or seeds differ")
    seeds = [SEEDS["search"], *SEEDS["validation"], *SEEDS["confirmation"]]
    require(len(seeds) == len(set(seeds)), "confirmation/search/validation seeds overlap")
    parser = types.ModuleType("archived_narrow_autotune")
    parser.__file__ = str(archive) + "/tools/autotune.py"
    sys.modules[parser.__name__] = parser
    exec(compile(files["tools/autotune.py"], parser.__file__, "exec"), parser.__dict__)
    return parser, {"archive": file_record(archive), "source_manifest": file_record(source_manifest_path),
                    "archive_file_count": len(files), "campaign_sources_verified": len(manifest["sources"]),
                    "parser_verified_from_archive": manifest["parser"],
                    "method": "Read archive members in memory; current production source hashes are not substituted."}


def audit_case(case, state):
    manifest, environment, parser = state["manifest"], state["environment"], state["parser"]
    directory = DATA / "cases" / case["name"]
    expected_paths, stages, configs = set(), {}, {}

    def saved(path):
        expected_paths.add(path)
        return read_json(path)

    def invocation(name, seed, samples, variants):
        stem = directory / name
        paths = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
        expected_paths.update(paths)
        command = [manifest["binary"]["path"]] + parser.workload_args(case["workload"], seed)
        command += ["--samples", str(samples), "--batch", "4"]
        command += ["--algorithm", "auto"] if variants is None else ["--variants", ",".join(variants)]
        metadata = read_json(paths[2])
        require(metadata["command"] == command and metadata["exit_code"] == 0
                and metadata["binary"] == metadata["executable"] == manifest["binary"]["path"]
                and metadata["binary_sha256"] == metadata["executable_sha256"] == BINARY_SHA
                and metadata["executables_unchanged"] is True
                and metadata["gpus_before"] == metadata["gpus_after"] == environment["gpus"],
                f"{stem}: command or executable/GPU identity mismatch")
        require(math.isfinite(metadata["seconds"]) and metadata["seconds"] >= 0
                and metadata["telemetry_before"] and metadata["telemetry_after"], f"{stem}: invalid recording metadata")
        _, rows = parser.parse_csv(paths[0].read_text())
        log = paths[1].read_text()
        require(f"Validated {len(rows)} configurations" in log, f"{stem}: missing benchmark validation status")
        labels = [variant(row) for row in rows]
        require(len(labels) == len(set(labels)), f"{stem}: duplicate candidate")
        if variants is None:
            require(len(rows) == 1 and rows[0]["algorithm"] in ("global", "warp"), "invalid resolved default")
        else:
            require(len(variants) == len(set(variants)) and set(labels) == set(variants), f"{stem}: candidate coverage differs")
        for row in rows:
            parser.verify_row(row, case["workload"], state["common"], seed, samples, 4)
            state["common"] = parser.extract(row, parser.ENVIRONMENT)
            require(row["gpu"] in {gpu["name"] for gpu in environment["gpus"]}, f"{stem}: wrong GPU name")
            require(row["timing_protocol"] == 3 and row["clear_policy"] == "kernel", f"{stem}: wrong timing/clear protocol")
            raw = sorted(row["raw_samples"])
            derived = {"median_us": raw[math.ceil(len(raw) * .5) - 1], "p95_us": raw[math.ceil(len(raw) * .95) - 1],
                       "min_us": raw[0], "max_us": raw[-1], "input_gb_s": row["n"] * 4 / (row["median_us"] * 1000)}
            require(all(math.isclose(row[k], value, rel_tol=1e-6, abs_tol=1e-6) for k, value in derived.items()),
                    f"{stem}: summary differs from raw timing samples")
            config = parser.extract(row, parser.CONFIG)
            if row["algorithm"] == "cub":
                expected = dict(algorithm="cub", tuning=2, threads=0, items=0, replicas=0, blocks=192,
                                scratch_bytes=row["scratch_bytes"], local_counter="native", load_policy="reference",
                                shared_limit=0, clear_policy="kernel")
                require(config == expected and row["scratch_bytes"] > 0, f"{stem}: invalid reference metadata")
            else:
                require(row["algorithm"] in ("global", "warp") and row["tuning"] in range(5)
                        and row["blocks"] in GRIDS, f"{stem}: custom policy outside search domain")
                require((row["threads"], row["items"], row["replicas"]) == POLICIES[row["tuning"]]
                        and row["shared_limit"] == 49152 and row["load_policy"] == "scalar"
                        and row["scratch_bytes"] == (row["bins"] * 4 if row["local_counter"] == "u32" else 0),
                        f"{stem}: incorrect policy or scratch metadata")
            label = variant(row)
            require(label not in configs or configs[label] == config, f"{stem}: changing config metadata")
            configs[label] = config
        stages[name] = {"seed": seed, "recorded": metadata, "log": log,
                        "artifacts": [file_record(path) for path in paths], "measurements": rows}
        return rows

    default = invocation("default", SEEDS["search"], 3, None)[0]
    default_variant = variant(default)
    search_variants = list(dict.fromkeys(case["search_variants"] + [default_variant]))
    require(len(search_variants) == 141, "expected all 140 custom candidates plus one reference")
    search = invocation("search", SEEDS["search"], 5, search_variants)
    ranked = sorted((row for row in search if row["algorithm"] in ("global", "warp")),
                    key=lambda row: (row["median_us"], variant(row)))
    finalists = ranked[:4]
    represented = {(row["algorithm"], row["local_counter"]) for row in finalists}
    for row in ranked:
        family = row["algorithm"], row["local_counter"]
        if family not in represented:
            finalists.append(row)
            represented.add(family)
    require(represented == {(a, c) for a in ("global", "warp") for c in ("native", "u32")}, "missing finalist family")
    finalist_variants = list(dict.fromkeys([variant(row) for row in finalists] + [default_variant, REFERENCE]))
    finalists_path = directory / "finalists.json"
    require(saved(finalists_path) == {"schema": 1, "case": case["name"],
                "search_csv": file_record(directory / "search.csv"), "default_csv": file_record(directory / "default.csv"),
                "default": parser.extract(default, parser.CONFIG), "variants": finalist_variants},
            "frozen finalists differ from independently reproduced search selection")
    validation = [invocation(f"validation-s{seed}", seed, 11, finalist_variants) for seed in SEEDS["validation"]]
    evaluations = []
    for label in finalist_variants:
        if label == REFERENCE:
            continue
        own = [next(row for row in rows if variant(row) == label) for rows in validation]
        refs = [next(row for row in rows if variant(row) == REFERENCE) for rows in validation]
        ratios = [ref["median_us"] / row["median_us"] for ref, row in zip(refs, own)]
        evaluations.append({"variant": label, "config": parser.extract(own[0], parser.CONFIG),
                            "validation_reference_speedups": ratios, "median_validation_reference_speedup": statistics.median(ratios),
                            "median_validation_us": statistics.median(row["median_us"] for row in own)})
    chosen = min(evaluations, key=lambda row: (-row["median_validation_reference_speedup"], row["median_validation_us"], row["variant"]))
    native = min((row for row in evaluations if row["config"]["local_counter"] == "native"),
                 key=lambda row: (row["median_validation_us"], -row["median_validation_reference_speedup"], row["variant"]))
    confirmation_variants = list(dict.fromkeys([chosen["variant"], native["variant"], default_variant, REFERENCE]))
    expected_selection = {"schema": 1, "kind": "narrow_global_experiment_plan", "case": case["name"],
                "workload": case["workload"], "binary_sha256": BINARY_SHA,
                "default": parser.extract(default, parser.CONFIG), "default_variant": default_variant,
                "chosen": chosen["config"], "chosen_variant": chosen["variant"],
                "native_comparator": native["config"], "native_comparator_variant": native["variant"],
                "evaluations": evaluations, "confirmation_variants": confirmation_variants,
                "validation_csvs": [file_record(directory / f"validation-s{seed}.csv") for seed in SEEDS["validation"]],
                "production_change": False}
    selection_path = directory / "selection.json"
    require(saved(selection_path) == expected_selection, "frozen winner/native comparator differ from validation selection")
    confirmations = []
    for seed in SEEDS["confirmation"]:
        name = f"confirmation-s{seed}"
        rows = {variant(row): row for row in invocation(name, seed, 21, confirmation_variants)}
        ours = rows[chosen["variant"]]
        entry = {"seed": seed, "chosen_median_us": ours["median_us"]}
        for title, label in (("native", native["variant"]), ("default", default_variant), ("reference", REFERENCE)):
            entry[title + "_median_us"] = rows[label]["median_us"]
            entry[title + "_over_chosen"] = rows[label]["median_us"] / ours["median_us"]
        entry["artifacts"] = stages[name]["artifacts"]
        confirmations.append(entry)
    confirmation_path = directory / "confirmation.json"
    require(saved(confirmation_path) == {"schema": 1, "case": case["name"],
                "selection": file_record(selection_path), "comparisons": confirmations}, "confirmation record differs")
    require({path for path in directory.iterdir() if path.is_file()} == expected_paths, "unexpected/missing case artifact")
    summary = {}
    for field in ("chosen_median_us", "native_over_chosen", "default_over_chosen", "reference_over_chosen"):
        values = [row[field] for row in confirmations]
        summary[field] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
    return {"case": case["name"], "workload": case["workload"], "status": "complete",
            "search_candidate_count": len(search), "search_fastest_custom": variant(ranked[0]),
            "finalist_variants": finalist_variants, "selection": expected_selection, "confirmations": confirmations,
            "chosen_is_native_comparator": chosen["variant"] == native["variant"],
            "chosen_is_default": chosen["variant"] == default_variant, "summary": summary,
            "stages": stages, "selection_artifacts": [file_record(path) for path in (finalists_path, selection_path, confirmation_path)]}


def markdown(report):
    lines = ["# Narrow global-counter evaluation", "",
             f"Independent CPU audit passed: **4/4 cases**, **{report['invocation_count']} invocations**, "
             f"**{report['measurement_count']} candidate measurements**, and **{report['raw_sample_count']} raw timing samples**.", "",
             "All cases use 16,777,216 uniform shuffled u32 input values, u64 output, warm graph execution, "
             "batch 4, and 200 ms requested warmup on the RTX A5000 Laptop GPU, driver 597.06. "
             "Search has exactly 140 custom configurations and one NVIDIA histogram reference per shape.", "",
             "| Bins | Selected algorithm:policy:grid:local:clear | Selected µs range | Native/selected | NVIDIA/selected |",
             "|---:|---|---:|---:|---:|"]
    for case in report["cases"]:
        summary = case["summary"]
        values = [summary[key] for key in ("chosen_median_us", "native_over_chosen", "reference_over_chosen")]
        lines.append(f"| {case['workload']['bins']:,} | {case['selection']['chosen_variant']} | "
                     + " | ".join(f"{v['min']:.6f}–{v['max']:.6f}" for v in values) + " |")
    lines += ["", "At 24,577 and 65,536 bins, validation selected native counters: narrowing was not the selected improvement. "
              "A native/selected ratio of 1 for an identical policy is a self-comparison, not evidence that narrowing helped. "
              "At 262,144 bins the selected narrow policy showed a small improvement; at 1,048,576 bins it showed a larger one. "
              "The intervals are ranges of two independent confirmation-seed medians, not confidence intervals.", "",
              "## Fresh confirmation measurements", "",
              "| Bins | Seed | Selected µs | Native µs | Default µs | NVIDIA µs | Native/selected | Default/selected | NVIDIA/selected |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        for row in case["confirmations"]:
            fields = ("chosen_median_us", "native_median_us", "default_median_us", "reference_median_us",
                      "native_over_chosen", "default_over_chosen", "reference_over_chosen")
            lines.append(f"| {case['workload']['bins']:,} | {row['seed']} | "
                         + " | ".join(f"{row[key]:.6f}" for key in fields) + " |")
    lines += ["", "## Frozen comparisons", "",
              "| Bins | Native comparator | Resolved production default |", "|---:|---|---|"]
    for case in report["cases"]:
        chosen = case["selection"]
        lines.append(f"| {case['workload']['bins']:,} | {chosen['native_comparator_variant']} | {chosen['default_variant']} |")
    lines += ["", "Search finalists contain the top four custom policies plus the fastest remaining policy from each "
              "(algorithm, local counter) family, the resolved default and the NVIDIA reference. "
              "Validation selects by median paired NVIDIA/custom ratio, then custom median latency and variant. "
              "The native comparator minimizes median validation latency. Both choices are reproduced from validation "
              "alone and fixed before the two disjoint confirmation seeds.", "",
              "## Provenance and limits", "",
              "- Raw samples, derived summaries, exact command/candidate coverage, executable/GPU identity, metadata, "
              "and all persisted finalist/selection/confirmation hashes passed independent checks.",
              "- Production source provenance is checked against narrow-source.tar.gz and its manifest, including the "
              "archived CSV parser. Ongoing source changes are not substituted into this historical audit.",
              f"- Source archive SHA256: `{ARCHIVE_SHA}`.", f"- Measured executable SHA256: `{BINARY_SHA}`.",
              "- Clocks were unlocked; telemetry is sampled before/after invocations. The small 262,144-bin gain "
              "does not establish statistical significance or broader superiority.",
              "- Confirmation compares the selected candidate with the fastest validated native candidate; this is "
              "a bounded search result, not a globally optimal native baseline or a universal speed claim.",
              "- No production defaults are promoted or changed. Different input distributions, sizes, cache/launch "
              "modes and devices require separate validation. NVIDIA histogram remains benchmark-only.",
              "- Audit performed no GPU queries or execution. The JSON preserves every recorded row, raw sample, "
              "command, telemetry record, log, and artifact hash.", ""]
    return "\n".join(lines)


def main():
    manifest_path, environment_path = DATA / "manifest.json", DATA / "environment.json"
    manifest, environment = read_json(manifest_path), read_json(environment_path)
    parser, provenance = audit_provenance(manifest, environment)
    require({path.name for path in (DATA / "cases").iterdir() if path.is_dir()}
            == {case["name"] for case in expected_cases()}, "unexpected/missing case directory")
    state = {"manifest": manifest, "environment": environment, "parser": parser, "common": None}
    cases = [audit_case(case, state) for case in expected_cases()]
    invocations = [stage for case in cases for stage in case["stages"].values()]
    report = {"schema": 1, "kind": "audited_narrow_global_confirmation", "status": "complete",
              "gpu_execution_performed": False, "production_defaults_promoted": False,
              "manifest": {**file_record(manifest_path), "recorded": manifest},
              "environment": {**file_record(environment_path), "recorded": environment},
              "source_provenance": provenance, "common_benchmark_environment": state["common"],
              "case_count": len(cases), "invocation_count": len(invocations),
              "measurement_count": sum(len(stage["measurements"]) for stage in invocations),
              "raw_sample_count": sum(len(row["raw_samples"]) for stage in invocations for row in stage["measurements"]),
              "cases": cases, "analyzer": file_record(Path(__file__))}
    prefix = DATA / "narrow-analysis"
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    prefix.with_suffix(".md").write_text(markdown(report))
    print(f"PASS: {len(cases)} cases, {len(invocations)} invocations, {report['measurement_count']} measurements, "
          f"{report['raw_sample_count']} raw samples; wrote {prefix}.json/.md")
    for case in cases:
        print(case["case"], case["selection"]["chosen_variant"], json.dumps(case["summary"]))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, StopIteration, tarfile.TarError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
