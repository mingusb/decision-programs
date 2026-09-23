#!/usr/bin/env python3
"""Freeze and record serial same-process preservation experiments (GPU runner)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "build/preservation-paired/histogram_preservation"
DEFAULT_OUTPUT = ROOT / "results/a5000-paired-preservation"
RECORDER_PATH = ROOT / "results/a5000-profiled/run_round.py"
ANALYZER_PATH = ROOT / "tools/analyze_paired_preservation.py"
PROVENANCE_PATH = ROOT / "build/preservation-paired/sources/provenance.json"
SOURCE_PATHS = [ROOT / "bench/preservation" / name for name in
                ("main.cpp", "backend.cpp", "backend.hpp", "CMakeLists.txt", "prepare_sources.py")]
ARCHIVE_HASHES = {
    "old": "8951509cb6b272507d8cbc14ef0b9b7e71fafc494686dcd59429ba9aee39f28e",
    "new": "8a302c90ec615b5bb04806ef62792ef46d277084ec65d8cef8bc0fb43b9b593b",
}
DATA_SEEDS = (2026092201, 2026092202)
ORDER_SEEDS = (2026092251, 2026092252)
SCHEDULE_SEED = 2026092261
PROTOCOL = {"quartets": 32, "batch": 32, "warmup_ms": 200}
CASES = {
    "single": {"n": 1048576, "bins": 256, "tuning": 6, "blocks": 192,
               "launch": "graph", "clear": "kernel"},
    "stream4096": {"n": 1048576, "bins": 4096, "tuning": 10, "blocks": 48,
                   "launch": "stream", "clear": "runtime"},
}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def pattern_order(seed: int, quartets: int = 32) -> list[str]:
    """Independently reproduce the harness's specified uint64 LCG shuffle."""
    if quartets < 2 or quartets % 2:
        raise ValueError("quartets must be positive and even")
    patterns = ["ABBA"] * (quartets // 2) + ["BAAB"] * (quartets // 2)
    state = seed
    for index in range(quartets - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        other = state % (index + 1)
        patterns[index], patterns[other] = patterns[other], patterns[index]
    return patterns


def schedule() -> list[dict]:
    jobs = []
    for case in CASES:
        for data_seed in DATA_SEEDS:
            for index, order_seed in enumerate(ORDER_SEEDS):
                for comparison in ("old-new" if index == 0 else "new-old", "old-old", "new-new"):
                    jobs.append({"case": case, "data_seed": data_seed,
                                 "order_seed": order_seed, "comparison": comparison})
    random.Random(SCHEDULE_SEED).shuffle(jobs)
    for index, job in enumerate(jobs):
        job["index"] = index
        job["stem"] = f"{index:02d}-{job['case']}-{job['comparison']}-d{job['data_seed']}-o{job['order_seed']}"
        job["patterns"] = pattern_order(job["order_seed"])
    return jobs


def command_for(exe: str, job: dict) -> list[str]:
    return [exe, "--case", job["case"], "--comparison", job["comparison"],
            "--seed", str(job["data_seed"]), "--order-seed", str(job["order_seed"]),
            "--quartets", str(PROTOCOL["quartets"]), "--batch", str(PROTOCOL["batch"]),
            "--warmup-ms", str(PROTOCOL["warmup_ms"])]


def validate_source_provenance(path: Path) -> dict:
    value = read_json(path)
    if (value.get("schema") != 1 or value.get("repo_root") != str(ROOT)
            or value.get("source_directory") != str(path.parent.resolve())
            or value.get("preparer") != str(SOURCE_PATHS[-1])
            or value.get("preparer_sha256") != sha256(SOURCE_PATHS[-1])):
        raise ValueError("source provenance identity or preparer differs")
    if [backend.get("backend") for backend in value.get("backends", [])] != ["old", "new"]:
        raise ValueError("source provenance must identify exactly the old and new archived backends")
    for backend in value["backends"]:
        role = backend["backend"]
        directory = path.parent / role
        if backend["namespace"] != "gh_" + role or backend["source_directory"] != str(directory.resolve()):
            raise ValueError("source namespace or staged directory differs")
        for field in ("archive", "manifest"):
            if sha256(Path(backend[field])) != backend[field + "_sha256"]:
                raise ValueError("changed backend archive/manifest: " + backend[field])
        archived = read_json(Path(backend["manifest"]))
        if (backend["archive_sha256"] != ARCHIVE_HASHES[role]
                or archived["archive_sha256"] != ARCHIVE_HASHES[role]
                or (ROOT / archived["archive"]).resolve() != Path(backend["archive"])
                or archived["files"] != backend["files"]):
            raise ValueError("backend archive does not match the declared historical source")
        expected = set()
        for record in backend["files"]:
            source = directory / record["path"]
            if not source.resolve().is_relative_to(directory.resolve()) or source in expected:
                raise ValueError("unsafe or duplicate staged source path")
            expected.add(source)
            if sha256(source) != record["sha256"]:
                raise ValueError("changed staged archive source: " + str(source))
        actual = {source for source in directory.rglob("*") if source.is_file()}
        if actual != expected:
            raise ValueError("staged source files differ from archived manifest")
    return value


def build_manifest(exe: Path, provenance: Path = PROVENANCE_PATH) -> dict:
    source_provenance = validate_source_provenance(provenance)
    jobs = schedule()
    for job in jobs:
        job["command"] = command_for(str(exe.resolve()), job)
    return {
        "schema": 1, "kind": "same_process_paired_preservation", "binary": file_record(exe),
        "protocol": PROTOCOL, "cases": CASES, "data_seeds": list(DATA_SEEDS),
        "order_seeds": list(ORDER_SEEDS), "schedule_seed": SCHEDULE_SEED,
        "pattern_shuffle": "uint64 LCG Fisher-Yates; a=6364136223846793005; c=1442695040888963407; first half ABBA, second half BAAB",
        "sources": [file_record(path) for path in SOURCE_PATHS],
        "build_artifacts": [file_record(exe.parent / name) for name in
                            ("CMakeCache.txt", "build.ninja", "libpaired_old.a", "libpaired_new.a")],
        "source_provenance": file_record(provenance),
        "source_provenance_content": source_provenance,
        "runner": file_record(Path(__file__)), "analyzer": file_record(ANALYZER_PATH),
        "recorder": file_record(RECORDER_PATH), "jobs": jobs,
        "production_default_promotion": False,
        "interpretation": "Quartets are timing clusters. Real-pair ratios are normalized new/old; AA control ratios are slot B/slot A. Ratios above one indicate a slower numerator.",
        "limitations": [
            "This is a bounded investigation of two previously flagged explicit configurations, not a proof of zero regressions.",
            "GPU clocks are not locked. Process/order summaries and AA controls are descriptive; they are not confidence intervals or significance tests.",
            "The same context, stream, input and output are used for both backends. Each graph slot has a separately captured graph.",
            "Graph submission time covers graph launch; stream submission time includes event recording and the batch of operations.",
            "Total host time covers enqueue through synchronization and can include waiting for preceding untimed warmup. It is not pure kernel latency.",
            "Every timed position is retained and followed by output/canary validation; no outlier deletion or fastest-repeat selection is allowed.",
            "Per-position output/canary checking performs an untimed device-to-host copy and stream synchronization; this differs from the historical benchmark's between-position protocol.",
            "The two archive backends are namespace-adapted into a single executable. This controls process differences but does not reproduce either original standalone binary's layout.",
        ],
    }


def checked_manifest(path: Path, exe: Path, provenance: Path) -> dict:
    wanted = build_manifest(exe, provenance)
    if path.exists():
        if read_json(path) != wanted:
            raise ValueError("frozen manifest differs from current protocol, executable, or sources; use a new output directory")
    else:
        write_new(path, wanted)
    return wanted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="CPU only: print the complete schedule and commands")
    mode.add_argument("--audit", action="store_true", help="CPU only: validate and summarize a completed campaign")
    options = parser.parse_args(argv)
    exe, output, provenance = options.exe.resolve(), options.output_root.resolve(), options.provenance.resolve()
    if options.audit:
        analyzer = import_file("paired_preservation_analyzer", ANALYZER_PATH)
        analyzer.publish(output)
        return 0
    if options.dry_run:
        jobs = schedule()
        for job in jobs:
            job["command"] = command_for(str(exe), job)
        print(json.dumps({"gpu_access": False, "invocations": len(jobs),
                          "raw_positions": len(jobs) * 4 * PROTOCOL["quartets"],
                          "quartet_clusters": len(jobs) * PROTOCOL["quartets"], "jobs": jobs}, indent=2))
        return 0
    manifest_path = output / "manifest.json"
    if not manifest_path.exists() and any((output / "measurements").glob("*")):
        raise ValueError("refusing to adopt existing measurements without a frozen manifest")
    manifest = checked_manifest(manifest_path, exe, provenance)
    recorder = import_file("paired_preservation_recorder", RECORDER_PATH)
    recorder.EXE = exe
    recorder.ensure_environment(output)
    analyzer = import_file("paired_preservation_analyzer", ANALYZER_PATH)
    incomplete_seen = False
    for job in manifest["jobs"]:
        stem = output / "measurements" / job["stem"]
        receipt = stem.with_suffix(".receipt.json")
        artifacts = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
        if receipt.exists():
            if incomplete_seen:
                raise ValueError("completed invocations are not a prefix of the frozen schedule")
            analyzer.validate_job(output, manifest, job)
            continue
        incomplete_seen = True
        if any(path.exists() for path in artifacts):
            raise ValueError("partial invocation artifacts retained; investigate before using a new output directory: " + str(stem))
        if build_manifest(exe, provenance) != manifest:
            raise ValueError("sources or binary changed during campaign")
        print(f"[{job['index'] + 1}/{len(manifest['jobs'])}] {job['stem']}", flush=True)
        recorder.run(stem, job["command"])
        write_new(receipt, {"schema": 1, "index": job["index"],
                            "manifest": file_record(manifest_path),
                            "environment": file_record(output / "environment.json"),
                            "artifacts": [file_record(path) for path in artifacts]})
        analyzer.validate_job(output, manifest, job)
    analyzer.publish(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
