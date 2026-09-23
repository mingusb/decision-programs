#!/usr/bin/env python3
"""Prepared E256 experiment driver; executing this file launches serial GPU jobs.

Contract recorded before execution: unchanged frozen models/features, complete
host-to-host prediction calls, 15 alternating pairs and 3 warmups by default.
Compare B/E and per-tree/E directly in the same new binary, with all exactness
checks enabled. Reuse all 18 matrix cases and actual Delicious validation.
Add zero/one-tree controls by preserving original serialized payload bytes and
changing only the tree-count header. No training, tuning, metric selection,
profiling, default promotion, or inference-quality claim is performed here.

The output directory must be new. Raw process streams, partial observations,
timeouts, telemetry and before/after identities are retained even on failure.
This driver has intentionally not been executed during its preparation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BIN = ROOT / "build/prediction-slab-20260923/ghb_prediction_bench"
ROWS = (32, 4096, 65536)
POLICIES = {
    "compare-slab": ("fused_output", "fused_output_slab"),
    "compare-slab-per-tree": ("per_tree", "fused_output_slab"),
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, update=False):
    with Path(path).open("w" if update else "x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def argument(argv, name):
    require(argv.count(name) == 1, "expected one argument " + name)
    return argv[argv.index(name) + 1]


def telemetry():
    argv = ["nvidia-smi", "--query-gpu=name,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate",
            "--format=csv"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"utc": utc(), "argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        return {"utc": utc(), "argv": argv, "timed_out": True,
                "stdout": decoded(error.stdout), "stderr": decoded(error.stderr)}
    except OSError as error:
        return {"utc": utc(), "argv": argv, "error": type(error).__name__ + ": " + str(error)}


def parse_model(raw):
    require(len(raw) >= 32, "truncated model header")
    magic, version, objective, outputs, features, count = struct.unpack_from("<8s4IQ", raw)
    require(magic == b"GHBMODEL" and version == 1 and outputs > 0, "unsupported model header")
    position = 32 + outputs * 8
    require(position <= len(raw), "truncated model bases")
    for _ in range(features):
        require(position + 12 <= len(raw), "truncated feature header")
        kind, cuts, categories = struct.unpack_from("<3I", raw, position)
        require(kind in (0, 1), "invalid feature kind")
        position += 12 + 4 * (cuts + categories)
        require(position <= len(raw), "truncated feature payload")
    metadata_end = position
    trees = []
    for index in range(count):
        require(position + 8 <= len(raw), "truncated tree header")
        output, nodes = struct.unpack_from("<2I", raw, position)
        require(output < outputs and nodes > 0, "invalid tree shape")
        end = position + 8 + 28 * nodes
        require(end <= len(raw), "truncated tree payload")
        trees.append({"index": index, "output": output, "nodes": nodes,
                      "begin": position, "end": end})
        position = end
    require(position == len(raw), "unexpected model trailing bytes")
    return {"objective": objective, "outputs": outputs, "features": features,
            "metadata_end": metadata_end, "trees": trees}


def frozen_cases():
    cases, inputs = [], set()
    for phase, count in (("actual", 1), ("matrix", 18)):
        directory = HERE / ("prediction-" + phase + "-idle")
        observations_path = directory / "observations.json"
        observations = read(observations_path)
        require(len(observations) == count, "frozen campaign case count differs")
        inputs.add(observations_path)
        for observation in observations:
            argv = observation["argv"]
            model = Path(argument(argv, "--model"))
            previous_path = Path(argument(argv, "--output"))
            require(observation["returncode"] == 0, "frozen campaign contains failed case")
            require(digest(model) == observation["model_sha256"], "frozen model changed")
            require(digest(previous_path) == observation["result_sha256"], "frozen result changed")
            previous = read(previous_path)
            require(previous["bitwise_frozen_reference_passed"] is True, "frozen exactness failed")
            features = Path(argument(argv, "--features-bin")) if "--features-bin" in argv else None
            inputs.update((model, previous_path, directory / (observation["name"] + ".invocation.json")))
            if features:
                provenance_path = directory / "input.json"
                provenance = read(provenance_path)
                require(digest(features) == provenance["features_sha256"], "actual feature bytes changed")
                fixture = Path(provenance["fixture"])
                require(digest(fixture) == provenance["fixture_sha256"], "actual source fixture changed")
                require(features.stat().st_size == previous["rows"] * previous["features"] * 4,
                        "actual feature extent changed")
                inputs.update((features, fixture, provenance_path))
            require(not previous["raw"], "expected frozen transformed-prediction campaign")
            cases.append({"name": observation["name"], "model": str(model),
                          "rows": previous["rows"], "features_file": str(features) if features else None,
                          "expected": {key: previous[key] for key in
                              ("rows", "features", "outputs", "trees", "nodes", "raw", "input_kind", "feature_fnv1a64")},
                          "frozen_result": str(previous_path)})
    expected_names = {f"{name}-rows{rows}" for name in
        ("regression1", "binary1", "independent3", "multiclass5", "independent65", "independent1024")
        for rows in ROWS} | {"delicious-validation"}
    require({case["name"] for case in cases} == expected_names, "frozen matrix shape set changed")
    # Fixed before seeing E timings: inspect the small model first, then all
    # remaining historical shapes. No case is removed after an observation.
    priority = {"independent3-rows32": 0, "independent3-rows4096": 1,
                "independent3-rows65536": 2, "delicious-validation": 3}
    cases.sort(key=lambda case: (priority.get(case["name"], 4), case["name"]))
    return cases, inputs


def add_controls(cases, out, inputs):
    source_case = next(case for case in cases if case["name"] == "independent3-rows32")
    source = Path(source_case["model"])
    source_hash = digest(source)
    raw = source.read_bytes()
    parsed = parse_model(raw)
    require(parsed["objective"] == 1 and parsed["outputs"] == 3 and parsed["trees"],
            "expected independent three-output control source")
    for count in (0, 1):
        retained = parsed["trees"][:count]
        payload = b"".join(raw[tree["begin"]:tree["end"]] for tree in retained)
        derived = raw[:24] + struct.pack("<Q", count) + raw[32:parsed["metadata_end"]] + payload
        check = parse_model(derived)
        require(len(check["trees"]) == count and derived[:24] == raw[:24], "control slicing failed")
        require(derived[32:check["metadata_end"]] == raw[32:parsed["metadata_end"]],
                "control bases or feature bytes changed")
        model = out / f"frozen-independent3-{count}tree.ghb"
        with model.open("xb") as stream:
            stream.write(derived)
        provenance_path = out / f"frozen-independent3-{count}tree-provenance.json"
        write(provenance_path, {"source": str(source), "source_sha256": source_hash,
            "derived": str(model), "derived_sha256": digest(model),
            "operation": "Only tree-count header bytes24..31 change; all bases, feature metadata and retained tree bytes are copied verbatim. No training or quality equivalence claim.",
            "unchanged_header_range": [0, 24], "unchanged_bases_features_range": [32, parsed["metadata_end"]],
            "retained_trees": retained, "outputs": parsed["outputs"], "trees": count})
        inputs.update((model, provenance_path))
        for rows in ROWS:
            original = next(case for case in cases if case["name"] == f"independent3-rows{rows}")
            expected = dict(original["expected"], trees=count, nodes=sum(t["nodes"] for t in retained))
            cases.append({"name": f"independent3-{count}tree-rows{rows}", "model": str(model),
                          "rows": rows, "features_file": None, "expected": expected,
                          "frozen_result": original["frozen_result"], "control_provenance": str(provenance_path)})
    require(digest(source) == source_hash, "control source changed during slicing")


def terminate(process):
    """Owned uninstrumented benchmark only; no profiler/session descendants."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def execute(argv, out, name, timeout):
    process = None
    status = {"started_utc": utc(), "timed_out": False, "interrupted": False}
    start = time.monotonic()
    with (out / (name + ".stdout")).open("x") as stdout, (out / (name + ".stderr")).open("x") as stderr:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            status["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status["timed_out"] = True
            terminate(process)
            status.update(returncode=124, process_returncode=process.returncode)
        except KeyboardInterrupt:
            status["interrupted"] = True
            if process is not None:
                terminate(process)
            status.update(returncode=130, process_returncode=process.returncode if process else None)
        except OSError as error:
            status.update(returncode=127, error=type(error).__name__ + ": " + str(error))
    status.update(finished_utc=utc(), process_wall_seconds=time.monotonic() - start)
    write(out / (name + ".returncode.json"), status)
    return status


def validate_result(result, case, mode, pairs, warmup):
    require(result["model"] == case["model"] and
            result["features_file"] == (case["features_file"] or ""), "result model/input paths differ")
    require(all(result[key] == value for key, value in case["expected"].items()), "frozen input/model identity differs")
    require(result["requested_policy"] == mode and result["pairs"] == pairs and
            result["warmup_per_policy"] == warmup, "benchmark selection differs")
    require(result["bitwise_frozen_reference_passed"] is True and result["slab_reference_checked"] is True,
            "required frozen-reference exactness absent")
    require(result["fused_reference_checked"] == (mode == "compare-slab"), "B exactness metadata differs")
    samples = result["samples"]
    require(len(samples) == 2 * pairs, "incomplete timing samples")
    for index, sample in enumerate(samples):
        pair, position = divmod(index, 2)
        order = POLICIES[mode] if pair % 2 == 0 else tuple(reversed(POLICIES[mode]))
        require(sample["pair"] == pair and sample["position"] == position and sample["policy"] == order[position],
                "paired sample ordering differs")
        require(math.isfinite(sample["milliseconds"]) and sample["milliseconds"] > 0, "invalid elapsed sample")
    require(set(result["median_ms"]) == set(POLICIES[mode]), "policy median set differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prediction-slab-idle")
    parser.add_argument("--pairs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    require(15 <= args.pairs <= 10000 and 3 <= args.warmup <= 1000 and
            math.isfinite(args.timeout) and args.timeout > 0, "invalid experiment extent")
    blocked = {key: os.environ[key] for key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH",
               "GH_PROFILE_CAPTURE", "CUDA_LAUNCH_BLOCKING") if os.environ.get(key) not in (None, "", "0")}
    require(not blocked, "ranking must not inherit instrumentation or forced synchronization: " + repr(blocked))
    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    observations, identities = [], {}
    completion = {"started_utc": utc(), "status": "setup", "expected_cases": 25, "expected_gpu_processes": 50}
    try:
        require(BIN.is_file(), "missing E benchmark binary: " + str(BIN))
        cases, inputs = frozen_cases()
        add_controls(cases, out, inputs)
        require(len(cases) == 25, "experiment case extent differs")
        inputs.update((BIN, Path(__file__).resolve(), HERE / "run_prediction_campaign.py",
                       ROOT / "training/INFERENCE_MODEL_SLAB_EXPERIMENT.md", HERE / "MODEL_SLAB_SELECTION.md"))
        inputs.update(p for p in (ROOT / "training").rglob("*")
                      if p.is_file() and p.suffix in (".cpp", ".cu", ".cuh", ".hpp", ".inc", ".txt"))
        identities = {str(path): digest(path) for path in sorted(inputs)}
        write(out / "identities.json", identities)
        jobs = []
        for index, case in enumerate(cases):
            modes = tuple(POLICIES) if index % 2 == 0 else tuple(reversed(POLICIES))
            for mode in modes:
                name = case["name"] + "-" + mode
                argv = [str(BIN), "--model", case["model"], "--rows", str(case["rows"]),
                        "--pairs", str(args.pairs), "--warmup", str(args.warmup), "--policy", mode,
                        "--output", str(out / (name + ".json"))]
                if case["features_file"]:
                    argv += ["--features-bin", case["features_file"]]
                jobs.append({"name": name, "case": case, "mode": mode, "argv": argv})
        write(out / "protocol.json", {"created_utc": utc(), "binary": str(BIN), "binary_sha256": identities[str(BIN)],
            "driver_sha256": identities[str(Path(__file__).resolve())], "execution": "serial, unprofiled, complete synchronous prediction calls",
            "default_unchanged": True, "training_or_test_selection": False, "pairs": args.pairs, "warmup": args.warmup,
            "timing_boundary": "Benchmark complete predict_gpu, including packing, allocation, quantization, transfers and cleanup. Process startup, fixture preparation, prechecks and result validation excluded equally.",
            "same_binary_controls": POLICIES, "alternation": "Variant order alternates within each paired process; mode order alternates by case. An odd pair count has one extra reference-first pair.",
            "acceptance": "Exact raw/transformed prechecks plus every measured output; record every failure. Paired uncertainty and capacity differences must be reviewed before any performance/default claim.",
            "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH")},
            "jobs": jobs})
        # --help returns before CUDA initialization in this benchmark. Check the
        # new control mode before spending any GPU time on a stale binary.
        help_argv = [str(BIN), "--help"]
        write(out / "cli-preflight.invocation.json", {"argv": help_argv, "cuda_initialization": False})
        preflight = execute(help_argv, out, "cli-preflight", 30)
        require(preflight["returncode"] == 0 and "compare-slab-per-tree" in (out / "cli-preflight.stdout").read_text(),
                "new same-process E/per-tree control missing from binary")
        for job in jobs:
            name, case = job["name"], job["case"]
            observation = {"name": name, "mode": job["mode"], "argv": job["argv"], "cwd": str(ROOT),
                "binary_sha256": identities[str(BIN)], "model_sha256": identities[case["model"]],
                "telemetry_before": telemetry()}
            write(out / (name + ".invocation.json"), observation)
            print("START", name, flush=True)
            observation.update(execute(job["argv"], out, name, args.timeout))
            observation["telemetry_after"] = telemetry()
            destination = out / (name + ".json")
            if destination.exists():
                observation["result_sha256"] = digest(destination)
                try:
                    result = read(destination)
                    validate_result(result, case, job["mode"], args.pairs, args.warmup)
                    observation.update(result_validation_passed=True, medians=result["median_ms"])
                except (ValueError, KeyError, TypeError) as error:
                    observation.update(result_validation_passed=False, validation_error=str(error))
            else:
                observation.update(result_validation_passed=False, validation_error="benchmark JSON absent")
            observation["raw_artifact_sha256"] = {suffix: digest(out / (name + suffix)) for suffix in
                                                   (".stdout", ".stderr", ".returncode.json", ".invocation.json")}
            observations.append(observation)
            write(out / "observations.json", observations, update=True)
            print("END", name, observation["returncode"], observation.get("medians"), flush=True)
            if observation["returncode"] or not observation["result_validation_passed"]:
                completion.update(status="failed", failed_job=name)
                return observation["returncode"] or 2
        completion["status"] = "complete"
        return 0
    except (OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        completion.update(status="failed", error=type(error).__name__ + ": " + str(error))
        return 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        if identities:
            post = {path: digest(path) if Path(path).is_file() else None for path in identities}
            write(out / "identities-after.json", post)
            changed = [path for path in identities if identities[path] != post[path]]
            write(out / "identity-comparison.json", {"all_unchanged": not changed, "changed": changed})
            if changed:
                completion.update(status="invalid_identity_drift", changed=changed)
        completion.update(finished_utc=utc(), completed_gpu_processes=len(observations))
        write(out / "completion.json", completion)
        # A completed timing loop with changed inputs is not a successful run.
        if completion["status"] == "invalid_identity_drift":
            raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
