#!/usr/bin/env python3
"""Resume only 13 unverified jobs of the interrupted E256 campaign.

Prepared recovery contract: accept the frozen first37 records only after their
commands, exit receipts, raw hashes, result identities and exactness checks
pass again. Job38's existing JSON/timings are retained but never accepted without
a process-completion receipt. Rerun jobs38..50 with only --output changed, in a
new exclusive directory. Do not regenerate cases, replace old evidence, select
timings or change the original 15-pair/3-warmup experiment. Execution launches
GPU workloads; preparing or AST-checking this file does not.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "results/optimization-20260923/prediction-slab-observed"
DEFAULT_OUTPUT = ROOT / "results/optimization-20260923/prediction-slab-resumed"
HELPER = ROOT / "results/optimization-20260923/run_prediction_slab_campaign.py"
EXPECTED = {
    "protocol.json": "886cf007fc9dd47a173b57b8c19f9334494fb174b53c2648db2d87794f4a9566",
    "observations.json": "b5c44a6d7c11e068ade18cace86abd00b825072a9346ed232689ad1978e9f120",
    "identities.json": "15ff97eb02ed5ce3be4779505168ce69a444b6a4a9d501119a5b1413602cf75b",
}
HELPER_SHA = "3717c8a4b30dd3d1e0867fe2260a36cd9d5920577e6728794ee493c4c6565d4a"
SUFFIXES = (".stdout", ".stderr", ".returncode.json", ".invocation.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_helper():
    require(digest(HELPER) == HELPER_SHA, "original helper changed")
    spec = importlib.util.spec_from_file_location("frozen_slab_campaign", HELPER)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)  # __main__ is not invoked.
    finally:
        sys.dont_write_bytecode = previous
    return module


def mapped_record(job, directory, observation_index, origin, observation):
    name = job["name"]
    return {"name": name, "mode": job["mode"], "origin": origin,
            "result_path": str(directory / (name + ".json")),
            "result_sha256": observation["result_sha256"],
            "observations_path": str(directory / "observations.json"),
            "observation_index": observation_index,
            "raw_paths": {suffix: str(directory / (name + suffix)) for suffix in SUFFIXES},
            "raw_artifact_sha256": observation["raw_artifact_sha256"],
            "returncode": 0, "result_validation_passed": True}


def validate_original(helper, protocol, observations, original_ids):
    require(protocol["pairs"] == 15 and protocol["warmup"] == 3, "original experiment settings differ")
    require(len(protocol["jobs"]) == 50 and len(observations) == 37, "expected exactly50 jobs and37 completed observations")
    require(protocol["driver_sha256"] == HELPER_SHA, "protocol helper identity differs")
    require(protocol["binary"] == str(helper.BIN), "protocol binary differs")
    require(original_ids[protocol["binary"]] == protocol["binary_sha256"], "protocol binary hash differs")
    require(len({job["name"] for job in protocol["jobs"]}) == 50, "duplicate original job name")
    records = []
    for index, observation in enumerate(observations):
        job = protocol["jobs"][index]
        name = job["name"]
        require(observation["name"] == name and observation["mode"] == job["mode"] and
                observation["argv"] == job["argv"], "original observation differs from protocol: " + name)
        require(observation["returncode"] == 0 and observation["result_validation_passed"] is True and
                not observation["timed_out"] and not observation["interrupted"], "original job did not complete: " + name)
        require(observation["binary_sha256"] == protocol["binary_sha256"] and
                observation["model_sha256"] == original_ids[job["case"]["model"]], "original executable/model identity differs")
        require(set(observation["raw_artifact_sha256"]) == set(SUFFIXES), "original raw receipt set differs")
        for suffix in SUFFIXES:
            require(digest(ORIGINAL / (name + suffix)) == observation["raw_artifact_sha256"][suffix],
                    "original raw evidence changed: " + name + suffix)
        receipt = helper.read(ORIGINAL / (name + ".returncode.json"))
        require(all(observation[key] == value for key, value in receipt.items()), "original completion receipt differs")
        invocation = helper.read(ORIGINAL / (name + ".invocation.json"))
        require(all(observation[key] == value for key, value in invocation.items()), "original invocation receipt differs")
        destination = ORIGINAL / (name + ".json")
        require(helper.argument(job["argv"], "--output") == str(destination), "original output location differs")
        require(digest(destination) == observation["result_sha256"], "original result changed: " + name)
        result = helper.read(destination)
        helper.validate_result(result, job["case"], job["mode"], 15, 3)
        require(result["median_ms"] == observation["medians"], "original median receipt differs")
        records.append(dict(mapped_record(job, ORIGINAL, index, "original", observation), original_job_index=index))
    interrupted = protocol["jobs"][37]
    require(interrupted["name"] == "regression1-rows65536-compare-slab-per-tree", "interrupted job differs")
    require(not (ORIGINAL / (interrupted["name"] + ".returncode.json")).exists(),
            "job38 now has a completion receipt; recovery selection must be reviewed")
    retained = [{"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                for path in sorted(ORIGINAL.glob(interrupted["name"] + ".*")) if path.is_file()]
    require((ORIGINAL / (interrupted["name"] + ".json")).is_file(), "expected interrupted result absent")
    return records, {"job": interrupted["name"], "original_job_index": 37,
        "accepted": False, "reason": "Missing process-completion receipt, irrespective of existing JSON or timing values.",
        "artifacts_retained_in_original_directory": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    require(math.isfinite(args.timeout) and args.timeout > 0, "invalid timeout")
    blocked = {key: os.environ[key] for key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH",
               "GH_PROFILE_CAPTURE", "CUDA_LAUNCH_BLOCKING") if os.environ.get(key) not in (None, "", "0")}
    require(not blocked, "refusing instrumented/forced-synchronization timing environment: " + repr(blocked))
    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    out = args.output.absolute()
    require(out != ORIGINAL and ORIGINAL not in out.parents, "recovery output must not be inside original evidence")
    out.mkdir(parents=True, exist_ok=False)
    helper = None
    identities, observations, records, jobs = {}, [], [], []
    completion = {"started_utc": utc(), "status": "setup", "expected_original_records": 37,
                  "expected_new_jobs": 13, "expected_combined_records": 50}
    code = 2
    try:
        helper = load_helper()
        for name, expected in EXPECTED.items():
            require(digest(ORIGINAL / name) == expected, "frozen recovery anchor changed: " + name)
        protocol = helper.read(ORIGINAL / "protocol.json")
        original_observations = helper.read(ORIGINAL / "observations.json")
        original_ids = helper.read(ORIGINAL / "identities.json")
        for path, expected in original_ids.items():
            require(digest(path) == expected, "original campaign input changed: " + path)
        records, unverified = validate_original(helper, protocol, original_observations, original_ids)
        inputs = {Path(path) for path in original_ids} | {Path(__file__).resolve(), HELPER}
        inputs.update(path for path in ORIGINAL.rglob("*") if path.is_file())
        identities = {str(path): digest(path) for path in sorted(inputs)}
        helper.write(out / "identities.json", identities)
        helper.write(out / "accepted-original.json", {"accepted_count": 37, "records": records,
            "validation": "Original protocol, binary/model identities, process receipts, raw/result hashes, exactness and every paired sample revalidated.",
            "original_campaign_identities_checked": len(original_ids), "original_job38_unverified": unverified})
        for index, original_job in enumerate(protocol["jobs"][37:], 37):
            argv = list(original_job["argv"])
            require(argv.count("--output") == 1, "original job output flag differs")
            offset = argv.index("--output") + 1
            argv[offset] = str(out / (original_job["name"] + ".json"))
            require(all(a == b for i, (a, b) in enumerate(zip(argv, original_job["argv"], strict=True)) if i != offset),
                    "recovery changed more than --output")
            jobs.append(dict(original_job, argv=argv, original_job_index=index))
        helper.write(out / "protocol.json", {"schema": "ghb.prediction_slab_resume.v1", "created_utc": helper.utc(),
            "original_protocol": str(ORIGINAL / "protocol.json"), "original_anchor_sha256": EXPECTED,
            "original_helper": str(HELPER), "original_helper_sha256": HELPER_SHA,
            "resume_runner_sha256": identities[str(Path(__file__).resolve())], "binary": protocol["binary"],
            "binary_sha256": protocol["binary_sha256"], "pairs": 15, "warmup": 3,
            "accepted_original_records": 37, "resumed_jobs": 13,
            "selection": "Unverified completion status only; no result-value or timing selection. Existing job38 artifacts remain excluded and untouched.",
            "input_cases": "Original frozen cases, models, control fixtures and feature expectations are reused verbatim.",
            "changed_command_argument": "--output only", "jobs": jobs})
        for job in jobs:
            name, case = job["name"], job["case"]
            observation = {"name": name, "mode": job["mode"], "argv": job["argv"], "cwd": str(ROOT),
                "original_job_index": job["original_job_index"], "binary_sha256": protocol["binary_sha256"],
                "model_sha256": original_ids[case["model"]], "telemetry_before": helper.telemetry()}
            helper.write(out / (name + ".invocation.json"), observation)
            print("START", name, flush=True)
            observation.update(helper.execute(job["argv"], out, name, args.timeout))
            observation["telemetry_after"] = helper.telemetry()
            destination = out / (name + ".json")
            if destination.exists():
                observation["result_sha256"] = digest(destination)
                try:
                    result = helper.read(destination)
                    helper.validate_result(result, case, job["mode"], 15, 3)
                    observation.update(result_validation_passed=True, medians=result["median_ms"])
                except (ValueError, KeyError, TypeError) as error:
                    observation.update(result_validation_passed=False, validation_error=str(error))
            else:
                observation.update(result_validation_passed=False, validation_error="benchmark JSON absent")
            observation["raw_artifact_sha256"] = {suffix: digest(out / (name + suffix)) for suffix in SUFFIXES}
            observations.append(observation)
            helper.write(out / "observations.json", observations, update=True)
            print("END", name, observation["returncode"], observation.get("medians"), flush=True)
            if observation["returncode"] or not observation["result_validation_passed"]:
                completion.update(status="failed", failed_job=name)
                code = observation["returncode"] or 2
                break
            records.append(dict(mapped_record(job, out, len(observations) - 1, "resumed", observation),
                                original_job_index=job["original_job_index"]))
        else:
            completion["status"] = "complete"
            code = 0
    except (Exception, KeyboardInterrupt) as error:
        completion.update(status="failed", error=type(error).__name__ + ": " + str(error))
        code = 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        try:
            post, errors = {}, {}
            for path in identities:
                try:
                    post[path] = digest(path)
                except OSError as error:
                    post[path] = None
                    errors[path] = str(error)
            if identities:
                write(out / "identities-after.json", post)
                changed = [path for path in identities if identities[path] != post[path]]
                write(out / "identity-comparison.json", {"all_unchanged": not changed,
                                                        "changed": changed, "errors": errors})
                if changed:
                    completion.update(status="invalid_identity_drift", changed=changed)
                    code = 2
            write(out / "combined-results.json", {"schema": "ghb.prediction_slab_combined.v1",
                "status": completion["status"], "expected_records": 50,
                "original_records": sum(r["origin"] == "original" for r in records),
                "resumed_records": sum(r["origin"] == "resumed" for r in records),
                "records": records, "original_observations_unchanged":
                    post.get(str(ORIGINAL / "observations.json")) == EXPECTED["observations.json"],
                "original_job38_inclusion": "Excluded; missing process-completion receipt. Use its independently rerun result only.",
                "missing_original_job_indices": sorted(set(range(50)) - {r["original_job_index"] for r in records}),
                "missing_job_names": [job["name"] for job in jobs if not any(r["name"] == job["name"] for r in records)]})
        except (Exception, KeyboardInterrupt) as error:
            completion.update(status="finalization_failed", finalization_error=type(error).__name__ + ": " + str(error))
            code = 2
        completion.update(finished_utc=utc(), accepted_original_records=sum(r["origin"] == "original" for r in records),
                          completed_new_processes=len(observations), verified_new_records=sum(r["origin"] == "resumed" for r in records))
        write(out / "completion.json", completion)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
