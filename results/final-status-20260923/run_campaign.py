#!/usr/bin/env python3
"""D2 serial experiment: 25 frozen E shapes, 3 additions, 4 separate encode stages.

Executing this script launches GPU jobs. Importing it or --help does not.
All shape choices and timing boundaries are fixed before measuring D2.
No default promotion, sample rejection, training or quality allowance occurs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / "build/final-status-20260923"
PREDICTION = BUILD / "ghb_prediction_bench"
ENCODING = HERE / "encode_bench"
HELPER = ROOT / "results/optimization-20260923/run_prediction_slab_campaign.py"
E = ROOT / "results/optimization-20260923/prediction-slab-observed"
SNAPSHOT = HERE / "source-snapshot"
EXPECTED_HASHES = {
    HELPER: "3717c8a4b30dd3d1e0867fe2260a36cd9d5920577e6728794ee493c4c6565d4a",
    E / "protocol.json": "886cf007fc9dd47a173b57b8c19f9334494fb174b53c2648db2d87794f4a9566",
    E / "identities.json": "15ff97eb02ed5ce3be4779505168ce69a444b6a4a9d501119a5b1413602cf75b",
    SNAPSHOT / "manifest.json": "afb0b32201aea66728500c06705462de384df1619f82bf9f75102b66dff91544",
}
POLICIES = {"prediction": ("fused_output", "fused_output_final_status"),
            "encoding": ("per_tile", "final_status")}
SUFFIXES = (".stdout", ".stderr", ".returncode.json", ".invocation.json")


def helper():
    import hashlib
    for path, expected in EXPECTED_HASHES.items():
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("frozen experiment dependency changed: " + str(path))
    spec = importlib.util.spec_from_file_location("frozen_e_campaign", HELPER)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def frozen_cases(h):
    """Reuse exact E model/input paths, including already frozen empty/one-tree controls."""
    protocol, identities = h.read(E / "protocol.json"), h.read(E / "identities.json")
    h.require(len(protocol["jobs"]) == 50, "frozen E extent differs")
    cases, inputs = {}, set(EXPECTED_HASHES)
    for job in protocol["jobs"]:
        case = job["case"]
        if case["name"] in cases:
            h.require(cases[case["name"]] == case, "E comparisons used different cases")
            continue
        for key in ("model", "features_file", "frozen_result", "control_provenance"):
            if case.get(key):
                path = Path(case[key])
                h.require(h.digest(path) == identities[str(path)], "frozen case input changed: " + str(path))
                inputs.add(path)
        parsed = h.parse_model(Path(case["model"]).read_bytes())
        h.require(parsed["features"] == case["expected"]["features"] and
                  parsed["outputs"] == case["expected"]["outputs"] and
                  len(parsed["trees"]) == case["expected"]["trees"], "frozen model shape differs")
        cases[case["name"]] = case
    h.require(len(cases) == 25, "expected all 25 frozen E shapes")
    return list(cases.values()), inputs


def compiled_provenance(h):
    """Frozen executable/library bytes map to frozen source, not newer working source."""
    manifest = h.read(SNAPSHOT / "manifest.json")
    records = {record["path"]: record for record in manifest["files"]}
    h.require(len(records) == len(manifest["files"]) == 95, "compiled source snapshot extent differs")
    inputs = {SNAPSHOT / "manifest.json"}
    for relative, record in records.items():
        path = SNAPSHOT / relative
        h.require(path.stat().st_size == record["bytes"] and h.digest(path) == record["sha256"],
                  "compiled source snapshot bytes changed: " + relative)
        inputs.add(path)
    binary_map = {}
    for binary in (PREDICTION, BUILD / "libghb.a", BUILD / "libghb_instrumentation.a", BUILD / "CMakeCache.txt"):
        relative = str(binary.relative_to(ROOT))
        h.require(h.digest(binary) == records[relative]["sha256"], "frozen compiled artifact changed: " + relative)
        inputs.add(binary)
        binary_map[str(binary)] = {"sha256": records[relative]["sha256"], "snapshot_copy": str(SNAPSHOT / relative)}
    headers = ("quantize.hpp", "kernels.cuh", "booster.hpp", "instrumentation.hpp")
    for name in headers:
        relative = "training/include/ghb/" + name
        h.require(h.digest(ROOT / relative) == records[relative]["sha256"], "encode helper ABI header differs: " + name)
        inputs.add(ROOT / relative)
    return {"snapshot": str(SNAPSHOT), "manifest_sha256": h.digest(SNAPSHOT / "manifest.json"),
            "compiled_artifacts": binary_map,
            "source_scope": "Prediction executable and linked CUDA libraries are the immutable pre-functional-refactor D2 build; current working source hashes are supplementary stability observations and are not its compiled-source mapping.",
            "encode_helper": "Separately built root-side host helper from encode_bench.cpp, linked to these frozen libraries; all four project headers it transitively includes match the frozen source snapshot."}, inputs


def synthetic_model(features, skew):
    """Deterministic validation-only model, little-endian GHBMODEL v1; no training."""
    dictionaries = []
    for f in range(features):
        kind = int(f % 5 in (1, 3))
        if skew and f == 0:
            kind, values = 1, tuple(float(i - 32767) for i in range(65535))
        elif skew:
            values = (0.0,) if f % 7 == 0 else ()
        else:
            count = (31, 17, 0, 0, 1)[f % 5]
            values = tuple(float(i - count // 2) for i in range(count))
        dictionaries.append((kind, values))
    payload = bytearray(struct.pack("<8s4IQ", b"GHBMODEL", 1, 1, 3, features, 3))
    payload += struct.pack("<3d", -0.5, 0.0, 0.5)
    for kind, values in dictionaries:
        payload += struct.pack("<3I", kind, len(values) if kind == 0 else 0, len(values) if kind else 0)
        payload += struct.pack("<" + str(len(values)) + "f", *values)
    for output, f in enumerate((0, features // 2, features - 1)):
        kind, values = dictionaries[f]
        bins = len(values) + (1 if kind else 2)
        threshold = min(1, bins - 1)
        payload += struct.pack("<2I", output, 3)
        payload += struct.pack("<3i2Id", f, 1, 2, threshold, output % 2, 0.0)
        payload += struct.pack("<3i2Id", -1, -1, -1, 0, 0, -(output + 1) / 8)
        payload += struct.pack("<3i2Id", -1, -1, -1, 0, 0, (output + 1) / 4)
    return bytes(payload), dictionaries


def additional_cases(h, out):
    inputs, models = set(), {}
    for name, columns, skew in (("wide1025", 1025, False), ("normal67", 67, False), ("skew257", 257, True)):
        payload, dictionaries = synthetic_model(columns, skew)
        path = out / (name + ".ghb")
        with path.open("xb") as stream:
            stream.write(payload)
        parsed = h.parse_model(payload)
        h.require(parsed["features"] == columns and parsed["outputs"] == 3 and
                  len(parsed["trees"]) == 3, "synthetic serialization self-check failed")
        counts = [len(values) for _, values in dictionaries]
        provenance = out / (name + ".provenance.json")
        h.write(provenance, {"model": str(path), "model_sha256": h.digest(path), "generator": str(Path(__file__).resolve()),
            "generator_sha256": h.digest(Path(__file__).resolve()), "objective": "independent binary logistic",
            "outputs": 3, "features": columns, "trees": 3, "nodes": 9, "metadata_lengths": counts,
            "metadata_elements": sum(counts), "maximum_metadata": max(counts),
            "input": "deterministic_model_derived_v1 generated by each benchmark; actual FP32 input FNV1a64 retained in result",
            "purpose": "Preselected performance and exactness fixture; synthetic model, not a learned-quality claim"})
        inputs.update((path, provenance))
        models[name] = {"model": str(path), "columns": columns, "max_metadata": max(1, max(counts))}
    prediction = []
    for name, source, rows in (("wide1025-rows4096", "wide1025", 4096),
                               ("large67-rows262144", "normal67", 262144),
                               ("skew257-rows8192", "skew257", 8192)):
        model = models[source]
        prediction.append({"name": name, "model": model["model"], "rows": rows, "features_file": None,
            "expected": {"rows": rows, "features": model["columns"], "outputs": 3, "trees": 3, "nodes": 9,
                         "raw": False, "input_kind": "deterministic_model_derived_v1"}})
    encoding = []
    for name, source, rows, width in (("minimum67-rows257-width1", "normal67", 257, 1),
                                     ("default67-rows257-width32", "normal67", 257, 32),
                                     ("wide1025-rows257-width32", "wide1025", 257, 32),
                                     ("skew257-rows4099-width32", "skew257", 4099, 32)):
        model = models[source]
        columns = model["columns"]
        resident = ((rows * columns * 2 + 3) & ~3) + (columns + 1) * 4 + columns * 4
        peak = resident + 4 + width * (rows * 4 + 4 + model["max_metadata"] * 4)
        encoding.append({"name": name, "model": model["model"], "rows": rows, "tile_width": width,
            "expected": {"rows": rows, "features": columns, "tile_width": width,
                "encoding_tiles": (columns + width - 1) // width, "metadata_stride": model["max_metadata"],
                "resident_bytes": resident, "peak_bytes": peak, "memory_limit": peak,
                "per_tile_host_lengths_bytes": width * 4, "final_status_host_lengths_bytes": columns * 4,
                "input_kind": "deterministic_model_derived_v1"}})
    return prediction, encoding, inputs


def validate_result(h, result, case, kind, pairs, warmup):
    h.require(result["model"] == case["model"], "result model path differs")
    h.require(all(result[key] == value for key, value in case["expected"].items()), "fixture identity/accounting differs")
    h.require(result["pairs"] == pairs and result["warmup_per_policy"] == warmup, "sample selection differs")
    h.require(isinstance(result["feature_fnv1a64"], str) and result["feature_fnv1a64"].isdigit(), "missing input fingerprint")
    if kind == "prediction":
        h.require(result["features_file"] == (case["features_file"] or "") and
                  result["requested_policy"] == "compare-encoding" and
                  result["bitwise_frozen_reference_passed"] is True and
                  result["fused_reference_checked"] is True and
                  result["final_status_reference_checked"] is True and
                  result["slab_reference_checked"] is False and
                  result["reference_encoding_policy"] == "per_tile", "prediction exactness/selection differs")
    else:
        h.require(result["schema"] == "ghb.encode_stage.v1" and
                  result["bitwise_cpu_reference_passed"] is True and
                  result["one_byte_below_minimum_rejected"] is True, "encoding exactness/budget gate failed")
    samples = result["samples"]
    h.require(len(samples) == 2 * pairs, "incomplete paired samples")
    for index, sample in enumerate(samples):
        pair, position = divmod(index, 2)
        order = POLICIES[kind] if pair % 2 == 0 else tuple(reversed(POLICIES[kind]))
        h.require(sample["pair"] == pair and sample["position"] == position and sample["policy"] == order[position],
                  "paired sample ordering differs")
        h.require(math.isfinite(sample["milliseconds"]) and sample["milliseconds"] > 0, "invalid elapsed sample")
    h.require(set(result["median_ms"]) == set(POLICIES[kind]), "median policy set differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "prediction", "encoding"), default="all")
    parser.add_argument("--pairs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    h = helper()
    h.require(15 <= args.pairs <= 10000 and 3 <= args.warmup <= 1000 and math.isfinite(args.timeout) and args.timeout > 0,
              "invalid campaign extent")
    blocked = {key: os.environ[key] for key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH",
                "GH_PROFILE_CAPTURE", "CUDA_LAUNCH_BLOCKING") if os.environ.get(key) not in (None, "", "0")}
    h.require(not blocked, "ranking inherited instrumentation/forced synchronization: " + repr(blocked))
    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    observations, identities = [], {}
    completion = {"started_utc": h.utc(), "status": "setup", "scope": args.scope}
    exit_code = 2
    try:
        cases, inputs = frozen_cases(h)
        provenance, compiled_inputs = compiled_provenance(h)
        inputs |= compiled_inputs
        extra, encoding, created = additional_cases(h, out)
        cases += extra
        inputs |= created
        h.require(len(cases) == 28 and len(encoding) == 4, "campaign shape set changed")
        jobs = []
        for kind, selected, binary in (("prediction", cases, PREDICTION), ("encoding", encoding, ENCODING)):
            if args.scope not in ("all", kind):
                continue
            h.require(binary.is_file(), "benchmark binary missing: " + str(binary))
            inputs.add(binary)
            for case in selected:
                name = kind + "-" + case["name"]
                argv = [str(binary), "--model", case["model"], "--rows", str(case["rows"]),
                        "--pairs", str(args.pairs), "--warmup", str(args.warmup), "--output", str(out / (name + ".json"))]
                if kind == "prediction":
                    argv += ["--policy", "compare-encoding"]
                    if case["features_file"]:
                        argv += ["--features-bin", case["features_file"]]
                else:
                    argv += ["--tile-width", str(case["tile_width"])]
                jobs.append({"name": name, "kind": kind, "case": case, "argv": argv, "binary": str(binary)})
        inputs.update((Path(__file__).resolve(), HERE / "encode_bench.cpp", HERE / "EXPERIMENT_RUNBOOK.md",
                       ROOT / "training/FINAL_STATUS_ENCODING_EXPERIMENT.md", BUILD / "CMakeCache.txt",
                       BUILD / "libghb.a", BUILD / "libghb_instrumentation.a"))
        inputs.update(p for p in (ROOT / "training").rglob("*")
                      if p.is_file() and p.suffix in (".cpp", ".cu", ".cuh", ".hpp", ".inc", ".txt"))
        identities = {str(path): h.digest(path) for path in sorted(inputs)}
        h.write(out / "identities.json", identities)
        h.write(out / "compiled-provenance.json", provenance)
        completion["expected_gpu_processes"] = len(jobs)
        h.write(out / "protocol.json", {"schema": "ghb.final_status_campaign.v1", "created_utc": h.utc(),
            "scope": args.scope, "pairs": args.pairs, "warmup": args.warmup, "jobs": jobs,
            "driver_sha256": identities[str(Path(__file__).resolve())], "execution": "serial, unprofiled, same-binary paired variants",
            "compiled_provenance_sha256": h.digest(out / "compiled-provenance.json"),
            "prediction_boundary": "complete synchronous predict_gpu; setup/context/model/input loading and explicit result comparisons outside timing",
            "encoding_boundary": "encode_quantize entry through synchronous return; resident result destruction and independent CPU/bin/metadata validation outside timing; separate stage evidence",
            "alternation": "reference then candidate on even pairs, reverse on odd pairs; odd count has one extra reference-first pair",
            "conditions": "Record desktop residual rendering; do not claim zero background load. No other GPU jobs, builds or active heavy CPU agents during ranking.",
            "acceptance": "Every raw/transformed prediction precheck, each measured output, each encoding bin/metadata/payload check must pass. No sample removal, quality allowance or default promotion.",
            "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH")}})
        for binary in dict.fromkeys(job["binary"] for job in jobs):
            name = "preflight-" + Path(binary).name
            argv = [binary, "--help"]
            h.write(out / (name + ".invocation.json"), {"argv": argv, "cuda_initialization": False})
            result = h.execute(argv, out, name, 30)
            expected = "compare-encoding" if binary == str(PREDICTION) else "--tile-width"
            h.require(result["returncode"] == 0 and expected in (out / (name + ".stdout")).read_text(), "CLI preflight failed")
        completion["status"] = "running"
        for job in jobs:
            name, case = job["name"], job["case"]
            observation = {"name": name, "kind": job["kind"], "argv": job["argv"], "cwd": str(ROOT),
                "binary_sha256": identities[job["binary"]], "model_sha256": identities[case["model"]], "telemetry_before": h.telemetry()}
            h.write(out / (name + ".invocation.json"), observation)
            print("START", name, flush=True)
            observation.update(h.execute(job["argv"], out, name, args.timeout))
            observation["telemetry_after"] = h.telemetry()
            destination = out / (name + ".json")
            if destination.exists():
                observation["result_sha256"] = h.digest(destination)
                try:
                    result = h.read(destination)
                    validate_result(h, result, case, job["kind"], args.pairs, args.warmup)
                    observation.update(result_validation_passed=True, medians=result["median_ms"])
                except (ValueError, KeyError, TypeError) as error:
                    observation.update(result_validation_passed=False, validation_error=str(error))
            else:
                observation.update(result_validation_passed=False, validation_error="benchmark JSON absent")
            observation["raw_artifact_sha256"] = {suffix: h.digest(out / (name + suffix)) for suffix in SUFFIXES}
            observations.append(observation)
            h.write(out / "observations.json", observations, update=True)
            print("END", name, observation["returncode"], observation.get("medians"), flush=True)
            if observation["returncode"] or not observation["result_validation_passed"]:
                completion.update(status="failed", failed_job=name)
                exit_code = observation["returncode"] or 2
                break
        else:
            completion["status"], exit_code = "complete", 0
    except (OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        completion.update(status="failed", error=type(error).__name__ + ": " + str(error))
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        if identities:
            post = {path: h.digest(path) if Path(path).is_file() else None for path in identities}
            h.write(out / "identities-after.json", post)
            changed = [path for path in identities if identities[path] != post[path]]
            h.write(out / "identity-comparison.json", {"all_unchanged": not changed, "changed": changed})
            if changed:
                completion.update(status="invalid_identity_drift", changed=changed)
                exit_code = 2
        completion.update(finished_utc=h.utc(), observed_gpu_processes=len(observations),
                          verified_gpu_processes=sum(o["returncode"] == 0 and o["result_validation_passed"] for o in observations))
        h.write(out / "completion.json", completion)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
