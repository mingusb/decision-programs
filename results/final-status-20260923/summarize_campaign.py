#!/usr/bin/env python3
"""Offline, standard-library-only paired D2 uncertainty; no CUDA execution."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
SEED = 2026092303
RESAMPLES = 20000


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("d2_campaign", HERE / "run_campaign.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    h = driver.helper()
    out, directory = args.output.absolute(), args.campaign.absolute()
    h.require(not out.exists(), "summary output already exists")
    completion = h.read(directory / "completion.json")
    protocol = h.read(directory / "protocol.json")
    identities = h.read(directory / "identities.json")
    after = h.read(directory / "identities-after.json")
    drift = h.read(directory / "identity-comparison.json")
    observations = h.read(directory / "observations.json")
    h.require(completion["status"] == "complete" and drift["all_unchanged"] and identities == after,
              "campaign incomplete or identities changed")
    expected = {"all": 32, "prediction": 28, "encoding": 4}[protocol["scope"]]
    h.require(len(observations) == len(protocol["jobs"]) == completion["expected_gpu_processes"] ==
              completion["observed_gpu_processes"] == completion["verified_gpu_processes"] == expected,
              "campaign job extent differs")
    h.require(h.digest(HERE / "run_campaign.py") == protocol["driver_sha256"] == identities[str(HERE / "run_campaign.py")],
              "campaign validator changed")
    h.require(h.digest(directory / "compiled-provenance.json") == protocol["compiled_provenance_sha256"], "compiled provenance changed")
    rng, rows = random.Random(SEED), []
    for job, observation in zip(protocol["jobs"], observations, strict=True):
        name = job["name"]
        h.require(observation["name"] == name and observation["kind"] == job["kind"] and observation["argv"] == job["argv"],
                  "observation differs from protocol")
        h.require(observation["returncode"] == 0 and not observation["timed_out"] and not observation["interrupted"] and
                  observation["result_validation_passed"] is True, "job did not complete cleanly")
        h.require(observation["binary_sha256"] == identities[job["binary"]] and
                  observation["model_sha256"] == identities[job["case"]["model"]], "job binary/model identities differ")
        h.require(set(observation["raw_artifact_sha256"]) == set(driver.SUFFIXES), "raw receipt extent differs")
        for suffix, expected_hash in observation["raw_artifact_sha256"].items():
            h.require(h.digest(directory / (name + suffix)) == expected_hash, "raw evidence changed")
        for suffix in (".invocation.json", ".returncode.json"):
            receipt = h.read(directory / (name + suffix))
            h.require(all(observation[key] == value for key, value in receipt.items()), "process receipt differs")
        result_path = directory / (name + ".json")
        h.require(h.argument(job["argv"], "--output") == str(result_path), "result output path differs")
        h.require(h.digest(result_path) == observation["result_sha256"], "timing result changed")
        result = h.read(result_path)
        driver.validate_result(h, result, job["case"], job["kind"], protocol["pairs"], protocol["warmup"])
        reference, candidate = driver.POLICIES[job["kind"]]
        by_pair = {}
        for sample in result["samples"]:
            pair = by_pair.setdefault(sample["pair"], {})
            h.require(sample["policy"] not in pair, "duplicate paired sample")
            pair[sample["policy"]] = sample["milliseconds"]
        ratios = [pair[candidate] / pair[reference] for _, pair in sorted(by_pair.items())]
        h.require(len(ratios) == protocol["pairs"] >= 15, "paired sample extent differs")
        medians = {policy: statistics.median(pair[policy] for pair in by_pair.values()) for policy in (reference, candidate)}
        h.require(medians == result["median_ms"] == observation["medians"], "reported medians differ from raw samples")
        bootstrap = sorted(statistics.median(ratios[rng.randrange(len(ratios))] for _ in ratios) for _ in range(RESAMPLES))
        low, high = percentile(bootstrap, .025), percentile(bootstrap, .975)
        ratio = statistics.median(ratios)
        classification = "faster" if high < 1 else "slower" if low > 1 else "inconclusive"
        row = {"case": job["case"]["name"], "kind": job["kind"], "reference": reference, "candidate": candidate,
            "median_ms": medians, "median_paired_candidate_over_reference": ratio, "median_paired_speedup": 1 / ratio,
            "paired_bootstrap_95_interval": [low, high], "casewise_timing_classification": classification,
            "paired_ratios": ratios, "exactness_passed": True, "timing_boundary": result["timing_boundary"],
            "source": str(result_path), "sha256": observation["result_sha256"]}
        if job["kind"] == "prediction":
            row["encoding_host_lengths"] = result["encoding_host_lengths"]
        else:
            row["payload"] = {key: result[key] for key in ("resident_bytes", "peak_bytes", "tile_width", "encoding_tiles",
                              "per_tile_host_lengths_bytes", "final_status_host_lengths_bytes")}
        rows.append(row)
        print(job["kind"], row["case"], f"{ratio:.6f}", f"[{low:.6f},{high:.6f}]", classification, flush=True)
    kinds = sorted({row["kind"] for row in rows})
    counts = {kind: {label: sum(row["kind"] == kind and row["casewise_timing_classification"] == label for row in rows)
                    for label in ("faster", "slower", "inconclusive")} for kind in kinds}
    h.write(out, {"schema": "ghb.final_status_summary.v1", "campaign": str(directory), "scope": protocol["scope"],
        "seed": SEED, "bootstrap_resamples": RESAMPLES, "bootstrap_rng": "Python random.Random",
        "uncertainty": "Percentile bootstrap of median paired candidate/reference ratios; casewise 95%, linear percentile interpolation, no multiplicity correction or global ranking.",
        "boundaries": "Complete prediction and synchronous encoding-stage evidence are separate; never pooled.",
        "conditions": protocol["conditions"], "default_promoted": False, "all_exactness_gates_passed": True,
        "classifications": counts, "cases": rows, "script_sha256": h.digest(Path(__file__)),
        "source_hashes": {name: h.digest(directory / name) for name in ("protocol.json", "observations.json", "completion.json",
                                                                       "identities.json", "identities-after.json", "identity-comparison.json",
                                                                       "compiled-provenance.json")}})
    print("COUNTS", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
