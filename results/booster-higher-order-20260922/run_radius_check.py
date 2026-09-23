#!/usr/bin/env python3
"""Fixed validation-only clipping-radius check; root executes GPU jobs serially.

The original campaign, protocol, fixtures, evaluator and results are read-only.
This follow-up investigates a known validation clipping confound. It performs no
test selection, no held-out-test reads and no automatic radius/default promotion.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BASE = WORKSPACE / "training/tools/higher_order_campaign.py"
SPEC = importlib.util.spec_from_file_location("radius_check_immutable_campaign", BASE)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

RADII = (4, 16)
ORDERS = (2, 3, 4)
DATASETS = ("magic", "delicious")
EXPECTED_JOBS = 12


def jobs(binary: Path, root: Path):
    """Enumeration depends only on the registered constants, never metrics."""
    result = []
    for block, (radius, dataset) in enumerate((r, d) for r in RADII for d in DATASETS):
        config = campaign.screen_config(dataset)
        expected = {"rounds": 25 if dataset == "magic" else 5,
                    "depth": 3 if dataset == "magic" else 2,
                    "l2": 1, "learning_rate": .1, "bins": 32,
                    "tree_build": "output-batch", "tree_execution": "graph",
                    "histogram": "global", "output_tile": 16, "tree_export_batch": 16,
                    "max_leaf_value": 1}
        if config != expected:
            raise RuntimeError("base screen configuration differs from the fixed radius-check contract")
        config = {**config, "max_leaf_value": radius}
        rotation = block % len(ORDERS)
        for order in ORDERS[rotation:] + ORDERS[:rotation]:
            name = f"radius-{dataset}-cap{radius}-o{order}"
            result.append({"name": name, "dataset": dataset, "split": "validation",
                           "optimization_order": order, "radius": radius, "config": config,
                           "command": campaign.command(binary, config, order, dataset, "validation", root / name / "result")})
    if len(result) != EXPECTED_JOBS or len({j["name"] for j in result}) != EXPECTED_JOBS:
        raise AssertionError("radius check must contain exactly 12 distinct jobs")
    return result


def fixture_hashes():
    # Deliberately do not call campaign.protocol(): that hashes test fixtures.
    return {f"{dataset}/{split}": campaign.digest(campaign.DATA / dataset / f"{split}.ghb")
            for dataset in DATASETS for split in ("train", "validation")}


def protocol(binary: Path, root: Path, screen: Path, plan):
    return {
        "schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "fixed validation-only check of clipping confounding in the initial higher-order screen",
        "motivation": "Known initial Delicious validation macro AP was about .10556 for unclipped order 2, .07084 for cap-1 order 2, .0723 for cap-1 order 3 and .07259 for cap-1 order 4. This exploratory follow-up tests larger fixed caps; it is not blind confirmation.",
        "jobs": EXPECTED_JOBS, "radii": RADII, "orders": ORDERS, "datasets": DATASETS,
        "configuration": "reuse exact screen shapes: MAGIC 25 rounds/depth 3; Delicious 5 rounds/depth 2; lambda 1, eta .1, bins 32; only max_leaf_value becomes 4 or 16",
        "execution_order": "radius outer, dataset inner; rotate optimizer order by block modulo three; one fresh process per fixed cell",
        "planned_jobs": plan, "run_root": str(root), "initial_validation_screen": str(screen),
        "selection": "none: run and retain all 12 cells; do not select radii, seeds, stopping rules, tests or defaults from this script",
        "data_access": "existing original training and validation fixtures only; no test fixture, test metric, test selection or test protocol reads",
        "binning": "same fixed training fixtures and max_bins=32; existing GPU fitter and missing/category semantics; no extra bin search",
        "thresholds": "unchanged campaign validation signal report: fixed .5 and validation-only threshold maximizing recall under exact five-percent FPR; thresholds are not applied to test data here",
        "comparisons": "all metrics and strict failures retained; matched-cap optimizer comparisons against order 2; cap1 and unclipped2 screen comparisons explicitly have different caps",
        "timing": "serial uninstrumented synchronized complete training/prediction; retain all measured scopes and process times; CPU evaluator runs after each GPU process exits",
        "limitations": ["one fit per cell cannot characterize unordered-FP training variability or data uncertainty",
                        "this is a bounded clipping-confound check, not a broad retuning campaign",
                        "higher orders also change stable/unfloored derivatives, so equal caps alone do not isolate only optimization order",
                        "surrogate safeguards do not guarantee actual loss decrease or held-out quality",
                        "no fastest-code, universal-quality or default-promotion claim"],
        "script": str(Path(__file__).resolve()), "script_sha256": campaign.digest(Path(__file__)),
        "base_driver": str(BASE), "base_driver_sha256": campaign.digest(BASE),
        "source_sha256": campaign.sources(), "fixture_sha256": fixture_hashes(),
        "binary": str(binary), "binary_sha256": campaign.digest(binary), "argv": sys.argv,
    }


def verify_registered(registered, binary):
    if campaign.digest(Path(__file__)) != registered["script_sha256"] or campaign.digest(BASE) != registered["base_driver_sha256"]:
        raise RuntimeError("radius script or immutable base driver changed after registration")
    if campaign.sources() != registered["source_sha256"] or campaign.digest(binary) != registered["binary_sha256"]:
        raise RuntimeError("captured evaluator/source/binary identity changed after registration")
    if fixture_hashes() != registered["fixture_sha256"]:
        raise RuntimeError("training or validation fixture changed after registration")


def validation_case(directory: Path, registered, expected=None):
    # Check split before using the base auditor, which reads the selected fixture.
    first = campaign.read(directory / "capture.json")
    if first.get("split") != "validation" or first.get("dataset") not in DATASETS:
        raise RuntimeError("radius report refuses non-validation captures")
    if expected is not None and any(first.get(key) != expected[key] for key in ("name", "dataset", "split", "config", "optimization_order")):
        raise RuntimeError("radius case differs from the preregistered job")
    capture, quality, measured, signal = campaign.audited(directory)
    if capture["binary_sha256"] != registered["binary_sha256"] or capture["source_sha256"] != registered["source_sha256"]:
        raise RuntimeError("radius/screen comparison does not share source and binary identities")
    dataset = capture["dataset"]
    if capture["training_fixture_sha256"] != registered["fixture_sha256"][f"{dataset}/train"] or capture["evaluation_fixture_sha256"] != registered["fixture_sha256"][f"{dataset}/validation"]:
        raise RuntimeError("radius/screen comparison does not share fixture identities")
    return {"case": directory.name, "directory": str(directory), "dataset": dataset,
            "optimization_order": capture["optimization_order"], "config": capture["config"],
            "capture_sha256": campaign.digest(directory / "capture.json"),
            "metrics": quality["metrics"], "signal": signal,
            "timing_ms": measured["timing_ms"], "memory": measured["memory"],
            "process_wall_ms": capture["process_wall_ms"],
            "quality_process_wall_ms": capture.get("quality_process_wall_ms"),
            "signal_process_wall_ms": capture.get("signal_process_wall_ms"),
            "training_loss_initial": measured["training_loss_initial"],
            "training_loss_final": measured["training_loss_final"]}


def metric_deltas(reference, candidate):
    if reference["dataset"] != candidate["dataset"]:
        raise RuntimeError("cross-dataset radius comparison")
    return {key: candidate["metrics"][key] - value for key, value in reference["metrics"].items()
            if type(value) in (int, float) and type(candidate["metrics"].get(key)) in (int, float)}


def comparison(reference, candidate, scope):
    return {"reference_case": reference["case"], "candidate_case": candidate["case"], "scope": scope,
            "metric_delta_candidate_minus_reference": metric_deltas(reference, candidate),
            "strict_all_metrics_nonregression": campaign.metric_gate(reference["metrics"], candidate["metrics"]),
            "training_time_ratio_reference_over_candidate": reference["timing_ms"]["training_wall"] / candidate["timing_ms"]["training_wall"],
            "prediction_time_ratio_reference_over_candidate": reference["timing_ms"]["prediction_wall"] / candidate["timing_ms"]["prediction_wall"]}


def summarize(root: Path, registered, outcomes, screen: Path):
    result = {"schema": 1, "jobs": EXPECTED_JOBS, "attempted": len(outcomes),
              "passed": sum(o["passed"] for o in outcomes),
              "failed": [o["name"] for o in outcomes if not o["passed"]],
              "scope": "validation-only exploratory clipping-confound check; no selection or promotion",
              "protocol_sha256": campaign.digest(root / "protocol.json"),
              "cases": [], "initial_screen_cases": [], "matched_radius_optimizer_comparisons": [],
              "initial_screen_comparisons": [], "audit_failures": []}
    by_name = {}
    for job in registered["planned_jobs"]:
        try:
            case = validation_case(root / job["name"], registered, job)
            result["cases"].append(case); by_name[job["name"]] = case
        except Exception as error:
            result["audit_failures"].append({"case": job["name"], "error": f"{type(error).__name__}: {error}"})
    for dataset in DATASETS:
        for radius in RADII:
            reference = by_name.get(f"radius-{dataset}-cap{radius}-o2")
            for order in (3, 4):
                candidate = by_name.get(f"radius-{dataset}-cap{radius}-o{order}")
                if reference is not None and candidate is not None:
                    result["matched_radius_optimizer_comparisons"].append(comparison(reference, candidate,
                        "same dataset/config/cap, changed optimizer and derivative convention; one fit per cell"))
    # Original validation screen is read-only; its held-out/test campaign is
    # neither loaded nor consulted. Keep all four original controls per dataset.
    try:
        campaign.completed(screen, 8)
        for dataset in DATASETS:
            for suffix in ("o2-clip1", "o3-clip1", "o4-clip1", "o2-unclipped"):
                directory = screen / f"screen-{dataset}-{suffix}"
                case = validation_case(directory, registered)
                expected = {**campaign.screen_config(dataset), "max_leaf_value": 0 if suffix.endswith("unclipped") else 1}
                if case["config"] != expected or case["optimization_order"] != int(suffix[1]):
                    raise RuntimeError("initial screen shape differs from the fixed clipping control")
                result["initial_screen_cases"].append(case); by_name[case["case"]] = case
        for case in result["cases"]:
            dataset, order = case["dataset"], case["optimization_order"]
            capped = by_name[f"screen-{dataset}-o{order}-clip1"]
            original = by_name[f"screen-{dataset}-o2-unclipped"]
            result["initial_screen_comparisons"].append(comparison(capped, case,
                "same optimizer, larger clipping radius; independent single fits, not a paired trajectory"))
            result["initial_screen_comparisons"].append(comparison(original, case,
                "unclipped order-2 operational reference; different cap and possibly optimizer, not an isolated optimizer comparison"))
    except Exception as error:
        result["audit_failures"].append({"case": "original-validation-screen", "error": f"{type(error).__name__}: {error}"})
    campaign.write(root / "summary.json", result)
    lines = ["# Validation clipping-radius check", "",
             "Fixed caps 4 and 16 were added after the original cap-1 screen exposed a clipping confound. All twelve cells use the original training/validation fixtures and unchanged screen shapes. This check does not select a radius or inspect test results.", "",
             f"Completed job receipts: {len(outcomes)}/{EXPECTED_JOBS}; passed jobs: {result['passed']}; audit failures: {len(result['audit_failures'])}. Complete metrics, operating points, timing scopes, memory payloads and strict comparison failures are retained in summary.json.", "",
             "| Dataset | Order | Cap | Validation AP / macro AP | Log loss | Training wall ms | Prediction wall ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for case in result["initial_screen_cases"] + result["cases"]:
        metric = campaign.SELECTION[case["dataset"]]
        lines.append(f"| {case['dataset']} | {case['optimization_order']} | {case['config']['max_leaf_value']} | {case['metrics'][metric]:.9g} | {case['metrics']['log_loss']:.9g} | {case['timing_ms']['training_wall']:.3f} | {case['timing_ms']['prediction_wall']:.3f} |")
    lines += ["", "Cap zero denotes the original unclipped order-2 control. Radius comparisons are single-fit validation observations; unordered floating-point variation is not estimated here. Higher orders also change stable/unfloored derivatives. A matching cap does not isolate that change, and neither a quality win nor a default promotion follows from this check.", ""]
    with (root / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=HERE / "real/radius-check")
    parser.add_argument("--initial-screen-root", type=Path, default=HERE / "real/screen")
    parser.add_argument("--custom-binary", type=Path, default=HERE / "bin/ghb_real_bench")
    parser.add_argument("--dry-run", action="store_true", help="print the fixed twelve commands without reading fixtures or launching jobs")
    args = parser.parse_args()
    root, screen, binary = args.run_root.resolve(), args.initial_screen_root.resolve(), args.custom_binary.resolve()
    plan = jobs(binary, root)
    if args.dry_run:
        print(json.dumps({"jobs": len(plan), "planned_jobs": plan, "gpu_execution": False}, indent=2))
        return
    if not binary.is_file():
        raise FileNotFoundError(binary)
    registered = protocol(binary, root, screen, plan)
    root.mkdir(parents=True, exist_ok=False)
    campaign.write(root / "protocol.json", registered)
    campaign.write(root / "jobs.json", plan)
    marker = root / ".incomplete"
    marker.write_text("Fixed twelve-job clipping check pending; retain this marker on interruption.\n")
    outcomes = []
    try:
        for job in plan:
            verify_registered(registered, binary)
            try:
                ok = campaign.execute(job["name"], job["dataset"], "validation", job["config"], job["optimization_order"],
                                      root, binary, registered["source_sha256"])
                outcome = {"name": job["name"], "passed": bool(ok)}
            except Exception as error:
                outcome = {"name": job["name"], "passed": False, "error": f"{type(error).__name__}: {error}"}
                campaign.write(root / (job["name"] + ".driver-error.json"), outcome)
            outcomes.append(outcome)
        verify_registered(registered, binary)
    except BaseException as error:
        campaign.write(root / "interrupted.json", {"expected_jobs": EXPECTED_JOBS, "attempted": outcomes,
                       "error": f"{type(error).__name__}: {error}"})
        raise
    completion = {"expected_jobs": EXPECTED_JOBS, "jobs": len(outcomes), "passed": sum(o["passed"] for o in outcomes),
                  "failed": [o["name"] for o in outcomes if not o["passed"]], "outcomes": outcomes,
                  "all_planned_jobs_attempted": len(outcomes) == EXPECTED_JOBS,
                  "source_binary_fixture_identity_rechecked": True,
                  "protocol_sha256": campaign.digest(root / "protocol.json"),
                  "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    campaign.write(root / "completion.json", completion)
    result = summarize(root, registered, outcomes, screen)
    marker.unlink()
    if completion["passed"] != EXPECTED_JOBS or result["audit_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
