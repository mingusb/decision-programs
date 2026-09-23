#!/usr/bin/env python3
"""Read-only CPU audit of recorded loss curves and frozen experiment identities.

This checks recorded per-round losses, not an independent loss recomputation:
per-round predictions are not retained. An exact positive increment fails the
monotonicity observation gate even though true-loss descent is not promised by
the experimental optimizer. No quality gate, configuration or evidence changes.
By default JSON goes to stdout; --output exclusively creates a new audit file.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def safe(value):
    """Preserve invalid observations while emitting strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return [safe(v) for v in value]
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    return value


def inspect_loss(measured):
    """Pure reference checks; no tolerance hides an observed increase."""
    issues = []
    parameters = measured.get("parameters", measured)
    rounds = parameters.get("rounds")
    losses = measured.get("training_loss")
    result = {"rounds": rounds, "optimization_order": parameters.get("optimization_order"),
              "max_leaf_value": parameters.get("max_leaf_value"),
              "learning_rate": parameters.get("learning_rate"), "issues": issues,
              "increases": [], "strict_monotonicity": "not_evaluable"}
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        issues.append("missing or invalid rounds")
    if not isinstance(losses, list) or not losses:
        issues.append("missing or empty training_loss array")
        return result
    result["losses"] = safe(losses)
    result["observed_length"] = len(losses)
    if isinstance(rounds, int) and len(losses) != rounds + 1:
        issues.append("loss count differs from rounds + 1")
    bad = [i for i, x in enumerate(losses) if not numeric(x) or not math.isfinite(x)]
    result["nonfinite_or_nonnumeric_indices"] = bad
    if bad:
        issues.append("nonfinite or nonnumeric training loss")
    else:
        result["initial_loss"], result["final_loss"] = losses[0], losses[-1]
        result["net_loss_change"] = safe(losses[-1] - losses[0])
        for name, actual in (("training_loss_initial", losses[0]), ("training_loss_final", losses[-1])):
            if name in measured and measured[name] != actual:
                issues.append(name + " disagrees with loss sequence endpoint")
        for i, (previous, current) in enumerate(zip(losses, losses[1:]), start=1):
            if current > previous:
                result["increases"].append({"round": i, "previous": previous, "current": current,
                                             "absolute_increase": safe(current - previous)})
        result["strict_monotonicity"] = "failed" if result["increases"] else "passed"
        result["equal_rounds"] = [i for i, (a, b) in enumerate(zip(losses, losses[1:]), 1) if b == a]
        if any(x < 0 for x in losses):
            issues.append("negative recorded logistic loss")
    result["valid_loss_sequence"] = not issues
    return result


def inspect_kernel(rows, pattern):
    observed = [row for row in rows if row.get("ID", "").isdigit()]
    result = {"captured_launches": len(observed), "passed": False}
    if len(observed) == 1:
        result["kernel"] = observed[0].get("Kernel Name", "")
        result["passed"] = bool(pattern) and re.fullmatch(pattern, result["kernel"]) is not None
    return result


def profile_settings(job, output_directory):
    """Permit retry changes only to output paths and the capture filter."""
    result = {k: v for k, v in job.items() if k not in ("command", "expected_kernel_regex")}
    command = []
    previous = None
    for argument in job.get("command", []):
        if previous == "--kernel-name":
            command.append("<capture-filter>")
        elif argument.startswith(str(output_directory) + "/"):
            command.append("<profile-output>/" + argument[len(str(output_directory)) + 1:])
        else:
            command.append(argument)
        previous = argument
    result["command"] = command
    return result


class Audit:
    def __init__(self, evidence, workspace):
        self.evidence = evidence.resolve()
        self.workspace = workspace.resolve()
        self.hash_cache = {}
        self.errors = []
        self.pending = []
        self.identities = []
        self.results = []
        self.seen_results = set()
        self.seen_completion = set()
        self.frozen_binaries = {}
        self.stage_status = {}
        self.profile_jobs = []
        self.historical_failures = []
        self.retry_provenance = {}

    def label(self, path):
        try:
            return str(path.resolve().relative_to(self.evidence))
        except ValueError:
            return str(path)

    def error(self, path, message):
        self.errors.append({"path": self.label(path), "message": message})

    def read(self, path):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            self.error(path, f"{type(error).__name__}: {error}")
            return None
        return value

    def required(self, path):
        if not path.exists():
            self.pending.append({"path": self.label(path), "reason": "required registered stage artifact missing"})
            return None
        return self.read(path)

    def incomplete(self, directory):
        marker = directory / ".incomplete"
        if marker.exists():
            self.pending.append({"path": self.label(marker), "reason": "runner completion marker remains"})

    def expected_jobs(self, path, jobs, expected, key="name"):
        if not isinstance(jobs, list) or any(not isinstance(j, dict) or key not in j for j in jobs):
            self.error(path, "invalid registered job list")
            return []
        if len(jobs) != expected or len({j[key] for j in jobs}) != expected:
            self.error(path, f"expected {expected} distinct registered jobs, found {len(jobs)}")
        return jobs

    def runner_summary(self, path, expected):
        summary = self.required(path)
        if not isinstance(summary, dict):
            if summary is not None:
                self.error(path, "invalid stage summary")
            return
        if summary.get("jobs") != expected or summary.get("passed") != expected or summary.get("failed"):
            self.error(path, "stage summary does not report every expected job passed")
        if summary.get("audit_failures"):
            self.error(path, "stage summary contains audit failures: " + str(summary["audit_failures"]))

    def real_stage_completion(self):
        # The original plan fixes workloads; it predates the final production
        # build. Executed stage protocols, not its earlier binary, anchor runs.
        plan_path = self.evidence / "real/plan/protocol.json"
        plan = self.required(plan_path)
        counts = {"screen": 8, "validation": 24, "test": 18, "radius-check": 12}
        if isinstance(plan, dict):
            for stage in ("screen", "validation", "test"):
                if plan.get(stage, {}).get("jobs") != counts[stage]:
                    self.error(plan_path, "original plan job count differs for " + stage)
        for stage, count in counts.items():
            directory = self.evidence / "real" / stage
            self.stage_status["real/" + stage] = {"expected_loss_curves": count}
            protocol_path = directory / "protocol.json"
            protocol = self.required(protocol_path)
            jobs = self.required(directory / "jobs.json")
            if jobs is not None:
                self.expected_jobs(directory / "jobs.json", jobs, count)
            if isinstance(protocol, dict):
                binary = Path(protocol.get("binary", ""))
                expected = self.frozen_binaries.get(binary.name)
                if expected is None or protocol.get("binary_sha256") != expected:
                    self.error(protocol_path, "executed stage binary does not match frozen production binary")
                else:
                    self.verify(binary, expected, "registered_stage_binary")
                if stage != "radius-check" and protocol.get("stage") != stage:
                    self.error(protocol_path, "registered stage name differs from directory")
                if stage == "radius-check":
                    if protocol.get("jobs") != count or protocol.get("planned_jobs") != jobs:
                        self.error(protocol_path, "radius registration differs from jobs.json")
                    for filename, hashkey in (("script", "script_sha256"), ("base_driver", "base_driver_sha256")):
                        if filename in protocol:
                            self.verify(Path(protocol[filename]), protocol.get(hashkey), "registered_radius_source")
            self.incomplete(directory)
            self.runner_summary(directory / "summary.json", count)
            if stage == "radius-check":
                for path in sorted(directory.glob("*.driver-error.json")) + ([directory / "interrupted.json"] if (directory / "interrupted.json").exists() else []):
                    self.error(path, "recorded radius driver/interruption failure: " + str(self.read(path)))
                completion_path = directory / "completion.json"
                completion = self.required(completion_path)
                if isinstance(completion, dict):
                    if any(completion.get(k) != count for k in ("expected_jobs", "jobs", "passed")) or completion.get("failed"):
                        self.error(completion_path, "radius completion does not report all jobs passed")
                    if completion.get("all_planned_jobs_attempted") is not True or completion.get("source_binary_fixture_identity_rechecked") is not True:
                        self.error(completion_path, "radius completion/identity confirmation is missing")
                    outcomes = completion.get("outcomes", [])
                    if not isinstance(outcomes, list) or len(outcomes) != count or any(o.get("passed") is not True for o in outcomes):
                        self.error(completion_path, "radius completion contains failed or missing outcomes")
                    if protocol_path.exists():
                        self.verify(protocol_path, completion.get("protocol_sha256"), "radius_completion_protocol")

    def sha(self, path):
        key = path.resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self.hash_cache.get(key)
        if cached and cached[0] == signature:
            return cached[1]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        # Identity must not be accepted if the file changed while being read.
        after = path.stat()
        if signature != (after.st_size, after.st_mtime_ns):
            raise OSError("file changed during hash calculation")
        result = digest.hexdigest()
        self.hash_cache[key] = (signature, result)
        return result

    def verify(self, path, expected, category):
        record = {"category": category, "path": self.label(path), "expected_sha256": expected}
        try:
            record["actual_sha256"] = self.sha(path)
            record["passed"] = record["actual_sha256"] == expected
            if not record["passed"]:
                self.error(path, category + " hash mismatch")
        except OSError as error:
            record["passed"] = False
            record["error"] = str(error)
            self.error(path, category + ": " + str(error))
        self.identities.append(record)

    def provenance(self):
        directory = self.evidence / "production-provenance"
        manifest = self.read(directory / "source-sha256.json")
        if not isinstance(manifest, dict) or not manifest:
            self.error(directory, "missing or invalid production source manifest")
        else:
            for relative, expected in manifest.items():
                self.verify(self.workspace / relative, expected, "production_source")
        before = self.read(self.evidence / "prechange/source-sha256.json")
        if isinstance(before, dict):
            counts = {k: v for k, v in before.items()
                      if k == "CMakeLists.txt" or k.startswith(("src/", "include/"))}
            if len(counts) != 9:
                self.error(directory, f"expected nine frozen count files, found {len(counts)}")
            for relative, expected in counts.items():
                self.verify(self.workspace / relative, expected, "frozen_count_source")
        else:
            self.error(directory, "missing or invalid prechange source manifest")
        binaries = self.read(directory / "binary-sha256.json")
        if not isinstance(binaries, dict) or not binaries:
            self.error(directory, "missing or invalid frozen binary manifest")
        else:
            self.frozen_binaries = binaries
            for name, expected in binaries.items():
                self.verify(self.evidence / "bin" / name, expected, "frozen_binary")

    def captured_binary(self, capture, path, expected=None):
        command = capture.get("command")
        if not isinstance(command, list) or not command:
            self.error(path, "missing captured command")
            return
        binary = Path(command[0])
        if not binary.is_absolute():
            binary = Path(capture.get("cwd", self.workspace)) / binary
        frozen = self.frozen_binaries.get(binary.name)
        recorded = capture.get("binary_sha256", expected)
        if frozen is None or recorded != frozen:
            self.error(path, "captured binary hash does not match frozen binary manifest")
        if recorded is not None:
            self.verify(binary, recorded, "captured_binary")
        if capture.get("binary_unchanged", True) is not True:
            self.error(path, "runner recorded a binary change")

    def result(self, path, scope, completion):
        self.seen_results.add(path.resolve())
        measured = self.read(path)
        if not isinstance(measured, dict):
            self.error(path, "missing or invalid training result")
            return
        inspected = inspect_loss(measured)
        inspected.update(path=self.label(path), scope=scope, completion_evidence=completion,
                         result_sha256=self.sha(path))
        self.results.append(inspected)
        for issue in inspected["issues"]:
            self.error(path, issue)

    def real(self):
        for path in sorted((self.evidence / "real").rglob("capture.json")):
            capture = self.read(path)
            if not isinstance(capture, dict):
                continue
            self.seen_completion.add(path.resolve())
            metrics = path.parent / "result/metrics.json"
            self.seen_results.add(metrics.resolve())
            if "returncode" not in capture:
                self.pending.append({"path": self.label(path), "reason": "training completion not recorded"})
                continue
            self.captured_binary(capture, path)
            for key in ("returncode", "quality_returncode", "signal_returncode"):
                if key in capture and capture[key] != 0:
                    self.error(path, f"recorded {key}={capture[key]}")
            if capture.get("contract_error"):
                self.error(path, "recorded contract_error: " + str(capture["contract_error"]))
            if capture["returncode"] != 0:
                # Partial results are still audited if the failed process left any.
                if not metrics.exists():
                    continue
            else:
                for key in ("quality_returncode", "signal_returncode"):
                    if key not in capture:
                        self.pending.append({"path": self.label(path), "reason": key + " not recorded"})
            expected = capture.get("artifact_sha256", {}).get("result/metrics.json")
            if expected:
                self.verify(metrics, expected, "captured_loss_result")
            self.result(metrics, "real/" + path.relative_to(self.evidence / "real").parts[0],
                        {"capture": self.label(path), "returncode": capture["returncode"]})
        for path in sorted((self.evidence / "real").rglob("jobs.json")):
            jobs = self.read(path)
            if not isinstance(jobs, list):
                self.error(path, "invalid planned real jobs")
                continue
            for job in jobs:
                capture = path.parent / job["name"] / "capture.json"
                if capture.resolve() not in self.seen_completion:
                    self.pending.append({"path": self.label(capture), "reason": "planned job has no capture"})

    def synthetic(self):
        directory = self.evidence / "synthetic"
        self.stage_status["synthetic"] = {"expected_loss_curves": 30, "measured": 27, "warmup": 3}
        protocol_path = directory / "protocol.json"
        if not protocol_path.exists():
            self.pending.append({"path": self.label(protocol_path), "reason": "synthetic campaign not recorded"})
            return
        protocol = self.read(protocol_path)
        if not isinstance(protocol, dict):
            return
        jobs = self.expected_jobs(protocol_path, protocol.get("jobs"), 30, "tag")
        if sum(j.get("case") == "warmup" for j in jobs) != 3:
            self.error(protocol_path, "expected three synthetic warmups")
        self.verify(self.evidence / "run_synthetic.py", protocol.get("script_sha256"), "registered_synthetic_source")
        self.required(directory / "summary.json")
        for job in jobs:
            path = directory / (job["tag"] + ".receipt.json")
            result_path = directory / job["tag"] / "result.json"
            self.seen_results.add(result_path.resolve())
            if not path.exists():
                self.pending.append({"path": self.label(path), "reason": "planned job has no receipt"})
                continue
            receipt = self.read(path)
            if not isinstance(receipt, dict):
                continue
            if receipt.get("command") != job.get("command"):
                self.error(path, "synthetic receipt command differs from registration")
            self.captured_binary(receipt, path, protocol.get("binary_sha256"))
            if "returncode" not in receipt:
                self.pending.append({"path": self.label(path), "reason": "completion not recorded"})
                continue
            if receipt["returncode"] != 0:
                self.error(path, f"recorded returncode={receipt['returncode']}")
                if not result_path.exists():
                    continue
            scope = "synthetic/warmup" if job.get("case") == "warmup" else "synthetic/measured"
            self.result(result_path, scope, {"receipt": self.label(path), "returncode": receipt["returncode"]})

    def profile_receipt(self, path, identity, expected_command=None):
        receipt = self.required(path)
        if not isinstance(receipt, dict):
            return None
        if "returncode" not in receipt:
            self.pending.append({"path": self.label(path), "reason": "profile command completion not recorded"})
            return receipt
        if receipt.get("returncode") != 0 or receipt.get("execution_error"):
            self.error(path, "profile command failed: " + str(receipt.get("execution_error", receipt.get("returncode"))))
        for key in ("binary_unchanged", "executable_unchanged", "runner_unchanged"):
            if receipt.get(key) is not True:
                self.error(path, "profile receipt does not confirm " + key)
        if receipt.get("binary_sha256") != self.frozen_binaries.get("ghb_real_bench"):
            self.error(path, "profile receipt binary differs from frozen production binary")
        if receipt.get("binary_sha256") != identity.get("binary_sha256") or receipt.get("runner_sha256") != identity.get("runner_sha256"):
            self.error(path, "profile receipt identity differs from registration")
        command = receipt.get("command")
        if expected_command is not None and command != expected_command:
            self.error(path, "profile command differs from registration")
        if not isinstance(command, list) or not command:
            self.error(path, "missing profile executable command")
        else:
            executable = Path(command[0])
            registered = identity.get("profiler_sha256", {}).get(str(executable))
            if registered is None or registered != receipt.get("executable_sha256"):
                self.error(path, "profile executable identity differs from registration")
            self.verify(executable, receipt.get("executable_sha256"), "captured_profiler_executable")
        stem = path.name.removesuffix("-command.json")
        for stream in ("stdout", "stderr"):
            self.verify(path.parent / f"{stem}.{stream}", receipt.get(stream + "_sha256"), "captured_profile_log")
        return receipt

    def profiles(self, historical=False):
        errors_before, pending_before = len(self.errors), len(self.pending)
        stage = "profiles-attempt1" if historical else "profiles-v2"
        directory = self.evidence / ("profiles" if historical else "profiles-v2")
        self.stage_status[stage] = {"expected_loss_curves": 5 if historical else 12,
                                   "timings_are_diagnostic_only": True,
                                   "historical_failed_attempt": historical}
        plan_path = self.evidence / ("profiles-plan.json" if historical else "profiles-plan-v2.json")
        plan = self.required(plan_path)
        planned = self.expected_jobs(plan_path, plan.get("jobs"), 12) if isinstance(plan, dict) else []
        protocol_path = directory / "protocol.json"
        protocol = self.required(protocol_path)
        if historical:
            if not (directory / ".incomplete").exists():
                self.error(directory, "historical failed profile attempt lost its incomplete marker")
        else:
            self.incomplete(directory)
        if not isinstance(protocol, dict):
            return
        jobs = self.expected_jobs(protocol_path, protocol.get("jobs"), 12)
        if jobs != planned:
            self.error(protocol_path, "executed profile job registration differs from pre-execution plan")
        if not historical:
            original_plan_path = self.evidence / "profiles-plan.json"
            original = self.required(original_plan_path)
            original_jobs = original.get("jobs", []) if isinstance(original, dict) else []
            settings_unchanged = [profile_settings(j, self.evidence / "profiles") for j in original_jobs] == [profile_settings(j, directory) for j in jobs]
            if not settings_unchanged:
                self.error(protocol_path, "profiling retry changes more than capture filters and output paths")
            if not isinstance(protocol.get("retry_reason"), str) or not protocol["retry_reason"]:
                self.error(protocol_path, "profiling retry reason is not recorded")
            self.retry_provenance = {"original_directory": "profiles", "retry_directory": "profiles-v2",
                                     "declared_retry_reason": protocol.get("retry_reason"),
                                     "only_capture_filter_and_output_path_changes": settings_unchanged,
                                     "production_identity": "both attempts verified against same frozen source/binary manifests",
                                     "original_failure_status_must_remain": "failed", "quality_success_claim": False}
            for label, path in (("original_plan_sha256", original_plan_path), ("retry_plan_sha256", plan_path),
                                ("retry_protocol_sha256", protocol_path), ("original_summary_sha256", self.evidence / "profiles/summary.json")):
                if path.exists():
                    self.retry_provenance[label] = self.sha(path)
        identity = protocol.get("identity", {})
        if identity.get("binary_sha256") != self.frozen_binaries.get("ghb_real_bench"):
            self.error(protocol_path, "registered profile binary differs from frozen production binary")
        self.verify(self.evidence / "bin/ghb_real_bench", identity.get("binary_sha256"), "registered_profile_binary")
        runner = self.evidence / ("run_profiles.py" if historical else "run_profiles_v2.py")
        self.verify(runner, identity.get("runner_sha256"), "registered_profile_runner")
        for filename, expected in identity.get("provenance_sha256", {}).items():
            self.verify(Path(filename), expected, "registered_profile_provenance")
        production_path = self.evidence / "production-provenance/source-sha256.json"
        production = self.read(production_path)
        if identity.get("source_sha256") != production:
            self.error(protocol_path, "registered profile production sources differ from frozen manifest")
        completed_audits = {}
        attempted_jobs = jobs[:5] if historical else jobs
        if historical:
            self.stage_status[stage]["original_registered_jobs"] = len(jobs)
            self.stage_status[stage]["unattempted_jobs"] = [job["name"] for job in jobs[5:]]
            if [job["name"] for job in attempted_jobs] != ["nsys-o2", "nsys-o3", "nsys-o4", "ncu-o2-derivative", "ncu-o2-root"]:
                self.error(protocol_path, "historical attempted job sequence differs from retained failure contract")
        for job in attempted_jobs:
            destination = directory / job["name"]
            expected_filter_failure = historical and job["name"] == "ncu-o2-root"
            metrics = destination / "benchmark/metrics.json"
            self.seen_results.add(metrics.resolve())
            command = job.get("command", [])
            binary = str(self.evidence / "bin/ghb_real_bench")
            if binary not in command:
                self.error(protocol_path, "profile command does not invoke the frozen benchmark: " + job["name"])
            profile = self.profile_receipt(destination / "profile-command.json", identity, command)
            names = ("stats",) if job.get("kind") == "systems" else ("details", "raw")
            if expected_filter_failure:
                names = ()  # No report existed for the failed original filter.
            for name in names:
                self.profile_receipt(destination / (name + "-command.json"), identity)
            audit_path = destination / "audit.json"
            audit = self.required(audit_path)
            entry = {"name": job["name"], "kind": job.get("kind"), "order": job.get("order"),
                     "attempt": stage, "timings_are_diagnostic_only": True, "audited_loss_curve": False}
            self.profile_jobs.append(entry)
            if isinstance(audit, dict):
                completed_audits[job["name"]] = audit
                entry["runner_passed"] = audit.get("passed")
                if expected_filter_failure:
                    if audit.get("passed") is not False or audit.get("error") != "RuntimeError: missing Nsight Compute report":
                        self.error(audit_path, "historical filter failure differs from its retained declared failure")
                elif audit.get("passed") is not True or audit.get("error"):
                    self.error(audit_path, "profiler runner recorded failure: " + str(audit.get("error", "passed=false")))
                if audit.get("configuration_verified") is not True:
                    self.error(audit_path, "profile workload configuration was not verified")
                for key in ("name", "kind", "order"):
                    if audit.get(key) != job.get(key):
                        self.error(audit_path, "profile audit identity differs from planned " + key)
                for filename, expected in audit.get("artifacts_sha256", {}).items():
                    self.verify(destination / filename, expected, "captured_profile_artifact")
                self.verify(metrics, audit.get("metrics_sha256"), "captured_profile_loss_result")
            if metrics.exists() and profile is not None and "returncode" in profile:
                self.result(metrics, stage + "/" + str(job.get("kind")),
                            {"receipt": self.label(destination / "profile-command.json"),
                             "returncode": profile.get("returncode"), "timings_are_diagnostic_only": True})
                entry["audited_loss_curve"] = True
            elif profile is not None and profile.get("returncode") == 0:
                self.error(metrics, "successful profiler command has no benchmark loss result")
            if expected_filter_failure:
                stdout_path = destination / "profile.stdout"
                stdout = stdout_path.read_text() if stdout_path.exists() else ""
                no_kernel = "No kernels were profiled." in stdout
                available = next((line for line in stdout.splitlines()
                                  if "accumulate<(unsigned int)16, (bool)1, (bool)1>" in line), None)
                absent = all(not (destination / name).exists() for name in
                             ("profile.ncu-repz", "details-command.json", "raw-command.json", "raw.stdout"))
                if not no_kernel or not available or not absent or profile is None or profile.get("returncode") != 0:
                    self.error(destination, "historical no-kernel/missing-report/exit-zero evidence is inconsistent")
                if available and re.fullmatch(job.get("expected_kernel_regex", ""), available):
                    self.error(stdout_path, "historical filter unexpectedly matches its listed available kernel")
                entry["kernel_observation"] = {"passed": False, "captured_launches": 0,
                                               "no_kernel_warning": no_kernel, "listed_available_kernel": available,
                                               "report_and_exports_absent": absent}
                self.historical_failures.append({"stage": stage, "job": job["name"],
                    "recorded_runner_passed": False, "recorded_error": audit.get("error") if isinstance(audit, dict) else None,
                    "process_returncode": profile.get("returncode") if isinstance(profile, dict) else None,
                    "captured_launches": 0, "loss_curve_still_audited": entry["audited_loss_curve"],
                    "classification": "instrumentation kernel-filter mismatch; original attempt remains failed",
                    "resolution": "pending verified profiles-v2 retry", "quality_success_claim": False})
            elif job.get("kind") == "compute" and (destination / "raw.stdout").exists():
                try:
                    with (destination / "raw.stdout").open(newline="") as stream:
                        observation = inspect_kernel(list(csv.DictReader(stream)), job.get("expected_kernel_regex", ""))
                    entry["kernel_observation"] = observation
                    if not observation["passed"]:
                        self.error(destination / "raw.stdout", "expected exactly one matching registered kernel capture")
                    if isinstance(audit, dict) and (audit.get("captured_launches") != observation["captured_launches"] or audit.get("kernel") != observation.get("kernel")):
                        self.error(audit_path, "kernel audit differs from retained raw Nsight output")
                except (OSError, csv.Error, re.error) as error:
                    self.error(destination / "raw.stdout", "cannot audit kernel evidence: " + str(error))
        summary_path = directory / "summary.json"
        summary = self.required(summary_path)
        if isinstance(summary, dict):
            expected_summary = {"expected": 12, "completed": 5 if historical else 12,
                                "passed": 4 if historical else 12, "complete": not historical}
            if any(summary.get(key) != value for key, value in expected_summary.items()):
                self.error(summary_path, "profile stage summary differs from expected retry/history status")
            self.stage_status[stage]["recorded_summary_status"] = {key: summary.get(key) for key in expected_summary}
            if summary.get("times_are_diagnostic_only") is not True:
                self.error(summary_path, "profile summary lacks diagnostic-only timing scope")
            summaries = summary.get("jobs", [])
            if not isinstance(summaries, list) or summaries != [completed_audits.get(job["name"]) for job in attempted_jobs]:
                self.error(summary_path, "profile summary jobs differ from individual retained audits")
        return len(self.errors) == errors_before and len(self.pending) == pending_before

    def run(self):
        self.provenance()
        self.real_stage_completion()
        self.real()
        self.synthetic()
        historical_verified = self.profiles(historical=True)
        retry_verified = self.profiles()
        for failure in self.historical_failures:
            failure["historical_evidence_verified"] = bool(historical_verified)
            if historical_verified and retry_verified:
                failure["resolution"] = "resolved instrumentation failure by separately verified complete profiles-v2 retry; original runner status remains failed"
        # Includes completed smoke checks and any orphan result left after failure.
        for name in ("result.json", "metrics.json"):
            for path in sorted(self.evidence.rglob(name)):
                if path.resolve() in self.seen_results or "prechange" in path.relative_to(self.evidence).parts:
                    continue
                self.result(path, "diagnostic/untracked", {"result_file_only": True,
                            "caveat": "No completion receipt associates this result with the frozen executable."})
        groups = {}
        for result in self.results:
            key = f"{result['scope']}/order{result['optimization_order']}"
            group = groups.setdefault(key, {"runs": 0, "valid_sequences": 0, "loss_increases": 0,
                                           "runs_with_increases": 0, "recorded_round_transitions": 0})
            group["runs"] += 1
            group["valid_sequences"] += bool(result.get("valid_loss_sequence"))
            group["loss_increases"] += len(result["increases"])
            group["runs_with_increases"] += bool(result["increases"])
            group["recorded_round_transitions"] += max(0, result.get("observed_length", 0) - 1)
        adverse = [{"path": r["path"], "optimization_order": r["optimization_order"],
                    "increases": r["increases"]} for r in self.results if r["increases"]]
        fully_evaluable = bool(self.results) and all(r.get("valid_loss_sequence") for r in self.results)
        for stage, state in self.stage_status.items():
            scopes = ["synthetic/measured", "synthetic/warmup"] if stage == "synthetic" else [stage + "/systems", stage + "/compute"] if stage.startswith("profiles") else [stage]
            state["observed_loss_curves"] = sum(r["scope"] in scopes for r in self.results)
            if state["observed_loss_curves"] != state["expected_loss_curves"]:
                self.pending.append({"path": stage, "reason": f"expected {state['expected_loss_curves']} curves, observed {state['observed_loss_curves']}"})
        return {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "audit_source_sha256": self.sha(Path(__file__)), "evidence": str(self.evidence),
                "status": "failed" if self.errors or adverse else "incomplete" if self.pending else "passed",
                "integrity_and_finiteness": "failed" if self.errors else "passed",
                "strict_loss_monotonicity": "failed" if adverse else "passed" if fully_evaluable else "not_evaluable",
                "increase_allowance": 0, "errors": self.errors, "pending": self.pending,
                "summary": groups, "nonmonotonic_runs": adverse, "results": self.results,
                "registered_stages": self.stage_status, "profile_jobs": self.profile_jobs,
                "historical_instrumentation_failures": self.historical_failures,
                "profile_retry_provenance": self.retry_provenance,
                "execution_history": "contains preserved failed profiling attempt",
                "expected_registered_loss_curves": sum(s["expected_loss_curves"] for s in self.stage_status.values() if not s.get("historical_failed_attempt")),
                "expected_historical_profile_loss_curves": 5,
                "observed_loss_curves_including_untracked_diagnostics": len(self.results),
                "identities": self.identities,
                "limits": ["Losses are the saved GPU-computed observations, not independently recomputed from per-round predictions.",
                           "A monotonicity failure is retained, although actual-loss descent is not an optimizer guarantee.",
                           "No unchanged quality evaluator, configuration selection or prior evidence is modified.",
                           "Hash agreement verifies file identities; it does not establish independent reproducible compilation.",
                           "Profile loss curves are audited separately; profiler wall/kernel durations are never used as uninstrumented speed evidence.",
                           "The original profiler attempt remains failed. Successful audit of its declared failure and a separate retry does not turn it into a quality success.",
                           "Incomplete planned jobs are not silently treated as successful."]}


def self_test():
    base = {"rounds": 2, "optimization_order": 4, "training_loss": [1.0, 0.5, 0.5]}
    assert inspect_loss(base)["strict_monotonicity"] == "passed"
    assert inspect_loss(base)["equal_rounds"] == [2]
    bad = inspect_loss({**base, "training_loss": [1.0, 0.5, math.nextafter(0.5, math.inf)]})
    assert bad["strict_monotonicity"] == "failed" and bad["increases"][0]["round"] == 2
    for value in (float("nan"), float("inf"), -float("inf"), True, None, "0.5"):
        assert inspect_loss({**base, "training_loss": [1.0, value, 0.5]})["issues"]
    assert inspect_loss({**base, "training_loss": [1.0]})["issues"]
    assert inspect_loss({**base, "training_loss_final": 0.6})["issues"]
    assert inspect_loss({**base, "training_loss": []})["issues"]
    assert inspect_loss({**base, "training_loss": [1.0, 0.5, -0.01]})["issues"]
    json.dumps(safe({"losses": [float("nan"), float("inf")]}), allow_nan=False)
    kernel = {"ID": "0", "Kernel Name": "ghb::derivatives<3>()"}
    assert inspect_kernel([kernel], r".*::derivatives<3>\(\)")["passed"]
    assert not inspect_kernel([kernel], r".*::derivatives<4>\(\)")["passed"]
    assert not inspect_kernel([kernel, kernel], r".*")["passed"]
    assert not inspect_kernel([{"ID": "non-kernel"}], r".*")["passed"]
    old_job = {"name": "root", "command": ["ncu", "--kernel-name", "old-filter", "--export", "/old/root/profile", "bench", "--rounds", "1"]}
    new_job = {"name": "root", "command": ["ncu", "--kernel-name", "new-filter", "--export", "/new/root/profile", "bench", "--rounds", "1"]}
    assert profile_settings(old_job, Path("/old")) == profile_settings(new_job, Path("/new"))
    changed_job = {**new_job, "command": [*new_job["command"][:-1], "2"]}
    assert profile_settings(old_job, Path("/old")) != profile_settings(changed_job, Path("/new"))

    class MemoryAudit(Audit):
        def __init__(self, documents):
            super().__init__(Path("/in-memory-evidence"), Path("/in-memory-workspace"))
            self.documents = documents
            self.frozen_binaries = {"ghb_real_bench": "binary-hash"}

        def required(self, path):
            if path.name not in self.documents:
                self.pending.append({"path": str(path), "reason": "missing in-memory fixture"})
                return None
            return self.documents[path.name]

        def verify(self, path, expected, category):
            self.identities.append({"path": str(path), "expected": expected, "category": category})

    identity = {"binary_sha256": "binary-hash", "runner_sha256": "runner-hash",
                "profiler_sha256": {"/ncu": "tool-hash"}}
    receipt = {"command": ["/ncu", "profile"], "returncode": 0, "binary_sha256": "binary-hash",
               "runner_sha256": "runner-hash", "executable_sha256": "tool-hash", "binary_unchanged": True,
               "runner_unchanged": True, "executable_unchanged": True, "stdout_sha256": "out", "stderr_sha256": "err"}
    receipt_path = Path("/in-memory-evidence/profile-command.json")
    good = MemoryAudit({receipt_path.name: receipt})
    good.profile_receipt(receipt_path, identity, ["/ncu", "profile"])
    assert not good.errors and not good.pending
    for changes in ({"returncode": 1}, {"returncode": None, "execution_error": "interrupted"},
                    {"binary_sha256": "wrong"}, {"executable_unchanged": False}, {"runner_unchanged": False}):
        failed = MemoryAudit({receipt_path.name: {**receipt, **changes}})
        failed.profile_receipt(receipt_path, identity, ["/ncu", "profile"])
        assert failed.errors
    pending = MemoryAudit({receipt_path.name: {k: v for k, v in receipt.items() if k != "returncode"}})
    pending.profile_receipt(receipt_path, identity)
    assert pending.pending and not pending.errors
    stage = MemoryAudit({"summary.json": {"jobs": 12, "passed": 11, "failed": ["case"]}})
    stage.runner_summary(Path("summary.json"), 12)
    assert stage.errors
    print("audit loss/kernel/receipt self-tests passed (CPU only; no evidence read or modified)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, help="exclusively create a new audit JSON; default stdout")
    parser.add_argument("--self-test", action="store_true", help="exercise in-memory reference checks only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = Audit(args.evidence, args.workspace).run()
    encoded = json.dumps(safe(result), indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
        print(json.dumps({k: result[k] for k in ("status", "integrity_and_finiteness", "strict_loss_monotonicity", "summary")}, indent=2))
    else:
        sys.stdout.write(encoded)
    return 1 if result["status"] == "failed" else 2 if result["status"] == "incomplete" else 0


if __name__ == "__main__":
    sys.exit(main())
