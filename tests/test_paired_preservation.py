#!/usr/bin/env python3
"""CPU-only rejection tests for the paired preservation evidence format."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_audit_test", ROOT / "tools/analyze_paired_preservation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
runner = audit.runner


class PairedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.exe = self.output / "never-executed-synthetic-binary"
        self.exe.write_text("synthetic fixture; never executed\n")
        self.job = copy.deepcopy(next(job for job in runner.schedule() if job["comparison"] == "new-old"))
        self.job["command"] = runner.command_for(str(self.exe), self.job)
        self.manifest = {"binary": runner.file_record(self.exe), "protocol": runner.PROTOCOL,
                         "cases": runner.CASES, "jobs": [self.job]}
        runner.write_new(self.output / "manifest.json", self.manifest)
        self.gpus = [{"driver_version": "fixture", "name": "Fixture GPU", "uuid": "GPU-fixture"}]
        runner.write_new(self.output / "environment.json", {
            "schema": 1, "binary": str(self.exe), "binary_sha256": runner.sha256(self.exe),
            "session_started_utc": "2026-09-22T00:00:00+00:00", "uname": {"system": "fixture"}, "gpus": self.gpus})
        self.rows = []
        slot_a, slot_b = self.job["comparison"].split("-")
        for quartet, pattern in enumerate(self.job["patterns"]):
            for position, slot in enumerate(pattern):
                backend = slot_a if slot == "A" else slot_b
                # Reversed mapping: old/new = 1/2, reported new/old must be 2.
                self.rows.append({"case": self.job["case"], "comparison": self.job["comparison"],
                    "slot_a": slot_a, "slot_b": slot_b, "data_seed": self.job["data_seed"],
                    "order_seed": self.job["order_seed"], "quartet": quartet, "pattern": pattern,
                    "position": position, "slot": slot, "backend": backend,
                    **runner.CASES[self.job["case"]], "batch": 32, "warmup_ms": 200,
                    "event_us": 20.0 if backend == "new" else 10.0,
                    "submit_us": 4.0 if backend == "new" else 2.0,
                    "total_host_us": 40.0 if backend == "new" else 20.0,
                    "gpu": "Fixture GPU", "gpu_uuid": "GPU-fixture", "driver": 13020, "runtime": 13040,
                    "old_launch_address": "0x1000", "new_launch_address": "0x2000",
                    "context_address": "0x3000", "stream_address": "0x4000"})
        self.stem = self.output / "measurements" / self.job["stem"]
        self.stem.parent.mkdir()
        self.metadata = {"command": self.job["command"], "seconds": 1.0, "exit_code": 0,
                         "executable": str(self.exe), "executable_sha256": runner.sha256(self.exe),
                         "binary": str(self.exe), "binary_sha256": runner.sha256(self.exe),
                         "executables_unchanged": True, "gpus_before": self.gpus, "gpus_after": self.gpus,
                         "telemetry_before": "fixture telemetry", "telemetry_after": "fixture telemetry"}
        self.save()

    def save(self):
        with self.stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=audit.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.stem.with_suffix(".log").write_text("synthetic fixture\n")
        self.stem.with_suffix(".command.json").write_text(json.dumps(self.metadata))
        self.stem.with_suffix(".receipt.json").write_text(json.dumps({
            "schema": 1, "index": self.job["index"],
            "manifest": runner.file_record(self.output / "manifest.json"),
            "environment": runner.file_record(self.output / "environment.json"),
            "artifacts": [runner.file_record(self.stem.with_suffix(suffix)) for suffix in (".csv", ".log", ".command.json")]}))

    def validate(self):
        return audit.validate_job(self.output, self.manifest, self.job)

    def test_accept_and_normalize_reversed_slots(self):
        checked = self.validate()
        result = audit.summarize_job(self.job, checked["rows"])
        self.assertEqual(len(result["clusters"]), 32)
        for metric in audit.METRICS:
            self.assertAlmostEqual(result["metrics"][metric]["quartet_ratios"]["geometric_mean"], 2.0)

    def test_schedule_contains_every_declared_invocation(self):
        jobs = runner.schedule()
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({job["stem"] for job in jobs}), 24)
        self.assertEqual(jobs, runner.schedule())
        for job in jobs:
            self.assertEqual(job["patterns"].count("ABBA"), 16)
            self.assertEqual(job["patterns"].count("BAAB"), 16)

    def test_detect_missing_position(self):
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "expected 128"):
            self.validate()

    def test_detect_reordered_position(self):
        self.rows[0], self.rows[1] = self.rows[1], self.rows[0]
        self.save()
        with self.assertRaisesRegex(ValueError, "balanced execution order"):
            self.validate()

    def test_detect_wrong_workload(self):
        self.rows[0]["batch"] = 31
        self.save()
        with self.assertRaisesRegex(ValueError, "workload/protocol"):
            self.validate()

    def test_detect_infinite_or_nonpositive_timing(self):
        for value in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(value=value):
                self.rows[0]["event_us"] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    self.validate()

    def test_detect_context_change(self):
        self.rows[-1]["context_address"] = "0x9999"
        self.save()
        with self.assertRaisesRegex(ValueError, "address changed"):
            self.validate()

    def test_detect_alias_backend_addresses(self):
        self.rows[0]["new_launch_address"] = self.rows[0]["old_launch_address"]
        self.save()
        with self.assertRaisesRegex(ValueError, "same address"):
            self.validate()

    def test_detect_wrong_backend_binding(self):
        self.rows[0]["backend"] = "new" if self.rows[0]["backend"] == "old" else "old"
        self.save()
        with self.assertRaisesRegex(ValueError, "balanced execution order"):
            self.validate()

    def test_detect_changed_csv_hash(self):
        with self.stem.with_suffix(".csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            self.validate()

    def test_detect_changed_command(self):
        self.metadata["command"] = self.metadata["command"] + ["--unexpected"]
        self.save()
        with self.assertRaisesRegex(ValueError, "exact command"):
            self.validate()

    def test_detect_changed_environment(self):
        self.metadata["gpus_after"] = [{**self.gpus[0], "uuid": "GPU-wrong"}]
        self.save()
        with self.assertRaisesRegex(ValueError, "GPU/driver identity"):
            self.validate()

    def test_detect_wrong_csv_gpu(self):
        for row in self.rows:
            row["gpu_uuid"] = "GPU-wrong"
        self.save()
        with self.assertRaisesRegex(ValueError, "CSV GPU differs"):
            self.validate()

    def test_aa_bindings_use_same_backend_but_distinct_global_addresses(self):
        self.job["comparison"] = "old-old"
        self.job["command"] = runner.command_for(str(self.exe), self.job)
        self.metadata["command"] = self.job["command"]
        for row in self.rows:
            row.update(comparison="old-old", slot_a="old", slot_b="old", backend="old", event_us=10.0)
        self.save()
        result = audit.summarize_job(self.job, self.validate()["rows"])
        self.assertAlmostEqual(result["metrics"]["event_us"]["quartet_ratios"]["geometric_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
