#!/usr/bin/env python3
"""CPU-only rejection tests for matched synchronization evidence and schedule."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sync_preservation as runner


class SyncEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.exe = self.output / "never-executed-fixture"
        self.exe.write_text("fixture, not a GPU executable\n")
        self.job = copy.deepcopy(next(job for job in runner.schedule() if job["sync_mode"] == "quartet"))
        self.job["command"] = runner.command_for(str(self.exe), self.job)
        self.manifest = {"binary": runner.file_record(self.exe), "protocol": runner.PROTOCOL,
                         "cases": runner.legacy.CASES, "jobs": [self.job]}
        runner.write_new(self.output / "manifest.json", self.manifest)
        self.gpus = [{"driver_version": "fixture", "name": "Fixture GPU", "uuid": "GPU-fixture"}]
        runner.write_new(self.output / "environment.json", {
            "schema": 1, "binary": str(self.exe), "binary_sha256": runner.legacy.sha256(self.exe),
            "session_started_utc": "2026-09-23T00:00:00+00:00", "uname": {"system": "fixture"}, "gpus": self.gpus})
        self.rows = []
        slot_a, slot_b = self.job["comparison"].split("-")
        for quartet, pattern in enumerate(self.job["patterns"]):
            for position, slot in enumerate(pattern):
                self.rows.append({"case": self.job["case"], "comparison": self.job["comparison"],
                    "slot_a": slot_a, "slot_b": slot_b, "data_seed": self.job["data_seed"],
                    "order_seed": self.job["order_seed"], "quartet": quartet, "pattern": pattern,
                    "position": position, "slot": slot, "backend": slot_a if slot == "A" else slot_b,
                    **runner.legacy.CASES[self.job["case"]], "batch": self.job["batch"], "warmup_ms": 200,
                    "event_us": 20.0 if slot == "B" else 10.0, "submit_us": 2.0, "total_host_us": "",
                    "gpu": "Fixture GPU", "gpu_uuid": "GPU-fixture", "driver": 13020, "runtime": 13040,
                    "old_launch_address": "0x1000", "new_launch_address": "0x2000",
                    "context_address": "0x3000", "stream_address": "0x4000",
                    "sync_mode": "quartet", "snapshot_storage": "pinned_four_snapshots",
                    "timing_pair": (0 if slot == "A" else 2) + pattern[:position].count(slot),
                    "quartet_host_us": 100000.0, "position_host_scope": "unobserved"})
        self.stem = self.output / "measurements" / self.job["stem"]
        self.stem.parent.mkdir()
        self.metadata = {"command": self.job["command"], "seconds": 1.0, "exit_code": 0,
                         "executable": str(self.exe), "executable_sha256": runner.legacy.sha256(self.exe),
                         "binary": str(self.exe), "binary_sha256": runner.legacy.sha256(self.exe),
                         "executables_unchanged": True, "gpus_before": self.gpus, "gpus_after": self.gpus,
                         "telemetry_before": "fixture", "telemetry_after": "fixture"}
        self.save()

    def save(self):
        with self.stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=runner.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.stem.with_suffix(".log").write_text("fixture\n")
        self.stem.with_suffix(".command.json").write_text(json.dumps(self.metadata))
        self.stem.with_suffix(".receipt.json").write_text(json.dumps({
            "schema": 1, "index": self.job["index"], "manifest": runner.file_record(self.output / "manifest.json"),
            "environment": runner.file_record(self.output / "environment.json"),
            "artifacts": [runner.file_record(self.stem.with_suffix(suffix)) for suffix in (".csv", ".log", ".command.json")]}))

    def validate(self):
        return runner.validate_job(self.output, self.manifest, self.job)

    def test_schedule_is_complete_and_counterbalances_adjacent_modes(self):
        jobs = runner.schedule()
        self.assertEqual(jobs, runner.schedule())
        self.assertEqual(len(jobs), 64)
        self.assertEqual(len({job["stem"] for job in jobs}), 64)
        strata = Counter(tuple(job[field] for field in runner.GROUP_FIELDS) for job in jobs)
        self.assertEqual(len(strata), 32)
        self.assertEqual(set(strata.values()), {2})
        orders = defaultdict(list)
        for first, second in zip(jobs[::2], jobs[1::2]):
            key = tuple(first[field] for field in ("case", "comparison", "batch"))
            self.assertEqual(key, tuple(second[field] for field in ("case", "comparison", "batch")))
            self.assertEqual(first["repetition"], second["repetition"])
            self.assertEqual(first["patterns"], second["patterns"])
            self.assertEqual({first["sync_mode"], second["sync_mode"]}, set(runner.MODES))
            orders[key].append(first["sync_mode"])
        self.assertTrue(all(set(modes) == set(runner.MODES) for modes in orders.values()))

    def test_quartet_mode_retains_cluster_count_without_fabricated_position_completion(self):
        checked = self.validate()
        self.assertTrue(all(row["total_host_us"] is None for row in checked["rows"]))
        result = runner.summarize_job(self.job, checked["rows"])
        self.assertEqual(len(result["clusters"]), 32)
        self.assertEqual(result["quartet_host_us"]["count"], 32)
        self.assertNotIn("total_host_us", result["position_durations"])
        self.assertAlmostEqual(result["event_aa_ratios"]["geometric_mean"], 2.0)

    def test_position_mode_records_its_own_scope(self):
        self.job["sync_mode"] = "position"
        self.job["command"] = runner.command_for(str(self.exe), self.job)
        self.metadata["command"] = self.job["command"]
        for row in self.rows:
            row.update(sync_mode="position", total_host_us=25.0,
                       position_host_scope="timed_enqueue_through_snapshot_completion_per_operation")
        self.save()
        result = runner.summarize_job(self.job, self.validate()["rows"])
        self.assertEqual(result["position_durations"]["total_host_us"]["count"], 128)

    def test_reject_fabricated_quartet_position_completion(self):
        self.rows[0]["total_host_us"] = 10.0
        self.save()
        with self.assertRaisesRegex(ValueError, "unobserved"):
            self.validate()

    def test_reject_timing_pair_reuse(self):
        self.rows[1]["timing_pair"] = self.rows[0]["timing_pair"]
        self.save()
        with self.assertRaisesRegex(ValueError, "unique timing pair"):
            self.validate()

    def test_reject_changed_quartet_interval(self):
        self.rows[1]["quartet_host_us"] += 1.0
        self.save()
        with self.assertRaisesRegex(ValueError, "identical across"):
            self.validate()

    def test_reject_too_short_quartet_interval(self):
        for row in self.rows[:4]:
            row["quartet_host_us"] = 1.0
        self.save()
        with self.assertRaisesRegex(ValueError, "shorter than enclosed"):
            self.validate()

    def test_reject_missing_position(self):
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "expected 128"):
            self.validate()

    def test_reject_nonfinite_or_nonpositive_duration(self):
        for field in ("event_us", "submit_us", "quartet_host_us"):
            original = self.rows[0][field]
            for value in (float("nan"), float("inf"), 0.0, -1.0):
                with self.subTest(field=field, value=value):
                    self.rows[0][field] = value
                    self.save()
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        self.validate()
            self.rows[0][field] = original

    def test_reject_wrong_workload(self):
        self.rows[0]["batch"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "workload/protocol"):
            self.validate()

    def test_reject_pageable_snapshot_schema(self):
        self.rows[0]["snapshot_storage"] = "pageable"
        self.save()
        with self.assertRaisesRegex(ValueError, "workload/protocol"):
            self.validate()

    def test_reject_context_change(self):
        self.rows[-1]["context_address"] = "0x9000"
        self.save()
        with self.assertRaisesRegex(ValueError, "address changed"):
            self.validate()

    def test_reject_backend_alias(self):
        self.rows[0]["old_launch_address"] = self.rows[0]["new_launch_address"]
        self.save()
        with self.assertRaisesRegex(ValueError, "same address"):
            self.validate()

    def test_reject_changed_csv_hash(self):
        with self.stem.with_suffix(".csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            self.validate()

    def test_reject_changed_command(self):
        self.metadata["command"] = self.metadata["command"] + ["--unexpected"]
        self.save()
        with self.assertRaisesRegex(ValueError, "exact command"):
            self.validate()

    def test_reject_changed_gpu_identity(self):
        self.metadata["gpus_after"] = [{**self.gpus[0], "uuid": "GPU-wrong"}]
        self.save()
        with self.assertRaisesRegex(ValueError, "GPU/driver identity"):
            self.validate()

    def test_reject_wrong_csv_gpu(self):
        for row in self.rows:
            row["gpu_uuid"] = "GPU-wrong"
        self.save()
        with self.assertRaisesRegex(ValueError, "CSV GPU differs"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
