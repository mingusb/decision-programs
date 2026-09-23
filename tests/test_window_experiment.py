#!/usr/bin/env python3
"""CPU-only tests of experiment identity and raw-result validation."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("window_runner", ROOT / "tools/run_window_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.csv"
        self.job = dict(case=runner.cases("window")[1], seed=runner.SEEDS[0])
        self.rows = []
        for algorithm, scratch, window, duration in (("global", 4194304, 0, 20),
                                                    ("global_window", 2097152, 524288, 10)):
            self.rows.append(dict(self.job["case"]["workload"], seed=self.job["seed"],
                samples=11, batch=4, timing_protocol=3, algorithm=algorithm, tuning=0,
                threads=128, items=4, replicas=1, blocks=48, local_counter="u32", clear_policy="kernel",
                shared_limit=49152, load_policy="scalar", window_bins=window, scratch_bytes=scratch,
                sample_us=";".join([str(duration)] * 11), min_us=duration, median_us=duration,
                max_us=duration, p95_us=duration, input_gb_s=(1 << 24) * 4 / (duration * 1000),
                gpu="test", sm=86, driver_api=13040, runtime=13040, cub_version=303000, eviction_bytes=0))

    def parse(self, rows=None, job=None):
        values = self.rows if rows is None else rows
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
        return runner.parse(self.path, self.job if job is None else job)

    def test_ratio_roles_stable_if_rows_reordered(self):
        values, _ = self.parse(list(reversed(self.rows)))
        self.assertTrue(values[0]["variant"].startswith("global:"))
        self.assertEqual(values[0]["median_us"] / values[1]["median_us"], 2)

    def test_wrong_window_and_scratch_rejected(self):
        for field in ("window_bins", "scratch_bytes"):
            rows = copy.deepcopy(self.rows)
            rows[1][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError): self.parse(rows)

    def test_false_summary_missing_samples_and_nonfinite_rejected(self):
        for field, value in (("median_us", 3), ("sample_us", "10;10"),
                             ("sample_us", ";".join(["nan"] * 11))):
            rows = copy.deepcopy(self.rows); rows[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.parse(rows)

    def test_missing_duplicate_or_undeclared_candidates_rejected(self):
        for rows in ([self.rows[0]], [self.rows[0], self.rows[0]]):
            with self.assertRaises(ValueError): self.parse(rows)
        rows = copy.deepcopy(self.rows); rows[1]["blocks"] = 192
        with self.assertRaises(ValueError): self.parse(rows)

    def test_workload_and_policy_identity_rejected(self):
        for field in ("seed", "bins", "batch", "threads", "shared_limit", "eviction_bytes"):
            rows = copy.deepcopy(self.rows); rows[1][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError): self.parse(rows)

    def test_fixed_matrix(self):
        cases = runner.cases("window")
        self.assertEqual(len(cases), 10)
        self.assertEqual({c["window"] for c in cases}, {524288, 1048576})
        self.assertTrue(any(c["workload"]["n"] == 1 << 28 for c in cases))
        skew = runner.cases("skew")
        self.assertEqual(len(skew), 16)
        for case in skew:
            self.assertIn("warp:4:96:native:kernel", case["variants"])
            hot = int(case["workload"]["distribution"].split("@")[1])
            self.assertLess(hot, case["workload"]["bins"])


if __name__ == "__main__":
    unittest.main()
