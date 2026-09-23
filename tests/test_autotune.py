"""CPU-only policy and command-line tests; no CUDA device or binary is used."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


TUNER = Path(__file__).resolve().parents[1] / "tools" / "autotune.py"
SPEC = importlib.util.spec_from_file_location("histogram_autotune", TUNER)
AUTOTUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOTUNE)

# The fake executable follows the benchmark CLI/CSV protocol, while all timings
# and metadata come from each test's fixture. Subprocess execution exercises the
# actual parser, argument generation, evidence files, saved plans and replay.
FAKE_BENCHMARK = r'''
import csv
import json
from pathlib import Path
import sys

location = Path(__file__)
state = json.loads(location.with_suffix(".json").read_text())
with location.with_suffix(".calls").open("a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
arguments = iter(sys.argv[1:])
options = {}
for key in arguments:
    options[key[2:]] = True if key == "--sweep" else next(arguments)
if state.get("exit_status"):
    raise SystemExit(state["exit_status"])

def identity(config):
    return ":".join(str(config[key]) for key in ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))

catalog = {identity(config): config for config in state["catalog"]}
if options.get("sweep"):
    variants = list(catalog)
elif "variants" in options:
    variants = options["variants"].split(",")
else:
    variants = [":".join(options[key] for key in ("algorithm", "tuning", "blocks", "local-counter", "clear"))]
fields = "gpu,sm,driver_api,runtime,cub_version,n,bins,input,counter,distribution,order,cache,launch,eviction_bytes,seed,algorithm,tuning,threads,items,replicas,blocks,scratch_bytes,local_counter,samples,batch,median_us,min_us,p95_us,max_us,input_gb_s,sample_us,timing_protocol,warmup_ms,load_policy,shared_limit,clear_policy".split(",")
writer = csv.DictWriter(sys.stdout, fieldnames=fields)
writer.writeheader()
for key in variants:
    times = state["search"] if options.get("sweep") else state["validation"][options["seed"]]
    elapsed = times[key]
    row = {name: options[name] for name in ("n", "bins", "input", "counter", "distribution", "order", "cache", "launch", "seed", "samples", "batch")}
    row.update(catalog[key])
    row.update(timing_protocol=3, warmup_ms=options["warmup-ms"])
    row.update(gpu="Synthetic GPU", sm="86", driver_api="13040", runtime="13040", cub_version="300400", eviction_bytes=4194304 if row["cache"] == "cold" else 0)
    row.update(median_us=elapsed, min_us=elapsed, p95_us=elapsed, max_us=elapsed, input_gb_s=1.0, sample_us=";".join([str(elapsed)] * int(options["samples"])))
    if not options.get("sweep"):
        row.update(state.get("validation_overrides", {}))
    writer.writerow(row)
'''


def config(algorithm: str, local: str = "native") -> dict:
    baseline = algorithm in ("cub", "nvidia_sample256")
    return {"algorithm": algorithm, "tuning": 2, "blocks": 192,
            "threads": 0 if baseline else 256, "items": 0 if baseline else 8,
            "replicas": 0 if baseline else 1, "scratch_bytes": 0,
            "local_counter": local, "load_policy": "reference" if baseline else "scalar", "shared_limit": 0 if baseline else 49152, "clear_policy": "runtime"}


def identity(configuration: dict) -> str:
    return ":".join(str(configuration[key])
                    for key in ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


class AutotuneIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gh-autotune-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.executable = self.directory / "fake_benchmark.py"
        self.executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_BENCHMARK))
        self.executable.chmod(0o700)
        self.plan_path = self.directory / "plan.json"
        self.catalog = [config("cub"), config("nvidia_sample256"), config("shared")]
        self.state = {
            "catalog": self.catalog,
            "search": {identity(item): 10.0 for item in self.catalog},
            "validation": {str(seed): {identity(item): 10.0 for item in self.catalog}
                           for seed in (67890, 24680)},
        }

    def times(self, seed: int, **timings: float) -> None:
        for item in self.catalog:
            if item["algorithm"] in timings:
                self.state["validation"][str(seed)][identity(item)] = timings[item["algorithm"]]

    def test_identical_batch_preserved_through_search_validation_replay(self):
        plan = self.tune("--batch", "64", "--search-samples", "5", "--validation-samples", "13")
        self.assertEqual(plan["search"]["batch"], 64)
        self.assertEqual(plan["validation"]["batch"], 64)
        self.assertEqual(plan["search"]["samples"], 5)
        self.assertEqual(plan["validation"]["samples"], 13)
        self.assertEqual(plan["selection"]["reference_comparison"]["ties"], 2)
        self.invoke("--replay", str(self.plan_path))
        for call in self.calls():
            self.assertEqual(call[call.index("--batch") + 1], "64")
        self.assertEqual(self.calls()[-1][self.calls()[-1].index("--samples") + 1], "13")

    def write_state(self) -> None:
        self.executable.with_suffix(".json").write_text(json.dumps(self.state))

    def invoke(self, *arguments: str, success: bool = True) -> subprocess.CompletedProcess:
        self.write_state()
        command = [sys.executable, str(TUNER), "--exe", str(self.executable), *arguments]
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=15)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def tune(self, *arguments: str) -> dict:
        self.invoke("--output", str(self.plan_path), "--n", "1024", "--bins", "256",
                    "--input", "u8", *arguments)
        return json.loads(self.plan_path.read_text())

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.executable.with_suffix(".calls").read_text().splitlines()]

    def test_reference_losses_are_reported_without_selecting_a_reference(self) -> None:
        # CUB wins seed one, NVIDIA wins seed two. The custom candidate beats
        # CUB overall, but loses against the relevant NVIDIA measurement.
        self.times(67890, cub=8.0, nvidia_sample256=10.0, shared=7.0)
        self.times(24680, cub=12.0, nvidia_sample256=6.0, shared=7.0)
        plan = self.tune()
        self.assertEqual(plan["chosen"]["algorithm"], "shared")
        custom = next(item for item in plan["finalists"] if item["config"]["algorithm"] == "shared")
        self.assertEqual([item["reference"] for item in custom["per_seed"]], ["cub", "nvidia_sample256"])
        self.assertAlmostEqual(custom["per_seed"][1]["reference_speedup"], 6.0 / 7.0)
        self.assertTrue(plan["selection"]["custom_only"])
        self.assertFalse(plan["selection"]["retained_reference"])
        self.assertFalse(plan["selection"]["threshold_is_selection_gate"])
        self.assertEqual(plan["selection"]["references"], ["cub", "nvidia_sample256"])
        comparison = plan["selection"]["reference_comparison"]
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (1, 0, 1))
        self.assertEqual(comparison["seeds_meeting_margin"], 1)
        self.assertFalse(comparison["all_seeds_meet_margin"])

    def test_submargin_custom_wins_do_not_select_reference(self) -> None:
        self.times(67890, cub=12.0, nvidia_sample256=8.0, shared=7.8)
        self.times(24680, cub=13.0, nvidia_sample256=9.0, shared=8.8)
        plan = self.tune()
        self.assertEqual(plan["chosen"]["algorithm"], "shared")
        self.assertAlmostEqual(plan["holdout_reference_speedup"], ((8.0 / 7.8) + (9.0 / 8.8)) / 2)
        self.assertFalse(plan["selection"]["retained_cub"])
        self.assertFalse(plan["selection"]["retained_reference"])
        comparison = plan["selection"]["reference_comparison"]
        self.assertEqual(comparison["wins"], 2)
        self.assertEqual(comparison["seeds_meeting_margin"], 0)

    def test_faster_references_still_select_best_custom(self) -> None:
        self.catalog.append(config("shared_partial"))
        self.state["search"][identity(self.catalog[-1])] = 12.0
        self.times(67890, cub=8.0, nvidia_sample256=9.0, shared=10.0, shared_partial=12.0)
        self.times(24680, cub=9.0, nvidia_sample256=10.0, shared=14.0, shared_partial=10.0)
        plan = self.tune()
        self.assertEqual(plan["chosen"]["algorithm"], "shared_partial")
        self.assertAlmostEqual(plan["holdout_reference_speedup"], ((8.0 / 12.0) + (9.0 / 10.0)) / 2)
        self.assertFalse(plan["selection"]["retained_reference"])
        self.assertEqual(plan["selection"]["reference_comparison"]["losses"], 2)
        self.assertEqual({item["config"]["algorithm"] for item in plan["finalists"]},
                         {"cub", "nvidia_sample256", "shared", "shared_partial"})
        result = self.invoke("--replay", str(self.plan_path))
        self.assertEqual(next(csv.DictReader(io.StringIO(result.stdout)))["algorithm"], "shared_partial")

    def test_reference_margin_is_diagnostic_not_selection_gate(self) -> None:
        self.times(67890, cub=20.0, nvidia_sample256=8.0, shared=4.0)
        self.times(24680, cub=20.0, nvidia_sample256=8.0, shared=8.0 / 1.04)
        plan = self.tune()
        self.assertEqual(plan["chosen"]["algorithm"], "shared")
        self.assertEqual(plan["selection"]["minimum_speedup_over_reference"], 1.05)
        self.assertFalse(plan["selection"]["threshold_is_selection_gate"])
        self.assertEqual(plan["selection"]["reference_comparison"]["seeds_meeting_margin"], 1)
        self.assertFalse(plan["selection"]["reference_comparison"]["all_seeds_meet_margin"])

    def test_custom_ranking_does_not_exclude_a_reference_loss(self) -> None:
        self.catalog.append(config("shared_partial"))
        self.state["search"][identity(self.catalog[-1])] = 6.8
        self.times(67890, cub=8.0, nvidia_sample256=9.0, shared=4.0, shared_partial=6.8)
        self.times(24680, cub=8.0, nvidia_sample256=9.0, shared=10.0, shared_partial=6.8)
        plan = self.tune()
        # The consistent custom option passes the old margin gate on both seeds,
        # but the requested median reference-normalized ranking favors shared.
        self.assertEqual(plan["chosen"]["algorithm"], "shared")
        self.assertAlmostEqual(plan["holdout_reference_speedup"], 1.4)
        self.assertEqual(plan["selection"]["reference_comparison"]["losses"], 1)

    def test_reference_only_search_cannot_produce_plan(self) -> None:
        self.state["catalog"] = self.catalog[:2]
        result = self.invoke("--output", str(self.plan_path), success=False)
        self.assertIn("did not provide a custom candidate", result.stderr)
        self.assertFalse(self.plan_path.exists())
        self.assertEqual(len(self.calls()), 1)

    def test_reference_only_finalist_pool_is_rejected(self) -> None:
        references = [{"config": config(algorithm), "holdout_reference_speedup": 1.0}
                      for algorithm in ("cub", "nvidia_sample256")]
        for finalists in ([], references):
            with self.subTest(finalists=finalists):
                with self.assertRaisesRegex(AUTOTUNE.TuningError, "no custom finalist"):
                    AUTOTUNE.choose_custom(finalists)

    def test_custom_can_win_against_crossing_references(self) -> None:
        self.times(67890, cub=8.0, nvidia_sample256=10.0, shared=7.0)
        self.times(24680, cub=12.0, nvidia_sample256=6.0, shared=5.0)
        plan = self.tune("--cache", "cold", "--launch", "graph")
        self.assertEqual(plan["schema"], 5)
        self.assertEqual(plan["chosen"]["algorithm"], "shared")
        self.assertAlmostEqual(plan["holdout_reference_speedup"], ((8.0 / 7.0) + (6.0 / 5.0)) / 2)
        self.assertEqual(plan["workload"]["cache"], "cold")
        self.assertEqual(plan["workload"]["launch"], "graph")
        self.assertEqual(plan["environment"]["eviction_bytes"], 4194304)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertIn("--sweep", calls[0])
        self.assertTrue(all("--variants" in call for call in calls[1:]))
        with self.plan_path.with_suffix(".validation.csv").open() as data:
            self.assertEqual(len(list(csv.DictReader(data))), 6)
        result = self.invoke("--replay", str(self.plan_path))
        row = next(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual((row["cache"], row["launch"]), ("cold", "graph"))

    def test_u32_local_counters_remain_a_distinct_candidate(self) -> None:
        self.catalog = [config("cub"), config("shared"), config("shared", "u32")]
        self.state["catalog"] = self.catalog
        self.state["search"] = {identity(item): value for item, value in zip(self.catalog, (10.0, 8.0, 6.0))}
        self.state["validation"] = {
            str(seed): {identity(item): value for item, value in zip(self.catalog, (10.0, 8.0, 6.0))}
            for seed in (67890, 24680)
        }
        plan = self.tune("--counter", "u64")
        shared = [item for item in plan["finalists"] if item["config"]["algorithm"] == "shared"]
        self.assertEqual({item["config"]["local_counter"] for item in shared}, {"native", "u32"})
        self.assertEqual(plan["chosen"]["local_counter"], "u32")
        result = self.invoke("--replay", str(self.plan_path))
        self.assertEqual(next(csv.DictReader(io.StringIO(result.stdout)))["local_counter"], "u32")

    def test_narrowed_global_and_warp_candidates_select_and_replay(self) -> None:
        for algorithm in ("global", "warp"):
            with self.subTest(algorithm=algorithm):
                self.plan_path = self.directory / (algorithm + ".json")
                native = config(algorithm)
                narrow = {**config(algorithm, "u32"), "scratch_bytes": 256 * 4}
                self.catalog = [config("cub"), native, narrow]
                self.state["catalog"] = self.catalog
                self.state["search"] = {identity(item): value for item, value in zip(self.catalog, (10, 8, 6))}
                self.state["validation"] = {
                    str(seed): {identity(item): value for item, value in zip(self.catalog, (10, 8, 6))}
                    for seed in (67890, 24680)}
                plan = self.tune("--counter", "u64")
                finalists = [x["config"] for x in plan["finalists"] if x["config"]["algorithm"] == algorithm]
                self.assertEqual({x["local_counter"] for x in finalists}, {"native", "u32"})
                self.assertEqual(plan["chosen"], narrow)
                result = self.invoke("--replay", str(self.plan_path))
                row = next(csv.DictReader(io.StringIO(result.stdout)))
                self.assertEqual((row["algorithm"], row["local_counter"], int(row["scratch_bytes"])),
                                 (algorithm, "u32", 1024))

    def test_shortlist_retains_narrowed_path_after_four_native_variants(self) -> None:
        variants = [config("global") for _ in range(4)]
        for index, row in enumerate(variants):
            row["tuning"] = index
        narrow = {**config("global", "u32"), "scratch_bytes": 1024}
        self.catalog = [config("cub"), *variants, narrow]
        self.state["catalog"] = self.catalog
        self.state["search"] = {identity(row): value for row, value in zip(self.catalog, (10, 5, 5.1, 5.2, 5.3, 6))}
        self.state["validation"] = {
            str(seed): {identity(row): 4 if row["local_counter"] == "u32" else 10 for row in self.catalog}
            for seed in (67890, 24680)}
        plan = self.tune("--counter", "u64")
        self.assertEqual(plan["chosen"], narrow)
        self.assertEqual(len(plan["finalists"]), 6)

    def test_narrowed_global_csv_enforces_total_count_and_workspace(self) -> None:
        # Exercise the real CSV parser with a recorded-shaped row, without any
        # CUDA allocation or iteration over the declared (possibly huge) input.
        for algorithm in ("global", "warp"):
            row = dict(config(algorithm, "u32"), gpu="Synthetic GPU", sm="86",
                       driver_api="13040", runtime="13040", cub_version="300400",
                       n=(1 << 32) - 1, bins=256, input="u32", counter="u64",
                       distribution="uniform", order="shuffled", cache="warm", launch="graph",
                       eviction_bytes=0, seed=1, samples=3, batch=4, median_us=1, min_us=1,
                       p95_us=1, max_us=1, input_gb_s=1, sample_us="1;1;1", timing_protocol=3,
                       warmup_ms=200)
            row["scratch_bytes"] = 1024

            def parse(candidate):
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=list(candidate))
                writer.writeheader(); writer.writerow(candidate)
                return AUTOTUNE.parse_csv(output.getvalue())[1][0]

            for blocks in (1, 192, (1 << 31) - 1):
                with self.subTest(algorithm=algorithm, blocks=blocks):
                    safe = {**row, "blocks": blocks}
                    self.assertEqual(parse(safe)["n"], (1 << 32) - 1)
                    with self.assertRaisesRegex(AUTOTUNE.TuningError, "narrowed global counters"):
                        parse({**safe, "n": 1 << 32})
            self.assertEqual(parse({**row, "n": 0, "scratch_bytes": 0})["scratch_bytes"], 0)
            for change in ({"scratch_bytes": 1023}, {"scratch_bytes": 0}, {"counter": "u32"},
                           {"load_policy": "vector4"}, {"n": 0}):
                with self.subTest(algorithm=algorithm, change=change):
                    with self.assertRaisesRegex(AUTOTUNE.TuningError, "narrowed global counters"):
                        parse({**row, **change})

    def test_clear_policies_remain_distinct_and_replay_exactly(self) -> None:
        runtime = config("shared")
        kernel = {**runtime, "clear_policy": "kernel"}
        self.catalog = [config("cub"), runtime, kernel]
        self.state["catalog"] = self.catalog
        self.state["search"] = {identity(row): 10 for row in self.catalog}
        self.state["validation"] = {
            str(seed): {identity(self.catalog[0]): 10, identity(runtime): 8, identity(kernel): 7}
            for seed in (67890, 24680)}
        plan = self.tune("--launch", "graph")
        shared = [item for item in plan["finalists"] if item["config"]["algorithm"] == "shared"]
        self.assertEqual({item["config"]["clear_policy"] for item in shared}, {"runtime", "kernel"})
        self.assertEqual(plan["chosen"]["clear_policy"], "kernel")
        result = self.invoke("--replay", str(self.plan_path))
        self.assertEqual(next(csv.DictReader(io.StringIO(result.stdout)))["clear_policy"], "kernel")
        self.state["validation_overrides"] = {"clear_policy": "runtime"}
        self.invoke("--replay", str(self.plan_path), success=False)

    def test_replay_rejects_changed_binary_before_execution(self) -> None:
        self.tune()
        count = len(self.calls())
        with self.executable.open("a") as output:
            output.write("\n# changed executable\n")
        result = self.invoke("--replay", str(self.plan_path), success=False)
        self.assertIn("SHA256", result.stderr)
        self.assertEqual(len(self.calls()), count)

    def test_replay_rejects_schema4_custom_and_reference_plans(self) -> None:
        plan = self.tune()
        count = len(self.calls())
        for algorithm in ("shared", "cub", "nvidia_sample256"):
            with self.subTest(algorithm=algorithm):
                plan["schema"] = 4
                plan["chosen"] = config(algorithm)
                self.plan_path.write_text(json.dumps(plan))
                result = self.invoke("--replay", str(self.plan_path), success=False)
                self.assertIn("requires schema 5", result.stderr)
                self.assertEqual(len(self.calls()), count)

    def test_replay_rejects_references_in_tampered_schema5_plan(self) -> None:
        plan = self.tune()
        count = len(self.calls())
        for algorithm in ("cub", "nvidia_sample256"):
            with self.subTest(algorithm=algorithm):
                # Keep schema, binary hash, and claimed custom-only policy valid;
                # replay must enforce the actual selected algorithm independently.
                plan["chosen"] = config(algorithm)
                self.plan_path.write_text(json.dumps(plan))
                result = self.invoke("--replay", str(self.plan_path), success=False)
                self.assertIn("must select a custom histogram", result.stderr)
                self.assertEqual(len(self.calls()), count)

    def test_replay_rejects_changed_environment(self) -> None:
        self.tune()
        self.state["validation_overrides"] = {"gpu": "Different GPU"}
        result = self.invoke("--replay", str(self.plan_path), success=False)
        self.assertIn("metadata changed", result.stderr)

    def test_replay_rejects_changed_cache_mode(self) -> None:
        self.tune("--cache", "cold")
        self.state["validation_overrides"] = {"cache": "warm", "eviction_bytes": 0}
        result = self.invoke("--replay", str(self.plan_path), success=False)
        self.assertIn("workload metadata", result.stderr)

    def test_nonfinite_measurements_cannot_produce_plan(self) -> None:
        self.state["validation"]["67890"][identity(config("shared"))] = "nan"
        result = self.invoke("--output", str(self.plan_path), success=False)
        self.assertIn("nonfinite", result.stderr)
        self.assertFalse(self.plan_path.exists())

    def test_failed_benchmark_cannot_produce_plan(self) -> None:
        self.state["exit_status"] = 7
        result = self.invoke("--output", str(self.plan_path), success=False)
        self.assertIn("status 7", result.stderr)
        self.assertFalse(self.plan_path.exists())

    def test_shortlist_keeps_other_family_after_four_fast_variants(self) -> None:
        variants = [config("shared") for _ in range(4)]
        for index, item in enumerate(variants):
            item["tuning"] = index
        self.catalog = [config("cub"), *variants, config("shared_partial")]
        self.state["catalog"] = self.catalog
        self.state["search"] = {identity(item): value for item, value in zip(self.catalog, (10, 5, 5.1, 5.2, 5.3, 6))}
        self.state["validation"] = {str(seed): {identity(item): (7 if item["algorithm"] == "shared_partial" else 10)
                                              for item in self.catalog} for seed in (67890, 24680)}
        plan = self.tune()
        self.assertEqual(plan["chosen"]["algorithm"], "shared_partial")
        self.assertEqual(len(plan["finalists"]), 6)

    def test_warmup_and_timing_protocol_are_replay_metadata(self) -> None:
        plan = self.tune("--warmup-ms", "100", "--launch", "graph")
        self.assertEqual(plan["workload"]["warmup_ms"], 100)
        self.assertEqual(plan["environment"]["timing_protocol"], 3)
        self.invoke("--replay", str(self.plan_path))
        self.state["validation_overrides"] = {"warmup_ms": 0}
        result = self.invoke("--replay", str(self.plan_path), success=False)
        self.assertIn("workload metadata", result.stderr)

    def test_old_timing_protocol_cannot_produce_plan(self) -> None:
        self.state["validation_overrides"] = {"timing_protocol": 2}
        self.invoke("--output", str(self.plan_path), success=False)
        self.assertFalse(self.plan_path.exists())


if __name__ == "__main__":
    unittest.main()
