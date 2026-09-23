"""CPU-only evidence contract tests. Every subprocess is intercepted; no CUDA tools run."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("ghb_observe", Path(__file__).parents[1] / "tools" / "observe.py")
observe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observe)


def result_fixture(instrumented=True, scale=1.0):
    work = dict(rows=1024, features=1, bins=256, seed=42, input="u32", counter="u64", distribution="uniform",
                cache="warm", launch="graph", repetitions=3, batch=4, warmup_ms=20, window_bins=128)
    timing = {field: scale for field in observe.TIMINGS - {"operation_ms"}}
    timing.update(operation_ms=[scale, 2 * scale, 3 * scale], operation_median_ms=2 * scale,
                  device_span_ms=30 * scale, end_to_end_wall_ms=60 * scale)
    memory = dict(input_bytes=4096, output_bytes=2048, scratch_bytes=512, validation_snapshot_bytes=6144,
                  free_before_bytes=900_000, free_after_alloc_bytes=880_000)
    result = dict(schema=1, kind="ghb.instrumentation", benchmark="count_histogram_component",
                  implementation="global_window:2:96:u32:kernel", instrumented=instrumented, nvtx=False,
                  build={"id": "a" * 64}, environment=dict(gpu="Fake GPU", sm=86, driver_api=13000, runtime=13000,
                  total_memory_bytes=1_000_000), workload=work,
                  validation=dict(passed=True, checked_bins=256, checked_repetitions=3, mismatched_bins=0),
                  timing=timing, memory=memory, samples=[])
    if instrumented:
        for rep in range(3):
            context = dict(round=-1, depth=-1, output=-1, repetition=rep, active_nodes=0, features=1, bins=256,
                           stream_id=1, rows=1024, logical_read_bytes=4096 * 2 * 4, logical_write_bytes=2048 * 4,
                           scratch_bytes=512, operations=4)
            # The inner recorder interval intentionally differs from the outer
            # batch interval. Equating these would reject legitimate evidence.
            result["samples"].append(dict(id=rep, stage="histogram", timing="gpu", context=context,
                                         host_start_ns=rep * 100, host_end_ns=rep * 100 + 50,
                                         gpu_ms=(rep + 1) * 4 * scale - 0.01))
        context = dict(result["samples"][0]["context"], repetition=-1, operations=1,
                       logical_read_bytes=0, logical_write_bytes=0)
        result["samples"].append(dict(id=3, stage="evaluate", timing="host", context=context,
                                     host_start_ns=500, host_end_ns=600, gpu_ms=None))
    return result


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.exe = self.base / "fake executable"
        self.exe.write_bytes(b"CPU test placeholder, never executed\n")
        self.source = self.base / "input source.cpp"
        self.source.write_text("source version one\n")
        (self.base / "CMakeCache.txt").write_text("compiler provenance\n")
        self.gpu_output = "GPU-0000, Fake GPU, 590.0, 1000, 5000, 45, 60\n"
        self.telemetry_missing = False
        self.process_code = 0
        self.process_error = None
        self.process_stdout = None
        self.change_exe = False
        self.payload = result_fixture()
        self.calls = []
        self.profile_artifact = True
        self.ncu_extension = ".ncu-rep"
        self.ncu_both_formats = False
        self.mock_run = mock.patch.object(observe.subprocess, "run", side_effect=self.fake_run).start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(observe.shutil, "which", side_effect=lambda name: "/mock/" + name).start()

    def fake_run(self, command, **kwargs):
        self.assertIsInstance(command, list)
        self.assertNotIn("shell", kwargs)
        self.calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            if self.telemetry_missing:
                raise FileNotFoundError("nvidia-smi unavailable in CPU-only test")
            return subprocess.CompletedProcess(command, 0, self.gpu_output, "")
        if command[-1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "fake tool version 1.2\n", "")
        if self.change_exe:
            self.exe.write_text("changed during observed process\n")
        if self.process_error:
            raise self.process_error
        if command[0].startswith("/mock/") and self.profile_artifact:
            if command[0].endswith("nsys"):
                target = next(value.removeprefix("--output=") for value in command if value.startswith("--output=")) + ".nsys-rep"
            elif command[0].endswith("ncu"):
                target = command[command.index("--export") + 1] + self.ncu_extension
                if self.ncu_both_formats:
                    for extension in (".ncu-rep", ".ncu-repz"):
                        Path(command[command.index("--export") + 1] + extension).write_bytes(b"ambiguous diagnostic report\n")
            else:
                target = command[command.index("--log-file") + 1]
            Path(target).write_bytes(b"raw diagnostic data, not a ranking duration\n")
        stdout = self.process_stdout if self.process_stdout is not None else json.dumps(self.payload)
        return subprocess.CompletedProcess(command, self.process_code, stdout, "retained stderr\n")

    def capture(self, name="capture", arguments=None, tool=None):
        output = self.base / name
        manifest = observe.capture(self.exe, output, arguments or [], timeout=17, sources=[self.source], tool=tool)
        return output, manifest

    def reseal(self, output, manifest=None):
        # Deliberately rewrite evidence ONLY inside tests to exercise semantic
        # validation independently of the outer corruption checksum barrier.
        if manifest is None:
            manifest = observe.read_json(output / "manifest.json")
        manifest["artifacts"] = observe.artifact_records(output)
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "manifest.sha256").write_text(observe.sha256(output / "manifest.json") + "\n")

    def test_capture_audit_and_historical_provenance(self):
        output, manifest = self.capture(arguments=["--n", "1024"])
        self.assertEqual(manifest["status"], "passed")
        self.assertTrue(observe.audit(output)["passed"])
        self.assertEqual(observe.read_json(output / "command.json"), [str(self.exe), "--n", "1024"])
        self.assertEqual(len(manifest["sources"]), 1)
        self.assertEqual(len(manifest["build_files"]), 1)
        self.assertEqual(manifest["binary"]["sha256_before"], manifest["binary"]["sha256_after"])
        self.assertEqual((output / "stdout.txt").read_text(), json.dumps(self.payload))
        self.assertFalse((output / ".incomplete").exists())
        self.source.write_text("later source changes must not invalidate archived provenance\n")
        self.exe.unlink()
        self.assertTrue(observe.audit(output)["passed"])
        self.assertEqual((output / manifest["sources"][0]["snapshot"]).read_text(), "source version one\n")

    def test_optional_telemetry_failure_is_logged(self):
        self.telemetry_missing = True
        output, manifest = self.capture()
        self.assertEqual(manifest["status"], "passed")
        report = observe.audit(output)
        self.assertIn("unavailable", report["environment"]["gpu_before"]["error"])
        self.assertIsNone(report["environment"]["gpu_before"]["exit_code"])

    def test_nonzero_process_fails_and_preserves_outputs(self):
        self.process_code, self.process_stdout = 9, "partial output"
        output, manifest = self.capture()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["process"]["exit_code"], 9)
        self.assertEqual((output / "stdout.txt").read_text(), "partial output")
        self.assertEqual((output / "stderr.txt").read_text(), "retained stderr\n")
        with self.assertRaisesRegex(ValueError, "observation failed"):
            observe.audit(output)

    def test_timeout_preserves_partial_stdout_stderr_and_fails(self):
        self.process_error = subprocess.TimeoutExpired([str(self.exe)], 17, output=b"partial stdout", stderr=b"partial stderr")
        output, manifest = self.capture()
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(manifest["process"]["timed_out"])
        self.assertEqual((output / "stdout.txt").read_bytes(), b"partial stdout")
        self.assertEqual((output / "stderr.txt").read_bytes(), b"partial stderr")
        with self.assertRaises(ValueError):
            observe.audit(output)

    def test_missing_process_is_recorded(self):
        self.process_error = PermissionError("executable inaccessible")
        output, manifest = self.capture()
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("inaccessible", manifest["process"]["error"])
        self.assertFalse(manifest["process"]["timed_out"])

    def test_overwrite_is_rejected_before_subprocess(self):
        output, _ = self.capture()
        old_manifest = (output / "manifest.json").read_bytes()
        call_count = len(self.calls)
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertEqual(len(self.calls), call_count)
        self.assertEqual((output / "manifest.json").read_bytes(), old_manifest)

    def test_changed_executable_fails_with_original_snapshot(self):
        original_hash = observe.sha256(self.exe)
        self.change_exe = True
        output, manifest = self.capture()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["binary"]["sha256_before"], original_hash)
        self.assertNotEqual(manifest["binary"]["sha256_after"], original_hash)
        self.assertEqual(observe.sha256(output / "provenance/executable"), original_hash)
        with self.assertRaises(ValueError):
            observe.audit(output)

    def test_manifest_corruption_is_detected(self):
        output, _ = self.capture()
        with (output / "manifest.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            observe.audit(output)

    def test_artifact_corruption_extra_file_and_symlink_are_rejected(self):
        for index, mutation in enumerate(("corrupt", "extra", "symlink")):
            with self.subTest(mutation=mutation):
                output, _ = self.capture(str(index))
                if mutation == "corrupt":
                    (output / "stderr.txt").write_text("changed")
                elif mutation == "extra":
                    (output / "unrecorded.txt").write_text("extra")
                else:
                    (output / "unrecorded-link").symlink_to(self.source)
                with self.assertRaises(ValueError):
                    observe.audit(output)

    def test_duplicate_artifact_and_provenance_records_rejected(self):
        for index, target in enumerate(("artifacts", "sources")):
            with self.subTest(target=target):
                output, manifest = self.capture(str(index))
                manifest[target].append(copy.deepcopy(manifest[target][0]))
                # Preserve intentional duplicate artifact rows.
                (output / "manifest.json").write_text(json.dumps(manifest))
                (output / "manifest.sha256").write_text(observe.sha256(output / "manifest.json") + "\n")
                with self.assertRaises(ValueError):
                    observe.audit(output)

    def test_nested_manifest_named_artifact_is_not_skipped(self):
        output, _ = self.capture()
        nested = output / "raw/manifest.json"
        nested.parent.mkdir()
        nested.write_text("nested diagnostic artifact")
        self.reseal(output)
        self.assertTrue(observe.audit(output)["passed"])

    def test_malformed_success_stdout_never_becomes_passed(self):
        payloads = ["not JSON", '{"schema":1,"schema":1}', '{"x":NaN}', '{"x":Infinity}', "[]", "null"]
        for index, text in enumerate(payloads):
            with self.subTest(text=text):
                self.process_stdout = text
                output, manifest = self.capture(str(index))
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual((output / "stdout.txt").read_text(), text)
                with self.assertRaises(ValueError):
                    observe.audit(output)

    def test_schema_semantic_corruption_rejected_after_rehash(self):
        mutations = [lambda r: r["validation"].update(checked_repetitions=2),
                     lambda r: r["validation"].update(checked_bins=255),
                     lambda r: r["validation"].update(passed=False),
                     lambda r: r["validation"].update(mismatched_bins=1),
                     lambda r: r["timing"].update(operation_median_ms=999),
                     lambda r: r["samples"][0].update(id=r["samples"][1]["id"])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                output, _ = self.capture(str(index))
                result = copy.deepcopy(self.payload)
                mutate(result)
                (output / "result.json").write_text(json.dumps(result))
                (output / "stdout.txt").write_text(json.dumps(result))
                self.reseal(output)
                with self.assertRaises(ValueError):
                    observe.audit(output)

    def test_raw_and_parsed_result_must_match(self):
        output, _ = self.capture()
        result = copy.deepcopy(self.payload)
        result["build"]["id"] = "b" * 64
        (output / "result.json").write_text(json.dumps(result))
        self.reseal(output)
        with self.assertRaisesRegex(ValueError, "raw executable stdout"):
            observe.audit(output)

    def test_nested_environment_and_process_schema_is_checked(self):
        for index, target in enumerate(("platform", "gpu_before", "performance_environment", "schema", "process")):
            with self.subTest(target=target):
                output, manifest = self.capture(str(index))
                if target == "process":
                    manifest["process"]["exit_code"] = False
                else:
                    env = observe.read_json(output / "environment.json")
                    if target == "platform":
                        env[target]["extra"] = 1
                    elif target == "schema":
                        env[target] = True
                    elif target == "gpu_before":
                        env[target]["wall_seconds"] = -1
                    else:
                        env[target]["CUDA_VISIBLE_DEVICES"] = 0
                    (output / "environment.json").write_text(json.dumps(env))
                self.reseal(output, manifest)
                with self.assertRaises(ValueError):
                    observe.audit(output)

    def test_atomic_publication_failure_does_not_leave_partial_target(self):
        target = self.base / "atomic.json"
        with mock.patch.object(observe.os, "link", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                observe.atomic_bytes(target, b"new artifact")
        self.assertFalse(target.exists())
        self.assertEqual(list(self.base.glob(".atomic.json.tmp-*")), [])
        observe.atomic_bytes(target, b"preserved")
        with self.assertRaises(ValueError):
            observe.atomic_bytes(target, b"replacement")
        self.assertEqual(target.read_bytes(), b"preserved")

    def test_partial_finalization_remains_unauditable(self):
        real_write = observe.write_json
        def failing_write(path, value):
            if path.name == "manifest.json":
                raise OSError("simulated full disk")
            return real_write(path, value)
        with mock.patch.object(observe, "write_json", side_effect=failing_write):
            with self.assertRaises(OSError):
                self.capture()
        output = self.base / "capture"
        self.assertTrue((output / ".incomplete").exists())
        self.assertTrue((output / "stdout.txt").exists())
        self.assertFalse((output / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "not finalized"):
            observe.audit(output)

    def test_profile_preserves_raw_artifacts_without_parsing_stdout(self):
        self.process_stdout = "profiler banner, not JSON; 0.000001 ms is not a ranking claim"
        for tool in observe.PROFILER:
            with self.subTest(tool=tool):
                output, manifest = self.capture(tool, arguments=["--instrumentation", "nvtx"], tool=tool)
                self.assertEqual(manifest["status"], "passed")
                self.assertEqual(manifest["measurement_role"], "diagnostic_only")
                self.assertIsNone(manifest["result"])
                audited = observe.audit(output)
                self.assertIsNone(audited["result"])
                self.assertTrue(any(item["path"].startswith("profile.") for item in manifest["artifacts"]))
                self.assertEqual(observe.read_json(output / "tool-version.json")["stdout"], "fake tool version 1.2\n")
                with self.assertRaisesRegex(ValueError, "diagnostic"):
                    observe.compare(output, output)

    def test_profile_evidence_can_be_moved_after_capture(self):
        output, _ = self.capture(tool="ncu")
        relocated = self.base / "relocated"
        shutil.move(output, relocated)
        self.assertTrue(observe.audit(relocated)["passed"])

    def test_ncu_accepts_each_native_report_format(self):
        for extension in (".ncu-rep", ".ncu-repz"):
            with self.subTest(extension=extension):
                self.ncu_extension = extension
                output, manifest = self.capture(extension, tool="ncu")
                self.assertEqual(manifest["status"], "passed")
                self.assertTrue(observe.audit(output)["passed"])
                reports = [record["path"] for record in manifest["artifacts"] if record["path"].startswith("profile.")]
                self.assertEqual(reports, ["profile" + extension])

    def test_ncu_rejects_ambiguous_reports_at_capture_and_audit(self):
        self.ncu_both_formats = True
        output, manifest = self.capture("ambiguous", tool="ncu")
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("exactly one raw artifact", manifest["error"])
        self.ncu_both_formats = False
        output, _ = self.capture("unambiguous", tool="ncu")
        (output / "profile.ncu-repz").write_bytes(b"second raw format")
        self.reseal(output)
        with self.assertRaisesRegex(ValueError, "exactly one raw artifact"):
            observe.audit(output)

    def test_ncu_rejects_empty_report(self):
        for extension in (".ncu-rep", ".ncu-repz"):
            with self.subTest(extension=extension):
                self.ncu_extension = extension
                output, _ = self.capture(extension, tool="ncu")
                (output / ("profile" + extension)).write_bytes(b"")
                self.reseal(output)
                with self.assertRaisesRegex(ValueError, "nonempty regular raw artifact"):
                    observe.audit(output)

    def test_failed_profiler_run_is_not_passed(self):
        self.process_code = 99
        output, manifest = self.capture(tool="memcheck")
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue((output / "profile.log").exists())
        with self.assertRaises(ValueError):
            observe.audit(output)

    def test_profiler_success_without_raw_artifact_fails(self):
        self.profile_artifact = False
        output, manifest = self.capture(tool="ncu")
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("raw artifact", manifest["error"])
        command = observe.read_json(output / "command.json")
        self.assertEqual(command[command.index("--launch-count") + 1], "2")
        self.assertEqual(command[command.index("--clock-control") + 1], "none")
        with self.assertRaises(ValueError):
            observe.audit(output)

    def test_compare_reports_overhead_in_both_directions(self):
        self.payload = result_fixture(False, 1.0)
        baseline, _ = self.capture("baseline", ["--instrumentation", "off", "--batch", "4"])
        self.payload = result_fixture(True, 1.1)
        measured, _ = self.capture("measured", ["--instrumentation", "timing", "--batch", "4"])
        report = observe.compare(baseline, measured)
        self.assertEqual(report["comparison_kind"], "instrumentation_overhead")
        self.assertAlmostEqual(report["right_over_left"]["operation_median_ms"], 1.1)
        self.assertEqual(report["raw_operation_ms"]["left"], [1, 2, 3])
        self.assertAlmostEqual(report["instrumented_over_uninstrumented_operation_ratio"], 1.1)
        reverse = observe.compare(measured, baseline)
        self.assertAlmostEqual(reverse["right_over_left"]["operation_median_ms"], 1 / 1.1)
        self.assertAlmostEqual(reverse["instrumented_over_uninstrumented_operation_ratio"], 1.1)
        self.assertTrue(any("zero cost" in value for value in report["limitations"]))

    def test_compare_workload_and_environment_mismatch_rejected(self):
        self.payload = result_fixture(False)
        baseline, _ = self.capture("baseline")
        changes = [("workload", "seed", 43), ("workload", "warmup_ms", 21),
                   ("workload", "launch", "stream"), ("environment", "driver_api", 13001)]
        for index, (section, field, value) in enumerate(changes):
            with self.subTest(field=field):
                self.payload = result_fixture(False)
                self.payload[section][field] = value
                other, manifest = self.capture(str(index))
                self.assertEqual(manifest["status"], "passed")
                with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                    observe.compare(baseline, other)

    def test_window_dimension_compares_even_when_global_ignores_it(self):
        self.payload = result_fixture(False)
        self.payload["implementation"] = "global:0:48:u32:kernel"
        self.payload["memory"]["scratch_bytes"] = 1024
        first, _ = self.capture("first")
        self.payload["workload"]["window_bins"] = 64
        second, _ = self.capture("second")
        with self.assertRaisesRegex(ValueError, "workload"):
            observe.compare(first, second)

    def test_compare_declared_arguments_and_gpu_inventory_mismatch(self):
        first, _ = self.capture("first", ["--seed", "42"])
        second, _ = self.capture("second", ["--seed", "43"])
        with self.assertRaisesRegex(ValueError, "argument protocol mismatch"):
            observe.compare(first, second)
        self.gpu_output = "GPU-1111, Fake GPU, 590.0, 1000, 5000, 45, 60\n"
        third, _ = self.capture("third", ["--seed", "42"])
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            observe.compare(first, third)

    def test_compare_allows_implementation_selector_change_with_caveat(self):
        self.payload = result_fixture(False)
        first, _ = self.capture("first", ["--variant", "global_window", "--tuning", "2", "--blocks", "96"])
        self.payload["implementation"] = "global:0:48:u32:kernel"
        self.payload["memory"]["scratch_bytes"] = 1024
        second, _ = self.capture("second", ["--variant", "global", "--tuning", "0", "--blocks", "48"])
        report = observe.compare(first, second)
        self.assertEqual(report["comparison_kind"], "count_component_comparison")
        self.assertEqual(report["right_over_left"]["operation_median_ms"], 1.0)

    def test_cli_argument_list_is_not_a_shell_command(self):
        output = self.base / "cli"
        argument = "$(never-run) ; literal `text`"
        with contextlib.redirect_stdout(io.StringIO()):
            code = observe.main(["capture", "--exe", str(self.exe), "--output-dir", str(output), "--", argument])
        self.assertEqual(code, 0)
        self.assertEqual(observe.read_json(output / "command.json"), [str(self.exe), argument])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(observe.main(["audit", str(output)]), 0)


class ResultContractTests(unittest.TestCase):
    def test_valid_instrumented_and_disabled_results(self):
        for enabled in (True, False):
            self.assertEqual(observe.validate_result(result_fixture(enabled))["instrumented"], enabled)

    def test_invalid_contract_fields_fail_closed(self):
        changes = [(("schema",), True), (("build", "id"), "bad"), (("instrumented",), 1),
                   (("environment", "sm"), False), (("workload", "rows"), 0),
                   (("workload", "rows"), 1 << 32), (("workload", "batch"), 4097),
                   (("workload", "features"), 2), (("workload", "warmup_ms"), 0.5),
                   (("workload", "cache"), "cold"), (("workload", "distribution"), "skew"),
                   (("workload", "counter"), "u32"), (("implementation",), "other:2:96:u32:kernel"),
                   (("implementation",), "global:6:96:u32:kernel"), (("implementation",), "global:2:0:u32:kernel"),
                   (("implementation",), "global:02:96:u32:kernel"), (("implementation",), "global:2:96:u64:kernel"),
                   (("memory", "input_bytes"), 1), (("memory", "scratch_bytes"), 1),
                   (("memory", "validation_snapshot_bytes"), 1), (("memory", "free_before_bytes"), 1_000_001),
                   (("timing", "operation_ms"), [1, 2]), (("timing", "operation_ms"), [1, 0, 3]),
                   (("timing", "operation_ms"), [1, float("nan"), 3]),
                   (("timing", "device_span_ms"), 0), (("timing", "prepare_wall_ms"), float("inf")),
                   (("timing", "readback_submit_wall_ms"), -1), (("validation", "checked_bins"), True)]
        for path, value in changes:
            with self.subTest(path=path, value=value):
                result = result_fixture()
                target = result
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises((ValueError, TypeError)):
                    observe.validate_result(result)

    def test_histogram_sample_coverage_and_context_checked(self):
        mutators = [lambda r: r["samples"].pop(0),
                    lambda r: r["samples"][0]["context"].update(repetition=1),
                    lambda r: r["samples"][0]["context"].update(operations=1),
                    lambda r: r["samples"][0]["context"].update(rows=100),
                    lambda r: r["samples"][0]["context"].update(logical_read_bytes=4096),
                    lambda r: r["samples"][0]["context"].update(stream_id=-1),
                    lambda r: r["samples"][0].update(id=True),
                    lambda r: r["samples"][0].update(host_end_ns=-1),
                    lambda r: r["samples"][0].update(host_start_ns=100, host_end_ns=99),
                    lambda r: r["samples"][0].update(stage="unknown"),
                    lambda r: r["samples"][0].update(gpu_ms=None),
                    lambda r: r["samples"][3].update(gpu_ms=1),
                    lambda r: r.update(instrumented=False)]
        for index, mutate in enumerate(mutators):
            with self.subTest(index=index):
                result = result_fixture()
                mutate(result)
                with self.assertRaises(ValueError):
                    observe.validate_result(result)

    def test_zero_stage_interval_is_valid_but_nvtx_requires_instrumentation(self):
        result = result_fixture()
        result["samples"][0]["gpu_ms"] = 0
        observe.validate_result(result)
        result = result_fixture(False)
        result["nvtx"] = True
        with self.assertRaises(ValueError):
            observe.validate_result(result)

    def test_json_duplicates_and_nonfinite_values_rejected(self):
        for text in ('{"a":1,"a":2}', '{"nested":{"x":0,"x":1}}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    observe.parse_json(text)


if __name__ == "__main__":
    unittest.main()
