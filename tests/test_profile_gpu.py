#!/usr/bin/env python3
"""CPU-only diagnostics-runner tests. Fake tools never load a CUDA library."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("profile_gpu", ROOT / "tools/profile_gpu.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.target = self.script("target", "print('target')\n")
        for name in ("nsys", "ncu", "compute-sanitizer", "nvdisasm"):
            self.script(name, "print('fake CPU tool')\n")
        self.env = {"PATH": str(self.bin), "NVBIT_ROOT": str(self.root / "nvbit"),
                    "CUPTI_ROOT": str(self.root / "cupti")}
        for name in ("instr_count", "instr_count_cuda_graph", "mem_trace"):
            self.file("nvbit/tools/" + name + "/" + name + ".so")
        for relative in ("cupti_trace_injection/libcupti_trace_injection.so", "profiling_injection/libinjection.so",
                         "pc_sampling_continuous/libpc_sampling_continuous.so", "pc_sampling_utility/pc_sampling_utility"):
            self.file("cupti/samples/" + relative)
        self.file("cupti/lib/libcupti.so")

    def script(self, name, content):
        path = self.bin / name
        path.write_text("#!" + sys.executable + "\n" + textwrap.dedent(content))
        path.chmod(0o755)
        return path

    def file(self, name, content="CPU fixture\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def args(self, tool="ncu", options=(), command=None, output="evidence"):
        return runner.arguments(["--tool", tool, "--output", str(self.root / output), *options,
                                 "--", *(command or [str(self.target)])])

    def plan(self, tool="ncu", options=(), command=None):
        return runner.build_plan(self.args(tool, options, command), self.env)

    def test_ncu_construction_and_sections(self):
        plan = self.plan(options=["--capture", "--kernel", "ghb::.*split", "--launch-count", "3", "--launch-skip", "2"])
        command = plan["argv"]
        self.assertIn("regex:ghb::.*split", command)
        self.assertEqual(command[command.index("--launch-count") + 1], "3")
        self.assertEqual(command[command.index("--launch-skip") + 1], "2")
        self.assertEqual(command[command.index("--set") + 1], "basic")
        self.assertEqual(command[command.index("--profile-from-start") + 1], "off")
        self.assertEqual(command[command.index("--clock-control") + 1], "none")
        self.assertEqual(command[command.index("--kernel-name-base") + 1], "demangled")
        self.assertEqual(plan["env_overrides"]["GH_PROFILE_CAPTURE"], "1")
        custom = self.plan(options=["--section", "SourceCounters", "--section", "PmSampling", "--metrics", "x,y"])
        self.assertNotIn("--set", custom["argv"])
        self.assertEqual(custom["argv"].count("--section"), 2)
        self.assertEqual(custom["argv"][-1], str(self.target))

    def test_nsys_repeated_capture_and_report_names(self):
        plan = self.plan("nsys", ["--capture"])
        self.assertIn("--capture-range-end=repeat", plan["argv"])
        self.assertIn("--capture-range=cudaProfilerApi", plan["argv"])
        self.assertTrue(plan["expected_reports"][0].endswith("profile*.nsys-rep"))
        self.assertEqual(plan["cwd"], str(self.root / "evidence"))
        self.assertIn("--resolve-symbols=false", plan["argv"])
        self.assertFalse(any(arg.startswith("--cuda-graph-trace") for arg in plan["argv"]))
        self.assertEqual(plan["collection_settings"]["graph_trace"], "tool-default")
        self.assertIn("--session-new=" + plan["nsys_session"], plan["argv"])
        self.assertNotEqual(plan["nsys_session"], self.plan("nsys")["nsys_session"])
        self.assertIn("--resolve-symbols=true", self.plan("nsys", ["--nsys-resolve-symbols"])["argv"])

    def test_replay_granularity_and_cache_plans(self):
        default = self.plan()
        self.assertEqual(default["collection_settings"],
                         {"replay_mode": "kernel", "graph_profiling": "node", "cache_control": "all"})
        application = self.plan(options=["--replay-mode", "application", "--cache-control", "none"])
        command = application["argv"]
        self.assertEqual(command[command.index("--app-replay-match") + 1], "all")
        self.assertEqual(command[command.index("--app-replay-mode") + 1], "strict")
        self.assertEqual(command[command.index("--cache-control") + 1], "none")
        for replay in ("range", "app-range"):
            plan = self.plan(options=["--replay-mode", replay, "--capture"])
            self.assertEqual(plan["workload_kind"], "range")
            self.assertIn(runner.CTA_METRIC, plan["argv"])
            self.assertNotIn("--app-replay-match", plan["argv"])
            # Range replay always uses profiler API/NVTX boundaries; the kernel
            # capture switch is rejected by NCU before it starts the target.
            self.assertNotIn("--profile-from-start", plan["argv"])
            self.assertEqual(plan["env_overrides"]["GH_PROFILE_CAPTURE"], "1")
        application_capture = self.plan(options=["--replay-mode", "application", "--capture"])
        self.assertIn("--profile-from-start", application_capture["argv"])
        plan = self.plan(options=["--graph-profiling", "graph", "--metrics", runner.CTA_METRIC])
        self.assertEqual(plan["workload_kind"], "graph")
        metrics = plan["argv"][plan["argv"].index("--metrics") + 1].split(",")
        self.assertEqual(metrics.count(runner.CTA_METRIC), 1)
        self.assertIn(runner.GRAPH_METRIC, metrics)
        for value in ("node", "graph"):
            plan = self.plan("nsys", ["--nsys-graph-trace", value])
            self.assertIn("--cuda-graph-trace=" + value, plan["argv"])
            self.assertEqual(plan["workload_kind"], "kernel" if value == "node" else "graph")

    def test_advanced_sanitizer_plans(self):
        plan = self.plan("memcheck", ["--padding", "128", "--leak-check", "full"])
        self.assertEqual(plan["collection_settings"], {"padding": 128, "leak-check": "full"})
        self.assertIn("128", plan["argv"])
        for space in ("global", "shared", "all"):
            plan = self.plan("initcheck", ["--initcheck-address-space", space, "--track-unused-memory",
                                           "--unused-memory-threshold", "20"])
            self.assertEqual(plan["collection_settings"]["initcheck-address-space"], space)
            self.assertEqual(plan["collection_settings"]["unused-memory-threshold"], 20)
            self.assertIn("--track-unused-memory", plan["argv"])
        for tool in runner.SANITIZERS:
            self.assertEqual(self.plan(tool)["collection_settings"], {})

    def test_tool_specific_and_unsupported_aggregate_options_rejected(self):
        cases = [("memcheck", ["--cache-control", "none"]),
                 ("ncu", ["--nsys-graph-trace", "graph"]),
                 ("initcheck", ["--padding", "1"]),
                 ("racecheck", ["--leak-check", "full"]),
                 ("memcheck", ["--initcheck-address-space", "shared"]),
                 ("memcheck", ["--track-unused-memory"]),
                 ("initcheck", ["--unused-memory-threshold", "0"]),
                 ("initcheck", ["--track-unused-memory", "--unused-memory-threshold", "101"]),
                 ("ncu", ["--replay-mode", "range"]),
                 ("ncu", ["--replay-mode", "app-range"]),
                 ("ncu", ["--replay-mode", "application", "--graph-profiling", "graph"]),
                 ("ncu", ["--capture", "--replay-mode", "app-range", "--graph-profiling", "graph"]),
                 ("ncu", ["--graph-profiling", "graph", "--kernel", "histogram"]),
                 ("ncu", ["--graph-profiling", "graph", "--section", "SourceCounters"]),
                 ("ncu", ["--capture", "--replay-mode", "range", "--section", "InstructionStats"]),
                 ("ncu", ["--capture", "--replay-mode", "app-range", "--metrics", "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"]),
                 ("ncu", ["--graph-profiling", "graph", "--metrics", "sass__inst_executed_per_opcode"])]
        for tool, options in cases:
            with self.subTest(tool=tool, options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(tool, options)
        # Instruction-level SASS is allowed for application range (except JIT kernels).
        self.plan(options=["--capture", "--replay-mode", "app-range", "--metrics", "sass__inst_executed_per_opcode"])
        # Sampling sections are separate from unsupported source sections.
        self.plan(options=["--graph-profiling", "graph", "--section", "PmSampling"])

    def test_range_replay_rejects_graph_target_but_app_range_accepts_it(self):
        booster = self.script("ghb_bench", "pass\n")
        with self.assertRaisesRegex(ValueError, "requires --tree-execution stream"):
            self.plan(options=["--capture", "--replay-mode", "range"], command=[str(booster)])
        with self.assertRaisesRegex(ValueError, "graph target"):
            self.plan(options=["--capture", "--replay-mode", "range"],
                      command=[str(booster), "--tree-execution=graph"])
        self.plan(options=["--capture", "--replay-mode", "range"], command=[str(booster), "--tree-execution", "stream"])
        self.plan(options=["--capture", "--replay-mode", "app-range"], command=[str(booster), "--tree-execution", "graph"])

    def test_sanitizers_have_error_exit_code(self):
        for tool in runner.SANITIZERS:
            command = self.plan(tool)["argv"]
            self.assertEqual(command[command.index("--error-exitcode") + 1], "99")
            self.assertEqual(command[command.index("--tool") + 1], tool)

    def test_inherited_injection_refused(self):
        for variable in runner.INJECTIONS:
            with self.subTest(variable=variable), self.assertRaisesRegex(ValueError, "inherited injection"):
                runner.build_plan(self.args(), {**self.env, variable: "/unwanted.so"})
        plan = runner.build_plan(self.args(), {**self.env, "LD_PRELOAD": ""})
        self.assertNotIn("LD_PRELOAD", plan["environment"])

    def test_graph_guards_and_default_booster_ambiguity(self):
        booster = self.script("ghb_bench", "pass\n")
        for tool in ("nvbit-count", "nvbit-memory", "cupti-range"):
            for tail in (["--launch", "graph"], ["--tree-execution=graph"]):
                with self.subTest(tool=tool, tail=tail), self.assertRaisesRegex(ValueError, "graph target"):
                    self.plan(tool, command=[str(self.target), *tail])
            with self.subTest(tool=tool), self.assertRaisesRegex(ValueError, "requires --tree-execution stream"):
                self.plan(tool, command=[str(booster)])
            self.plan(tool, command=[str(booster), "--tree-execution", "stream"])
            histogram = self.script("ghb_instrumentation_bench", "pass\n")
            with self.assertRaisesRegex(ValueError, "requires --launch stream"):
                self.plan(tool, command=[str(histogram)])
            self.plan(tool, command=[str(histogram), "--launch=stream"])
        self.plan("nvbit-graph", command=[str(booster), "--tree-execution", "graph"])

    def test_nvbit_graph_function_bounds_distinct_from_launch_bounds(self):
        plan = self.plan("nvbit-graph", ["--instruction-end", "64", "--function-skip", "2", "--function-count", "4"])
        env = plan["env_overrides"]
        self.assertEqual((env["START_GRID_NUM"], env["END_GRID_NUM"], env["INSTR_END"]), ("2", "6", "64"))
        self.assertEqual(plan["argv"], [str(self.target)])
        self.assertNotIn("CUDA_INJECTION64_PATH", plan["environment"])

    def test_cupti_paths_and_modes(self):
        trace = self.plan("cupti-trace")["env_overrides"]
        self.assertEqual(trace["CUDA_INJECTION64_PATH"], trace["NVTX_INJECTION64_PATH"])
        self.assertNotIn("LD_PRELOAD", trace)
        pc = self.plan("cupti-pc")
        self.assertIn("--collection-mode 1", pc["env_overrides"]["INJECTION_PARAM"])
        self.assertIn("--file-name pcsampling.dat", pc["env_overrides"]["INJECTION_PARAM"])
        self.assertTrue(pc["decoder"].endswith("/pc_sampling_utility"))
        self.assertEqual(self.plan("cupti-range")["env_overrides"]["INJECTION_METRICS"], "sm__ctas_launched.sum")

    def test_counter_proof_rejects_loaded_banner_zero_and_metadata_address(self):
        output = self.root / "proof"
        output.mkdir()
        log = output / "stdout.log"
        for text in ("NVBit Loaded\n", "Total app instructions: 123\n", "kernel instructions 0\n"):
            log.write_text(text)
            self.assertFalse(runner.evidence("nvbit-count", output)["passed"])
        log.write_text("kernel 0 - my_histogram - #thread-blocks 1, kernel instructions 202, total instructions 202\n")
        proof = runner.evidence("nvbit-graph", output)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["kernel_names"], ["my_histogram"])
        with log.open("a") as stream:
            stream.write("We ran out of kernel_counters, please increase MAX_NUM_KERNEL to 101\n")
        self.assertFalse(runner.evidence("nvbit-graph", output)["passed"])
        log.write_text("MEMTRACE: CTX 0x123 - LAUNCH - Kernel pc 0x456\n")
        self.assertFalse(runner.evidence("nvbit-memory", output)["passed"])
        log.write_text("MEMTRACE: CTX 0x123 - grid_launch_id 0 - CTA 0,0,0 - warp 0 - LDG.E.64 - 0x456\n")
        self.assertTrue(runner.evidence("nvbit-memory", output)["passed"])

    def test_cupti_pc_requires_decoded_user_records_and_keeps_drops(self):
        output = self.root / "pcproof"
        output.mkdir()
        (output / "1_pcsampling.dat").write_bytes(b"sample")
        log = output / "stdout.log"
        log.write_text("Initialize injection\nNon User Kernels Total Samples: 123\n")
        self.assertFalse(runner.evidence("cupti-pc", output)["passed"])
        log.write_text("Count of PC records: 2, Total Samples: 9, Total Dropped Samples: 3\n"
                       "pc, functionName: ghb_kernel, functionIndex: 1, correlationId: 0, pcOffset: 4\n")
        proof = runner.evidence("cupti-pc", output)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["pc_records"], 2)
        self.assertEqual(proof["dropped_samples"], 3)

    def test_report_gate_requires_nonempty_file(self):
        output = self.root / "reports"
        output.mkdir()
        pattern = output / "profile*.nsys-rep"
        self.assertFalse(runner.evidence("nsys", output, [pattern])["passed"])
        (output / "profile1.nsys-rep").touch()
        self.assertFalse(runner.evidence("nsys", output, [pattern])["passed"])
        (output / "profile1.nsys-rep").write_bytes(b"CPU fixture")
        self.assertTrue(runner.evidence("nsys", output, [pattern])["passed"])

    def test_ncu_compressed_report_needs_actual_matching_kernel_export(self):
        self.script("ncu", """
            import pathlib,sys
            if '--import' in sys.argv:
                print('"ID","Kernel Name","Metric"')
                print('"0","void ghb::histogram<int>()","42"')
            else:
                pathlib.Path('profile.ncu-repz').write_bytes(b'CPU report fixture')
            """)
        args = self.args("ncu", ["--kernel", "ghb::histogram"])
        self.assertEqual(runner.run(args, self.env), 0)
        receipt = json.loads((args.output / "manifest.json").read_text())
        self.assertTrue(receipt["evidence"]["kernel_data"]["passed"])
        self.assertEqual(receipt["evidence"]["kernel_data"]["kernel_records"], 1)
        bad = self.args("ncu", ["--kernel", "different_kernel"], output="wrong-kernel")
        self.assertEqual(runner.run(bad, self.env), 1)

    def test_csv_header_or_units_row_is_not_a_kernel(self):
        path = self.file("header.csv", '"ID","Kernel Name","Metric"\n"","","unit"\n')
        self.assertFalse(runner.exported_kernels("ncu", [path])["passed"])
        path.write_text('Time (%),Instances,Name\n100,0,empty\n')
        self.assertFalse(runner.exported_kernels("nsys", [path])["passed"])
        path.write_text('Time (%),Instances,Name\n100,2,"my_kernel<int>()"\n')
        self.assertEqual(runner.exported_kernels("nsys", [path])["kernel_records"], 2)

    def test_aggregate_ncu_requires_positive_device_work_not_names(self):
        path = self.file("aggregate.csv", f'"ID","{runner.CTA_METRIC}"\n"","block"\n"0","64"\n')
        proof = runner.exported_kernels("ncu", [path], workload="range")
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["ctas_launched"], 64)
        self.assertEqual(proof["workload_records"], 1)
        self.assertEqual(proof["workload_names"], [])
        self.assertFalse(proof["per_kernel_attribution"])
        self.assertEqual(proof["kernel_records"], 0)
        path.write_text(f'"ID","Range Name","Metric Name","Metric Value"\n"7","train","{runner.CTA_METRIC}","1,234"\n')
        proof = runner.exported_kernels("ncu", [path], workload="range")
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["ctas_launched"], 1234)
        self.assertEqual(proof["workload_names"], ["train"])
        for content in ('"ID","Kernel Name"\n"0","metadata only"\n',
                        f'"ID","{runner.CTA_METRIC}"\n"","block"\n',
                        f'"ID","{runner.CTA_METRIC}"\n"0","0"\n',
                        f'"ID","{runner.CTA_METRIC}"\n"0","nan"\n',
                        f'"ID","Metric Name","Metric Value"\n"0","other","99"\n'):
            with self.subTest(content=content):
                path.write_text(content)
                self.assertFalse(runner.exported_kernels("ncu", [path], workload="range")["passed"])

    def test_whole_graph_requires_graph_identity_on_positive_workload(self):
        path = self.file("whole-graph.csv", f'"ID","{runner.CTA_METRIC}","{runner.GRAPH_METRIC}"\n"0","64","0"\n')
        proof = runner.exported_kernels("ncu", [path], workload="graph")
        self.assertFalse(proof["passed"])
        self.assertEqual(proof["ignored_standalone_records"], 1)
        # A separate graph metadata-only record cannot certify standalone work.
        with path.open("a") as stream:
            stream.write('"1","0","12"\n')
        self.assertFalse(runner.exported_kernels("ncu", [path], workload="graph")["passed"])
        with path.open("a") as stream:
            stream.write('"2","32","13"\n')
        proof = runner.exported_kernels("ncu", [path], workload="graph")
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["ctas_launched"], 32)
        self.assertEqual(proof["workload_records"], 1)
        self.assertEqual(proof["graph_ids"], ["13"])
        # Long-form exports can carry count and identity in distinct metric rows.
        path.write_text(f'"ID","Metric Name","Metric Value"\n"0","{runner.CTA_METRIC}","16"\n'
                        f'"0","{runner.GRAPH_METRIC}","9"\n')
        self.assertTrue(runner.exported_kernels("ncu", [path], workload="graph")["passed"])
        path.write_text(f'"ID","{runner.CTA_METRIC}"\n"0","64"\n')
        self.assertFalse(runner.exported_kernels("ncu", [path], workload="graph")["passed"])

    def test_systems_graph_requires_device_graph_activity(self):
        path = self.root / "graphs.sqlite"
        with contextlib.closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,name TEXT)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,10,'cudaGraphLaunch')")
        self.assertFalse(runner.exported_graphs([path])["passed"])
        with contextlib.closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER,end INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (3,3)")
        self.assertFalse(runner.exported_graphs([path])["passed"])
        with contextlib.closing(sqlite3.connect(path)) as db, db:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (4,14)")
        proof = runner.exported_graphs([path])
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["workload_records"], 1)
        self.assertEqual(proof["device_graph_duration_ns"], 10)
        self.assertFalse(proof["per_kernel_attribution"])

    def test_aggregate_ncu_export_end_to_end(self):
        self.script("ncu", """
            import pathlib,sys
            if '--import' in sys.argv:
                print('"ID","Range Name","sm__ctas_launched.sum"')
                print('"0","train","32"')
            else:
                pathlib.Path('profile.ncu-repz').write_bytes(b'CPU report fixture')
            """)
        args = self.args("ncu", ["--capture", "--replay-mode", "app-range"])
        self.assertEqual(runner.run(args, self.env), 0)
        receipt = json.loads((args.output / "manifest.json").read_text())
        self.assertEqual(receipt["workload_kind"], "range")
        self.assertEqual(receipt["evidence"]["kernel_data"]["ctas_launched"], 32)

    def test_systems_graph_export_end_to_end(self):
        self.script("nsys", """
            import contextlib,pathlib,sqlite3,sys
            if sys.argv[1] == 'export':
                path = sys.argv[sys.argv.index('--output')+1]
                with contextlib.closing(sqlite3.connect(path)) as db, db:
                    db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER,end INTEGER)')
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (10,100)')
            else:
                pathlib.Path('profile1.nsys-rep').write_bytes(b'CPU report fixture')
            """)
        args = self.args("nsys", ["--capture", "--nsys-graph-trace", "graph"])
        self.assertEqual(runner.run(args, self.env), 0)
        receipt = json.loads((args.output / "manifest.json").read_text())
        proof = receipt["evidence"]["kernel_data"]
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["device_graph_duration_ns"], 90)
        self.assertEqual(receipt["exports"][0]["argv"][1:4], ['export', '--type', 'sqlite'])

    def test_success_manifest_hashes_and_overwrite_refusal(self):
        self.script("compute-sanitizer", "import sys\nprint('target output')\nprint('ERROR SUMMARY: 0 errors', file=sys.stderr)\n")
        args = self.args("memcheck")
        self.assertEqual(runner.run(args, self.env), 0)
        receipt = json.loads((args.output / "manifest.json").read_text())
        self.assertEqual(receipt["status"], "passed")
        self.assertTrue(receipt["diagnostic_only"])
        self.assertFalse(receipt["performance_ranking_valid"])
        self.assertEqual(len(receipt["artifacts"]["target"]["sha256"]), 64)
        original = (args.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.run(args, self.env)
        self.assertEqual((args.output / "manifest.json").read_bytes(), original)

    def test_zero_exit_without_tool_records_fails_and_logs_retained(self):
        args = self.args("memcheck")
        self.assertEqual(runner.run(args, self.env), 1)
        self.assertIn("fake CPU tool", (args.output / "stdout.log").read_text())
        self.assertEqual(json.loads((args.output / "manifest.json").read_text())["status"], "failed")

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux process-group test")
    def test_timeout_kills_descendant_and_keeps_partial_logs(self):
        child = ("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                 "open('descendant.pid','w').write(str(os.getpid())); print('descendant ready',flush=True); time.sleep(60)")
        self.script("compute-sanitizer", "import subprocess,sys,time\n"
                    f"subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True)\n"
                    "print('partial stdout',flush=True)\nprint('partial stderr',file=sys.stderr,flush=True)\ntime.sleep(60)\n")
        args = self.args("memcheck", ["--timeout", "0.5"])
        self.assertEqual(runner.run(args, self.env), 124)
        receipt = json.loads((args.output / "manifest.json").read_text())
        self.assertEqual(receipt["status"], "timed_out")
        self.assertIn("partial stdout", (args.output / "stdout.log").read_text())
        self.assertIn("partial stderr", (args.output / "stderr.log").read_text())
        pid = int((args.output / "descendant.pid").read_text())
        self.assertEqual(receipt["process"]["cleanup"]["survivors"], [])
        captured = [p for p in receipt["process"]["cleanup"]["owned_processes"] if p["pid"] == pid]
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["session"], pid)
        status = Path(f"/proc/{pid}/status")
        deadline = time.monotonic() + 1
        while status.exists() and "State:\tZ" not in status.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        if status.exists():
            # A killed orphan may remain a zombie until the init process reaps it.
            text = status.read_text()
            if "State:\tZ" not in text:
                os.kill(pid, signal.SIGKILL)
            self.assertIn("State:\tZ", text)

    def test_pid_reuse_and_session_substrings_are_not_signaled(self):
        with mock.patch.object(runner, "process_identity", return_value={"pid": 123, "start_ticks": 20}), \
             mock.patch.object(runner.os, "kill") as kill:
            self.assertFalse(runner.signal_owned({"pid": 123, "start_ticks": 10}, signal.SIGKILL))
            kill.assert_not_called()
        for arg in (b"--session-name\0mine-other", b"--session-name=other-mine", b"--unrelated=mine"):
            with mock.patch.object(Path, "read_bytes", return_value=b"fake\0" + arg + b"\0"):
                self.assertFalse(runner.has_session_argument(123, "mine"))
        with mock.patch.object(Path, "read_bytes", return_value=b"fake\0--session-name=mine\0"):
            self.assertTrue(runner.has_session_argument(123, "mine"))

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux detached-daemon test")
    def test_timeout_reaps_detached_owned_session_but_leaves_other_session(self):
        daemon = ("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  "open('daemon.pid','w').write(str(os.getpid())); print('daemon ready',flush=True); time.sleep(60)")
        intermediate = ("import subprocess,sys; subprocess.Popen([sys.executable,'-c'," + repr(daemon) +
                        ",'--session-name',sys.argv[1]],start_new_session=True)")
        self.script("nsys", "import subprocess,sys,time\n"
                    "session=next(a.split('=',1)[1] for a in sys.argv if a.startswith('--session-new='))\n"
                    f"subprocess.run([sys.executable,'-c',{intermediate!r},session],check=True)\n"
                    "print('launcher ready',flush=True)\ntime.sleep(60)\n")
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                                  "--session-name", "gh-profile-cpu-owned-unrelated"], start_new_session=True)
        self.addCleanup(lambda: other.poll() is None and other.kill())
        try:
            with mock.patch.object(runner.uuid, "uuid4") as new_uuid:
                new_uuid.return_value.hex = "cpu-owned"
                args = self.args("nsys", ["--timeout", "0.6"])
                self.assertEqual(runner.run(args, self.env), 124)
            receipt = json.loads((args.output / "manifest.json").read_text())
            cleanup = receipt["process"]["cleanup"]
            pid = int((args.output / "daemon.pid").read_text())
            self.assertIn(pid, [p["pid"] for p in cleanup["owned_processes"]])
            self.assertEqual(cleanup["survivors"], [])
            self.assertIsNone(other.poll())
        finally:
            other.kill()
            other.wait()

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux detached-daemon test")
    def test_early_launcher_exit_still_cleans_owned_detached_session(self):
        session = "gh-profile-test-early-exit"
        child = ("import os,time; open('early.pid','w').write(str(os.getpid())); "
                 "print('early daemon output',flush=True); time.sleep(60)")
        parent = ("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) +
                  ",'--session-name',sys.argv[1]],start_new_session=True); time.sleep(.05)")
        result = runner.execute([sys.executable, "-c", parent, session], dict(os.environ), self.root, 3,
                                self.root / "early.stdout.log", self.root / "early.stderr.log", session)
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertIsNotNone(result["cleanup"])
        self.assertEqual(result["cleanup"]["survivors"], [])
        self.assertIn("early daemon output", (self.root / "early.stdout.log").read_text())


if __name__ == "__main__":
    unittest.main()
