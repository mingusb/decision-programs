#!/usr/bin/env python3
"""Offline evidence parser contracts; no CUDA runtime or GPU is invoked."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("compiler_report", Path(__file__).resolve().parents[1] / "tools/compiler_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def cubin_fixture():
    names = b"\0.shstrtab\0.text.kernel\0.nv.constant0.kernel\0"
    code = b"12345678"
    constant = b"ABCD"
    content = names + code + constant
    start = 64 + len(content)
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01" + b"\0" * 10,
                         2, 190, 1, 0, 0, start, 0, 64, 0, 0, 64, 4, 1)
    rows = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0),
            struct.pack("<IIQQQQIIQQ", names.index(b".text"), 1, 6, 0, 64 + len(names), len(code), 0, 0, 1, 0),
            struct.pack("<IIQQQQIIQQ", names.index(b".nv.constant"), 1, 2, 0, 64 + len(names) + len(code), len(constant), 0, 0, 1, 0)]
    return header + content + b"".join(rows)


class EvidenceTests(unittest.TestCase):
    def test_resource_fields_and_absence(self):
        parsed = report.resources(" Common:\n GLOBAL:12\n Function foo:\n REG:32 SHARED:128 LOCAL:0 CONSTANT[0]:360\n Function bar:\n REG:8\n")
        self.assertEqual(parsed[0]["resources"], {"REG": 32, "SHARED": 128, "LOCAL": 0, "CONSTANT[0]": 360})
        self.assertNotIn("SHARED", parsed[1]["resources"])

    def test_spills_not_invented(self):
        data = "ptxas info : Function properties for foo\n 8 bytes stack frame, 4 bytes spill stores, 12 bytes spill loads\nptxas info : Function properties for bar\n"
        a, b = report.ptxas_records(data)
        self.assertEqual(a["spill_store_bytes"], 4)
        self.assertIsNone(b["spill_load_bytes"])

    def test_elf_code_and_constant_bytes(self):
        values = report.elf_sections(cubin_fixture())
        self.assertEqual([(v["name"], v["kind"], v["bytes"]) for v in values],
                         [(".text.kernel", "code", 8), (".nv.constant0.kernel", "constant", 4)])
        with self.assertRaises(ValueError): report.elf_sections(cubin_fixture()[:-1])
        with self.assertRaises(ValueError): report.elf_sections(b"not an elf")

    def test_trace_units_and_overlap(self):
        result = report.trace_summary({"traceEvents": [{"ph": "X", "name": "outer", "ts": 1000, "dur": 8000},
                                                       {"ph": "X", "name": "inner", "ts": 2000, "dur": 4000}]})
        self.assertEqual(result["complete_event_span_ms"], 8)
        self.assertEqual(result["inclusive_duration_ms_by_name"], {"inner": 4, "outer": 8})
        self.assertIsNone(report.trace_summary({"traceEvents": []})["complete_event_span_ms"])

    def test_invalid_trace_rejected(self):
        for duration in (-1, float("nan"), "3"):
            with self.assertRaises(ValueError):
                report.trace_summary({"traceEvents": [{"ph": "X", "ts": 1, "dur": duration}]})

    def test_ninja_history_retained(self):
        parsed = report.ninja_costs("# ninja log v7\n0\t7\t123\ta.o\tx\n0\t4\t456\ta.o\ty\n")
        self.assertEqual([p["duration_ms"] for p in parsed], [7, 4])
        with self.assertRaises(ValueError): report.ninja_costs("1\t0\t1\ta\tx")

    def test_comparison_cannot_hide_unmatched_symbols(self):
        def data(name=".text.kernel", value="abc"):
            return {"binaries": [{"name": "lib.a", "cubins": [{"name": "a.cubin", "resources": [],
                      "sections": [{"name": name, "bytes": 8, "sha256": value, "kind": "code"}]}]}]}
        self.assertTrue(report.compare_reports(data(), data())["exact_section_identity_match"])
        changed = report.compare_reports(data(), data(value="def"))
        self.assertEqual(len(changed["changed_sections"]), 1)
        renamed = report.compare_reports(data(), data(name=".text.renamed"))
        self.assertEqual(renamed["matched_sections"], 0)
        self.assertFalse(renamed["exact_section_identity_match"])
        self.assertEqual(len(renamed["baseline_only"]), 1)

    def test_source_identity_excludes_build_and_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.cu").write_text("source")
            (root / "build").mkdir(); (root / "build/other.cu").write_text("generated")
            (root / "results").mkdir(); (root / "results/other.cu").write_text("archived")
            self.assertEqual(set(report.source_identity(root)), {"a.cu"})

    def test_refuses_existing_output_before_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "binary").write_bytes(b"x")
            status = report.main(["--build-dir", str(root), "--source-root", str(root),
                                  "--binary", str(root / "binary"), "--output", str(root),
                                  "--cuobjdump", "/does/not/exist"])
            self.assertEqual(status, 2)
            self.assertFalse((root / "report.json").exists())

    def test_failure_preserved_as_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "binary").write_bytes(b"x")
            status = report.main(["--build-dir", str(root), "--source-root", str(root),
                                  "--binary", str(root / "binary"), "--output", str(root / "evidence"),
                                  "--cuobjdump", "/does/not/exist"])
            self.assertEqual(status, 1)
            data = json.loads((root / "evidence/report.json").read_text())
            self.assertEqual(data["status"], "failed")
            self.assertFalse(data["gpu_executed"])
            self.assertIsNone(data["compile_cost"]["build_wall_ms"])


if __name__ == "__main__":
    unittest.main()
