#!/usr/bin/env python3
"""CPU-only identity/metric/SASS audit of completed Nsight Compute captures."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
BINARY = WORKSPACE / "build/booster-level-batch/ghb_bench"

def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(path.read_text())

def audit():
    output = ROOT / "compute-audit.json"
    require(not output.exists(), "refusing to overwrite compute audit")
    binary_hash = digest(BINARY)
    sass_path = ROOT / "compute-sass.txt"
    sass = sass_path.read_text()
    require("arch = sm_86" in sass, "SASS architecture missing")
    sections = re.split(r"\n\s*Function : ", sass)[1:]
    specifications = {
        "deeper": ("global_accumulate<16, 1>", "global_accumulateILj16ELb1E", 256, 256),
        "materialize": ("materialize_small(", "materialize_small", 4, 16),
        "split": ("warp_candidates<1>", "warp_candidatesILb1E", 32, 1024),
    }
    captures = []
    for name, (kernel_token, sass_token, block, grid) in specifications.items():
        paths = {suffix: ROOT / f"ncu-{name}{suffix}" for suffix in ["-command.json", ".stdout", ".stderr", ".ncu-repz", "-raw-command.json", "-raw.stdout", "-raw.stderr", "-details-command.json", "-details.stdout", "-details.stderr", "-benchmark/result.json"]}
        hashes = {key: digest(path) for key, path in paths.items()}
        receipt = read(paths["-command.json"])
        require(receipt["returncode"] == 0 and receipt["command_executables_unchanged"], "failed/changed executable receipt")
        require(receipt["command_executables_sha256"][str(BINARY)] == binary_hash, "capture binary differs from disassembled binary")
        command = receipt["command"]
        require(command[command.index("--launch-count") + 1] == "1" and command[command.index("--clock-control") + 1] == "none", "unexpected launch/clock scope")
        require("--launch-skip" not in command, "unexpected skipped launches")
        result = read(paths["-benchmark/result.json"])
        expected = dict(rows=4096, test_rows=128, features=16, outputs=16, rounds=1, max_depth=3, max_bins=32,
                        output_tile_size=16, tree_batch_size=16, tree_build="output-batch", tree_execution="stream",
                        instrumentation="off", effective_deeper_histogram="global")
        require(all(result.get(k) == v for k, v in expected.items()), "unexpected profiler workload")
        emitted = [json.loads(line) for line in paths[".stdout"].read_text().splitlines() if line.startswith('{"schema_version"')]
        require(emitted == [result], "emitted/saved training result mismatch")
        for page in ("raw", "details"):
            imp = read(paths[f"-{page}-command.json"])
            require(imp["returncode"] == 0 and imp["command_executables_unchanged"], "failed report import")
            require(imp["command"][imp["command"].index("--import") + 1] == str(paths[".ncu-repz"].relative_to(WORKSPACE)), "import names another report")
            require(imp["command"][imp["command"].index("--page") + 1] == page, "wrong import page")
        with paths["-raw.stdout"].open() as source:
            raw_rows = list(csv.DictReader(source))
        data = [row for row in raw_rows if row["ID"]]
        units = [row for row in raw_rows if not row["ID"]]
        require(len(data) == 1 and len(units) == 1, "expected exactly one profiled launch plus units row")
        raw = data[0]
        with paths["-details.stdout"].open() as source:
            details = list(csv.DictReader(source))
        require(all(row["ID"] == raw["ID"] and row["Kernel Name"] == raw["Kernel Name"] and row["Process ID"] == raw["Process ID"] for row in details), "raw/detail identity mismatch")
        require(kernel_token in raw["Kernel Name"] and raw["CC"] == "8.6", "wrong selected kernel/architecture")
        require(int(raw["launch__block_size"]) == block and int(raw["launch__grid_size"]) == grid, "unexpected launch geometry")
        require(raw["Block Size"] == f"({block}, 1, 1)" and raw["Grid Size"] == f"({grid}, 1, 1)", "dimension/flat launch disagreement")
        metrics = {f"{row['Section Name']}/{row['Metric Name']}": {"value": row["Metric Value"], "unit": row["Metric Unit"]} for row in details if row["Metric Name"]}
        rules = [{key: row[key] for key in ["Section Name", "Rule Name", "Rule Type", "Rule Description", "Estimated Speedup Type", "Estimated Speedup"]} for row in details if row["Rule Name"]]
        total = int(raw["memory_l2_theoretical_sectors_global"])
        ideal = int(raw["memory_l2_theoretical_sectors_global_ideal"])
        selected_raw = {k: {"value": v, "unit": units[0].get(k)} for k, v in raw.items() if any(token in k for token in
            ["scoreboard_per_issue_active", "math_pipe_throttle_per_issue_active", "mio_throttle_per_issue_active", "memory_l2_theoretical_sectors", "sm__pipe_fp64_cycles_active.avg", "spilling", "bank_conflicts", "sass_inst_executed_op_shared"])}
        functions = [section for section in sections if sass_token in section.splitlines()[0]]
        require(len(functions) == 1, "SASS function identity ambiguous")
        function = functions[0]
        mangled = function.splitlines()[0]
        demangled = subprocess.check_output(["c++filt", mangled], text=True).strip()
        instructions = []
        for line in function.splitlines():
            match = re.search(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line)
            if match:
                instructions.append(match[1])
        counts = dict(collections.Counter(instructions))
        captures.append({"name": name, "kernel": raw["Kernel Name"], "process_id": raw["Process ID"], "block": block, "grid": grid,
                         "configuration": expected, "seed": result["seed"], "test_seed": result["test_seed"],
                         "metrics": metrics, "selected_raw_metrics": selected_raw, "rules": rules,
                         "global_theoretical_sectors": {"total": total, "ideal": ideal, "excessive": total - ideal,
                                                       "excessive_percent": 100 * (total - ideal) / total if total else 0},
                         "sass": {"mangled": mangled, "demangled": demangled, "static_instruction_counts": counts,
                                  "scope": "static disassembly, not executed instruction counts"},
                         "artifacts": {key: {"path": str(path), "sha256": hashes[key]} for key, path in paths.items()}})
        require(all(digest(path) == hashes[key] for key, path in paths.items()), "artifact changed during audit")
    sources = ["training/src/deeper_histogram.cu", "training/src/split_search.cu", "training/src/batch_resident.cu", "training/src/batch_training.inc"]
    report = {"schema": 1, "audit": "CPU-only completed Nsight Compute report identity, geometry, metrics and same-binary SASS",
              "binary": {"path": str(BINARY), "sha256": binary_hash}, "sass": {"path": str(sass_path), "sha256": digest(sass_path)},
              "auditor_sha256": digest(Path(__file__)), "source_sha256": {p: digest(WORKSPACE / p) for p in sources}, "captures": captures,
              "limits": ["Each report captures only the first matching kernel launch, not all depths/tiles/distributions.",
                         "Counters are replayed by Nsight Compute; profiler duration and estimated speedups never rank uninstrumented implementations.",
                         "No explicit launch skip is used. Kernel arguments/active-node counts are not exported, so depth attribution also relies on the captured call sequence/source.",
                         "Global theoretical sectors are source-counter estimates, not literal external-DRAM bytes.",
                         "Static SASS instruction counts show available operations, not dynamic instruction frequencies or a proven causal stall location."]}
    require(digest(BINARY) == binary_hash, "binary changed during audit")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"captures": len(captures), "binary_sha256": binary_hash, "output": str(output)}))

if __name__ == "__main__":
    audit()
