#!/usr/bin/env python3
"""Follow up mismatched initial profiler clocks with an explicit warmup."""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
spec = importlib.util.spec_from_file_location("rec", ROOT / "results/a5000-profiled/run_round.py")
rec = importlib.util.module_from_spec(spec); spec.loader.exec_module(rec)
source = json.loads((BASE / "manifest.json").read_text())
rec.EXE = Path(source["binary"])
rec.ensure_environment(BASE)
jobs = []
for name, original in source["jobs"]:
    if name.endswith("timeline"): continue
    command = list(original)
    command[command.index("--launch-skip") + 1] = "300"
    name += "-warmed"
    command[command.index("-o") + 1] = str(BASE / name)
    command += ["--warmup-ms", "2000"]
    jobs.append((name, command))
with (BASE / "warmed-manifest.json").open("x") as stream:
    json.dump(dict(parent_sha256=rec.sha256(BASE / "manifest.json"),
        target_sha256=source["binary_sha256"], script_sha256=rec.sha256(Path(__file__)), jobs=jobs,
        reason="Initial profiles recorded different clocks. Retain them; profile after 300 matching kernel launches with 2s requested warmup. Clocks remain unlocked and must be checked in exports."), stream, indent=2)
for name, command in jobs:
    assert rec.sha256(rec.EXE) == source["binary_sha256"]
    rec.run(BASE / name, command, ".stdout")
    files = list(BASE.glob(name + ".ncu-rep*")); assert len(files) == 1
    rec.run(BASE / (name + "-export"), ["ncu", "--import", str(files[0]), "--page", "details"], ".txt")
    assert rec.sha256(rec.EXE) == source["binary_sha256"]
