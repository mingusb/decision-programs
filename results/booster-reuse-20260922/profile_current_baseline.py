"""Current-binary diagnostic controls matching cached-root and batch-split captures."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode = True
from run_diagnostic import run
OUT = Path(__file__).resolve().parent
for source, name in (("ncu-cached-root", "ncu-per-output-root"), ("ncu-batched-split", "ncu-per-tree-split")):
    command = json.loads((OUT / (source + "-command.json")).read_text())["command"]
    command = [value.replace(source, name) for value in command]
    command[command.index("--root-counts") + 1] = "per-output"
    command[command.index("--split-batch") + 1] = "per-tree"
    if run(name, command): raise SystemExit(name + " failed")
    report = command[command.index("--export") + 1] + ".ncu-repz"
    for page in ("details", "raw"):
        if run(name + "-" + page, [command[0], "--import", report, "--page", page, "--csv"]):
            raise SystemExit(name + " import failed")
