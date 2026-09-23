#!/usr/bin/env python3
"""CPU-only identity/code audit of the isolated old/current comparison build."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
REFERENCE = {
    "old": (ROOT / "build/custom-only/histogram_bench", "d7568b5c93b80fbe39002c82866d08f2c75683b92ae99869d920c9896c481de7"),
    "new": (ROOT / "build/large-bins-overflow/histogram_bench", "ab480a2e08254cb54a5579125ccef3d1101a776aca434ccd0e2e891ddc642ea1"),
}
FAMILIES = ("shared_histogram_loaded", "shared_histogram", "global_histogram",
            "clear_histogram_output", "reduce_partials", "bitplane_histogram_full_tile",
            "bitplane_histogram", "global_narrow_histogram", "clear_narrow_counts",
            "widen_counts", "shared_overflow_histogram")
SELECTED = (
    "clear_histogram_outputIjEEvPT_j",
    "shared_histogram_loadedIjjjLi256ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE1ELm49152EEEvPKT_mjPT0_PT1_",
    "shared_histogram_loadedIjjjLi512ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm49152EEEvPKT_mjPT0_PT1_",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(symbol):
    for family in FAMILIES:
        token = str(len(family)) + family
        if token in symbol:
            return family + symbol.split(token, 1)[1]
    raise ValueError("Unexpected custom kernel " + symbol)


def resource_map(raw, namespace):
    result = {}
    prefix = f"_ZN{len(namespace)}{namespace}"
    for symbol, line in re.findall(r"^\s*Function (\S+):\n[ \t]+([^\n]+)", raw, re.M):
        if not symbol.startswith(prefix):
            continue
        key = normalized(symbol)
        if key in result:
            raise ValueError("Duplicate kernel key " + key)
        result[key] = {"symbol": symbol, "resources": line.strip()}
    if not result:
        raise ValueError("No kernels in namespace " + namespace)
    return result


def words(raw):
    result = {}
    chunks = re.split(r"\bFunction\s*:\s*(\S+)", raw)
    for symbol, body in zip(chunks[1::2], chunks[2::2]):
        key = normalized(symbol)
        sequence = re.findall(r"/\* (0x[0-9a-f]{16}) \*/", body)
        if not sequence or key in result:
            raise ValueError("Empty or duplicate encoding sequence " + key)
        result[key] = sequence
    if set(result) != set(SELECTED):
        raise ValueError("Selected SASS keys differ from the declaration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/preservation-paired/histogram_preservation")
    args = parser.parse_args()
    exe = args.exe.resolve()
    paths = {"paired": exe, **{name: pair[0] for name, pair in REFERENCE.items()}}
    hashes = {name: sha(path) for name, path in paths.items()}
    for name, (_, expected) in REFERENCE.items():
        if hashes[name] != expected:
            raise ValueError("Frozen reference hash mismatch: " + name)
    commands = []

    def run(command, name):
        commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (BASE / name).write_text(result.stdout)
        if result.stderr:
            (BASE / (name + ".stderr")).write_text(result.stderr)
        return result.stdout

    raw = {name: run([CUOBJDUMP, "--gpu-architecture", "sm_86", "--dump-resource-usage", str(path)],
                     name + "-resources.txt") for name, path in paths.items()}
    maps = {}
    comparisons = {}
    for version in REFERENCE:
        reference = resource_map(raw[version], "gh")
        paired = resource_map(raw["paired"], "gh_" + version)
        maps[version] = {"reference": reference, "paired": paired}
        missing = sorted(set(reference) - set(paired))
        added = sorted(set(paired) - set(reference))
        mismatch = [key for key in sorted(reference.keys() & paired.keys())
                    if reference[key]["resources"] != paired[key]["resources"]]
        sequences = {}
        for identity, mapping, binary in (("reference", reference, paths[version]), ("paired", paired, exe)):
            text = run([CUOBJDUMP, "--gpu-architecture", "sm_86", "--dump-sass", "--function",
                        ",".join(mapping[key]["symbol"] for key in SELECTED), str(binary)],
                       f"{version}-{identity}-selected.sass")
            sequences[identity] = words(text)
        instructions = []
        for key in SELECTED:
            entry = {"kernel": key, "identical": sequences["reference"][key] == sequences["paired"][key]}
            for identity in sequences:
                value = sequences[identity][key]
                entry[identity] = {"encoding_bytes": len(value) * 8,
                                   "sha256": hashlib.sha256("".join(value).encode()).hexdigest()}
            instructions.append(entry)
        comparisons[version] = {"reference_count": len(reference), "paired_count": len(paired),
                                "missing": missing, "added": added, "resource_mismatches": mismatch,
                                "instruction_comparisons": instructions}

    symbols = run(["nm", "--defined-only", "-C", str(exe)], "host-symbols.txt")
    dynamic = run(["readelf", "-d", str(exe)], "dynamic-linking.txt")
    assembly = run(["objdump", "-dC", str(exe)], "host-disassembly.txt")
    api_symbols = {}
    for namespace in ("gh_old", "gh_new"):
        for function in ("histogram", "supported", "workspace_bytes"):
            matches = re.findall(r"^([0-9a-f]+) ([TW]) (" + re.escape(namespace + "::" + function) + r"\(.*)$", symbols, re.M)
            if len(matches) != 1:
                raise ValueError("Expected one strong backend function: " + namespace + "::" + function)
            api_symbols[namespace + "::" + function] = {"address": matches[0][0], "binding": matches[0][1], "symbol": matches[0][2]}
    for function in ("histogram", "supported", "workspace_bytes"):
        if api_symbols["gh_old::" + function]["address"] == api_symbols["gh_new::" + function]["address"]:
            raise ValueError("Backend functions unexpectedly share an address")
    blocks = re.split(r"(?m)^([0-9a-f]+) <(.+)>:\n", assembly)
    launch_blocks = []
    for address, name, body in zip(blocks[1::3], blocks[2::3], blocks[3::3]):
        if name.startswith("(anonymous namespace)::launch("):
            targets = sorted(set(re.findall(r"<(gh_(?:old|new)::histogram\([^>]+)>" , body)))
            launch_blocks.append({"address": address, "name": name, "targets": targets, "body": body})
    if len(launch_blocks) != 2 or sorted(target[:6] for block in launch_blocks for target in block["targets"]) != ["gh_new", "gh_old"]:
        raise ValueError("Adapters do not each target a distinct backend histogram")
    (BASE / "adapter-launch-disassembly.txt").write_text("\n".join(
        item["address"] + " <" + item["name"] + ">:\n" + item["body"] for item in launch_blocks))
    if "cub::" in symbols or "ghbench::" in symbols or "nvidia_sample" in symbols:
        raise ValueError("Unexpected benchmark-reference implementation in paired executable")
    if len(re.findall(r"NEEDED.*\[libcudart\.so", dynamic)) != 1:
        raise ValueError("Expected exactly one dynamic CUDA runtime dependency")
    if hashes != {name: sha(path) for name, path in paths.items()}:
        raise ValueError("An inspected binary changed")
    passed = all(not data["missing"] and not data["added"] and not data["resource_mismatches"] and
                 all(entry["identical"] for entry in data["instruction_comparisons"])
                 for data in comparisons.values())
    report = {
        "schema": 1, "gpu_activity": False, "passed": passed,
        "binaries": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
        "comparisons": comparisons, "host_symbols": api_symbols,
        "adapter_launch_targets": [{key: value for key, value in block.items() if key != "body"} for block in launch_blocks],
        "commands": commands,
        "normalization": "Match full template/parameter suffix after removing enclosing namespace and source anonymous-namespace prefix; compare complete resource records and all ordered 64-bit instruction/control words for three explicitly selected kernels.",
        "limits": ["Rebuilt source diagnostic, not identity of the preserved historical host executables.",
                   "Device instruction comparison covers only the two flagged counting kernels and u32 clear; other kernels receive resource comparisons.",
                   "Static matches and namespace isolation do not prove equal runtime performance."],
    }
    (BASE / "kernel-symbol-map.json").write_text(json.dumps(maps, indent=2) + "\n")
    (BASE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (BASE / "README.md").write_text(
        "# Paired-build static audit\n\n"
        f"Audit {'passed' if passed else 'found mismatches'}. Old/current namespaces retain "
        f"{comparisons['old']['paired_count']}/{comparisons['new']['paired_count']} device functions. "
        "See [comparison.json](comparison.json) for exact source-binary comparisons and hashes.\n\n"
        "Both host adapters target their own namespaced histogram function, and both backends share one dynamic CUDA runtime. "
        "No NVIDIA benchmark histogram symbols were found.\n\n"
        "Full instruction/control words are compared for the two flagged counting kernels and u32 clear, "
        "with resource records checked for every custom device function. These checks establish static preservation "
        "and backend isolation only; they do not establish equal runtime performance. No GPU queries or launches occurred.\n")
    print(json.dumps({"passed": passed, "binary": hashes["paired"], "comparisons": comparisons}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
