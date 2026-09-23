#!/usr/bin/env python3
"""CPU-only comparison of existing CUDA kernel resources and selected SASS words."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
BINARIES = {"old": ROOT / "build/custom-only/histogram_bench",
            "new": ROOT / "build/large-bins/histogram_bench"}
LEGACY = ("shared_histogram_loaded", "shared_histogram", "global_histogram",
          "clear_histogram_output", "reduce_partials", "bitplane_histogram_full_tile", "bitplane_histogram")
NARROW = ("global_narrow_histogram", "clear_narrow_counts", "widen_counts")
SELECTED = [
    "clear_histogram_outputIjEEvPT_j",
    "clear_histogram_outputIyEEvPT_j",
    "shared_histogram_loadedIjyyLi512ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm49152EEEvPKT_mjPT0_PT1_",
    "shared_histogram_loadedIjyjLi512ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm49152EEEvPKT_mjPT0_PT1_",
    "shared_histogram_loadedIjyjLi512ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm98304EEEvPKT_mjPT0_PT1_",
    "shared_histogram_loadedIjjjLi1024ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm49152EEEvPKT_mjPT0_PT1_",
    "shared_histogram_loadedIhjjLi512ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm49152EEEvPKT_mjPT0_PT1_",
    "shared_histogram_loadedIhjjLi1024ELi8ELi1ELNS0_6UpdateE0ELb0ELNS_10LoadPolicyE2ELm49152EEEvPKT_mjPT0_PT1_",
    "shared_histogramIjjjLi256ELi8ELi1ELNS0_6UpdateE0ELb0EEEvPKT_mjPT0_PT1_",
    "shared_histogramIjyjLi256ELi8ELi1ELNS0_6UpdateE0ELb1EEEvPKT_mjPT0_PT1_",
    "global_histogramIjyLi256ELi8ELNS0_6UpdateE0EEEvPKT_mPT0_",
    "bitplane_histogramIhjLj128ELi256ELi8EEEvPKT_mPT0_j",
    "bitplane_histogram_full_tileIhjLj128ELi256ELi8EEEvPKT_mPT0_j",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_key(symbol):
    for basename in (*LEGACY, *NARROW):
        position = symbol.find(str(len(basename)) + basename)
        if position >= 0:
            return symbol[position + len(str(len(basename))):], basename
    raise ValueError("unrecognized custom device symbol: " + symbol)


def resources(text):
    result = {}
    for symbol, resource_line in re.findall(r"^\s*Function (\S+):\n[ \t]+([^\n]+)", text, re.M):
        if not symbol.startswith("_ZN2gh"):
            continue
        key, family = kernel_key(symbol)
        if key in result:
            raise ValueError("duplicate normalized resource key: " + key)
        result[key] = {"symbol": symbol, "family": family, "resources": resource_line.strip()}
    return result


def encodings(text):
    entries = re.split(r"\bFunction\s*:\s*(\S+)", text)
    result = {}
    for symbol, body in zip(entries[1::2], entries[2::2]):
        key, _ = kernel_key(symbol)
        words = re.findall(r"/\* (0x[0-9a-f]{16}) \*/", body)
        if not words or key in result:
            raise ValueError("empty or duplicate selected encoding: " + key)
        result[key] = words
    if set(result) != set(SELECTED):
        raise ValueError("selected SASS dump does not contain exactly the declared 13 functions")
    return result


def main():
    hashes = {name: sha256(path) for name, path in BINARIES.items()}
    commands, maps, words = [], {}, {}
    for name, binary in BINARIES.items():
        resource_command = [CUOBJDUMP, "--gpu-architecture", "sm_86", "--dump-resource-usage", str(binary)]
        commands.append(resource_command)
        text = subprocess.run(resource_command, check=True, text=True, capture_output=True).stdout
        (BASE / f"{name}-resources.txt").write_text(text)
        maps[name] = resources(text)
        symbols = [maps[name][key]["symbol"] for key in SELECTED]
        sass_command = [CUOBJDUMP, "--gpu-architecture", "sm_86", "--dump-sass", "--function",
                        ",".join(symbols), str(binary)]
        commands.append(sass_command)
        text = subprocess.run(sass_command, check=True, text=True, capture_output=True).stdout
        (BASE / f"{name}-selected.sass").write_text(text)
        words[name] = encodings(text)
    if hashes != {name: sha256(path) for name, path in BINARIES.items()}:
        raise ValueError("a binary changed during CPU inspection")
    old_keys, new_keys = set(maps["old"]), set(maps["new"])
    shared = sorted(old_keys & new_keys)
    resource_mismatches = [{"kernel": key, "old": maps["old"][key]["resources"],
                            "new": maps["new"][key]["resources"]}
                           for key in shared if maps["old"][key]["resources"] != maps["new"][key]["resources"]]
    additions = {key: maps["new"][key] for key in sorted(new_keys - old_keys)}
    selected = []
    for key in SELECTED:
        item = {"kernel": key, "identical": words["old"][key] == words["new"][key]}
        for name in BINARIES:
            sequence = words[name][key]
            item[name] = {"encoding_bytes": len(sequence) * 8,
                          "encoding_words_sha256": hashlib.sha256("".join(sequence).encode()).hexdigest()}
        selected.append(item)
    report = {
        "schema": 1, "gpu_execution_performed": False,
        "binaries": {name: {"path": str(path), "sha256": hashes[name]} for name, path in BINARIES.items()},
        "old_custom_function_count": len(old_keys), "new_custom_function_count": len(new_keys),
        "resource_comparisons": len(shared), "resource_mismatches": resource_mismatches,
        "removed_functions": sorted(old_keys - new_keys), "added_functions": additions,
        "added_function_families": dict(collections.Counter(item["family"] for item in additions.values())),
        "old_function_families": dict(collections.Counter(item["family"] for item in maps["old"].values())),
        "instruction_comparisons": selected, "commands": commands,
        "tool_version": subprocess.run([CUOBJDUMP, "--version"], check=True, text=True, capture_output=True).stdout.strip(),
        "normalization": {
            "resources": "Match kernel basename plus its entire remaining template/parameter mangling; "
                         "drop only the enclosing namespace/translation-unit anonymous-namespace prefix. "
                         "Compare each complete resource line after trimming surrounding whitespace. "
                         "Normalized names must be unique in each binary.",
            "sass": "For each of the 13 exactly selected mangled functions, compare every emitted "
                    "16-hex-digit instruction/control encoding word in order. Ignore textual module "
                    "headers, symbol prefixes, addresses and instruction spelling. Encoding hashes "
                    "are SHA256 of concatenated literal 0x-prefixed word strings; each word is 8 bytes.",
        },
        "interpretation": [
            "Only the selected 13 functions receive full ordered instruction-word comparison.",
            "Matching static resources or encodings does not prove equal runtime performance.",
            "Host dispatch, workspace behavior and launch sequences differ for the new narrow path; "
            "they are outside this device-code inspection.",
            "NVIDIA benchmark-reference and cache-eviction functions are preserved in the complete "
            "resource dumps but excluded from custom-kernel comparison counts.",
        ],
    }
    (BASE / "kernel-symbol-map.json").write_text(json.dumps(maps, indent=2) + "\n")
    (BASE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    matched_sass = sum(item["identical"] for item in selected)
    text = f"""# CPU inspection of the narrow-global build

Compared `{BINARIES['old'].relative_to(ROOT)}` with `{BINARIES['new'].relative_to(ROOT)}`.

- Old custom device functions: **{len(old_keys)}**; new: **{len(new_keys)}**.
- Matching function identities: **{len(shared)}**; resource-record differences: **{len(resource_mismatches)}**.
- Removed old functions: **{len(old_keys - new_keys)}**; added functions: **{len(additions)}**.
- Selected full instruction-word sequences matching: **{matched_sass}/{len(SELECTED)}**.

The 13 selected kernels cover both output clears, scalar and vector shared histograms,
native/u32-local u64 output, the 96 KiB specialization, u8/u32 wide blocks,
partial construction, global accumulation, and scalar/full-tile bitplanes.
The other matching functions have resource comparisons only, including the three partial-reduction specializations.
New narrow-global functions are listed separately in [comparison.json](comparison.json).

Exact binary SHA256 hashes, every selected kernel name and encoding hash, normalization
rules, tool version and commands are in that JSON. Complete resource dumps and focused
SASS dumps are retained. Reproduce this CPU-only inspection with `python3 {Path(__file__).relative_to(ROOT)}`.

This is static preservation evidence, **not proof of equal runtime performance**.
It does not inspect host dispatch or measure execution. No GPU query or launch was performed.
"""
    (BASE / "README.md").write_text(text)
    artifact_hashes = {path.name: sha256(path) for path in sorted(BASE.iterdir())
                       if path.is_file() and path.name != "artifact-hashes.json"}
    (BASE / "artifact-hashes.json").write_text(json.dumps(artifact_hashes, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("binaries", "old_custom_function_count",
                     "new_custom_function_count", "resource_comparisons", "resource_mismatches",
                     "removed_functions", "added_function_families")}, indent=2))
    print(f"Identical selected instruction sequences: {matched_sass}/{len(SELECTED)}")


if __name__ == "__main__":
    main()
