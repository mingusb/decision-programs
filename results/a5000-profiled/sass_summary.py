#!/usr/bin/env python3
"""Summarize/extract histogram kernels from cuobjdump --dump-sass output.

CPU only: consumes an existing text dump and calls c++filt to demangle names.
Example:
  python3 sass_summary.py all.sass --match shared_histogram \
      --match 'unsigned char, unsigned int, unsigned int, 256, 8, 1,' \
      --output byte-summary.md --extract byte-kernels.sass

Repeat --match for ANDed regular expressions against demangled names. JSON
includes full opcode counts and class-count deltas against a matching scalar
kernel. Static counts include all fallback/tail branches; they are not dynamic
instruction counts, register usage, timing measurements, or a hot-path trace.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


FUNCTION = re.compile(r"^\s*Function\s*:\s*(.+?)\s*$")
ARCH = re.compile(r"code for (sm_\d+)")
INSTRUCTION = re.compile(
    r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_.]*)\b[^;]*;"
)
KERNEL = re.compile(r"\b(shared_histogram(?:_loaded)?|bitplane_histogram(?:_full_tile)?)<(.+?)>\(")
TYPE_NAMES = {"unsigned char": "u8", "unsigned int": "u32", "unsigned long long": "u64"}
CLASS_NAMES = (
    "total", "ldg", "ldg_u8", "ldg_32", "ldg_64", "ldg_128", "shared_atomic",
    "shared_cas64", "global_atomic", "compare", "branch", "vote", "match", "popc", "local_memory",
)


def read_kernels(path):
    """Keep relevant function sections, preserving original instruction text."""
    current = None
    arch = "unknown"
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if match := ARCH.search(line):
                arch = match.group(1)
            if match := FUNCTION.match(line):
                if current:
                    yield current
                name = match.group(1)
                current = {"symbol": name, "arch": arch, "lines": []} if (
                    "shared_histogram" in name or "bitplane_histogram" in name
                ) else None
            elif current is not None:
                current["lines"].append(line)
    if current:
        yield current


def demangle(kernels, executable):
    if not kernels:
        return
    names = "\n".join(kernel["symbol"] for kernel in kernels) + "\n"
    result = subprocess.run([executable], input=names, text=True, capture_output=True, check=True)
    decoded = result.stdout.splitlines()
    if len(decoded) != len(kernels):
        raise ValueError("c++filt returned an unexpected number of names")
    for kernel, name in zip(kernels, decoded):
        kernel["demangled"] = name


def integer(token):
    # Handles enum casts such as '(gh::LoadPolicy)2' and size_t literals '49152ul'.
    match = re.search(r"(\d+)[uUlL]*$", token)
    if not match:
        raise ValueError(f"cannot parse template integer: {token}")
    return int(match.group(1))


def metadata(name):
    match = KERNEL.search(name)
    if not match:
        raise ValueError(f"unrecognized histogram template: {name}")
    kernel, arguments = match.groups()
    args = [arg.strip() for arg in arguments.split(",")]
    fields = {"input": TYPE_NAMES.get(args[0], args[0]), "counter": TYPE_NAMES.get(args[1], args[1])}
    if kernel.startswith("shared_histogram"):
        if len(args) not in (8, 9, 10):
            raise ValueError(f"unexpected shared template signature: {name}")
        fields.update(family="shared", local=TYPE_NAMES.get(args[2], args[2]),
                      threads=integer(args[3]), items=integer(args[4]), replicas=integer(args[5]),
                      update={0: "atomic", 1: "warp", 2: "rle"}[integer(args[6])],
                      partial=args[7] in ("true", "1"), capacity=None,
                      load="scalar" if kernel == "shared_histogram" else
                           {0: "scalar", 1: "full_tile", 2: "vector4"}[integer(args[8])],
                      shared_limit=integer(args[9]) if len(args) == 10 else 48 * 1024)
    else:
        if len(args) != 5:
            raise ValueError(f"unexpected bitplane template signature: {name}")
        fields.update(family="bitplane", local=fields["counter"], capacity=integer(args[2]),
                      threads=integer(args[3]), items=integer(args[4]), replicas=1, update="bitplane",
                      partial=False, load="full_tile" if kernel.endswith("_full_tile") else "scalar",
                      shared_limit=48 * 1024)
    return fields


def instruction_counts(lines):
    opcodes = Counter()
    for line in lines:
        if match := INSTRUCTION.match(line):
            opcodes[match.group(2)] += 1
    counts = dict.fromkeys(CLASS_NAMES, 0)
    counts["total"] = sum(opcodes.values())
    for opcode, count in opcodes.items():
        base = opcode.split(".")[0]
        if base == "LDG":
            counts["ldg"] += count
            parts = opcode.split(".")
            width = next((part for part in parts if part in ("U8", "S8", "U16", "S16", "64", "128")), "32")
            key = "ldg_u8" if width in ("U8", "S8") else "ldg_" + width
            if key in counts:
                counts[key] += count
        if base == "ATOMS":
            counts["shared_atomic"] += count
            if "CAST" in opcode and ".64" in opcode:
                counts["shared_cas64"] += count
        for key, condition in (
            ("global_atomic", base in ("ATOM", "RED")),
            ("compare", base in ("ISETP", "UISETP")),
            ("branch", base in ("BRA", "BRX", "JMP", "JMX")),
            ("vote", base == "VOTE"), ("match", base == "MATCH"),
            ("popc", base == "POPC"), ("local_memory", base in ("LDL", "STL")),
        ):
            if condition:
                counts[key] += count
    return dict(sorted(opcodes.items())), counts


def add_comparisons(kernels):
    def key(kernel):
        return (kernel["arch"], tuple(sorted((k, v) for k, v in kernel["policy"].items() if k != "load")))
    baselines = {}
    for kernel in kernels:
        if kernel["policy"]["load"] == "scalar":
            if key(kernel) in baselines:
                raise ValueError("multiple scalar kernels have the same comparison key; filter one architecture/build")
            baselines[key(kernel)] = kernel
    for kernel in kernels:
        baseline = baselines.get(key(kernel))
        kernel["scalar_baseline"] = baseline["symbol"] if baseline else None
        kernel["delta_vs_scalar"] = {
            name: kernel["classes"][name] - baseline["classes"][name] for name in CLASS_NAMES
        } if baseline else None


def resource_usage(path, executable):
    records = {}
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := re.search(r"Function properties for (\S+)", line):
            current = records.setdefault(match.group(1), {"symbol": match.group(1)})
        if current is not None:
            if match := re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line):
                current.update(zip(("stack_bytes", "spill_store_bytes", "spill_load_bytes"), map(int, match.groups())))
            if match := re.search(r"Used (\d+) registers", line):
                current["registers"] = int(match.group(1))
    values = list(records.values())
    demangle(values, executable)
    return values


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def markdown(report):
    lines = [f"Input: `{report['source']}`; SHA256 `{report['source_sha256']}`.", "",
             "Static instruction sites in complete function bodies, including fallback/tail paths. "
             "These counts do not measure executed instructions, spills, occupancy, or runtime.", "",
             "Scalar comparisons require equal architecture, types, dimensions, update method, partial mode, "
             "bin capacity, and shared-memory policy limit. A dash means no matching scalar kernel is present.", "",
             "| ID | Policy | Load | Registers | Spill store/load bytes | Total | Δtotal vs scalar | LDG8 | LDG32 | LDG64 | LDG128 | Shared atomic | CAS64 | Compare | Branch | Vote | Match | POPC | LDL/STL |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index, kernel in enumerate(report["kernels"], 1):
        p, c = kernel["policy"], kernel["classes"]
        label = (f"{kernel['arch']} {p['family']}/{p['update']} "
                 f"{p['input']}→{p['counter']} local={p['local']} "
                 f"t{p['threads']} i{p['items']} r{p['replicas']} cap={p['shared_limit']//1024}KiB")
        if p["partial"]:
            label += " partial"
        if p["capacity"]:
            label += f" bins≤{p['capacity']}"
        delta = kernel["delta_vs_scalar"]
        resources = kernel.get("ptxas", {})
        spills = f"{resources['spill_store_bytes']}/{resources['spill_load_bytes']}" if "spill_store_bytes" in resources else "—"
        values = [index, label, p["load"], resources.get("registers", "—"), spills,
                  c["total"], f"{delta['total']:+d}" if delta else "—"]
        values.extend(c[name] for name in ("ldg_u8", "ldg_32", "ldg_64", "ldg_128", "shared_atomic",
                                          "shared_cas64", "compare", "branch", "vote", "match", "popc", "local_memory"))
        lines.append("| " + " | ".join(map(str, values)) + " |")
    if "ptxas_log" in report:
        log = report["ptxas_log"]
        lines.extend(["", f"PTXAS log: `{log['source']}`; SHA256 `{log['sha256']}`. "
                      f"Parsed {log['function_count']} function records; "
                      f"{len(log['nonzero_spills'])} report nonzero spill stores or loads."])
    lines.extend(["", "Exact function identities:", ""])
    lines.extend(f"{index}. `{kernel['demangled']}`" for index, kernel in enumerate(report["kernels"], 1))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sass", type=Path, help="existing cuobjdump --dump-sass text")
    parser.add_argument("--match", action="append", default=[], help="ANDed regex against demangled function name")
    parser.add_argument("--arch", help="architecture filter, e.g. sm_86")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="summary destination; default stdout")
    parser.add_argument("--extract", type=Path, help="optional full SASS of selected functions")
    parser.add_argument("--ptxas-log", type=Path, help="optional verbose build log for exact register/spill counts")
    parser.add_argument("--cxxfilt", default="c++filt")
    options = parser.parse_args()
    filters = [re.compile(pattern) for pattern in options.match]
    kernels = list(read_kernels(options.sass))
    demangle(kernels, options.cxxfilt)
    kernels = [kernel for kernel in kernels
               if (not options.arch or kernel["arch"] == options.arch)
               and all(pattern.search(kernel["demangled"]) for pattern in filters)]
    if not kernels:
        parser.error("no histogram kernels matched the input and filters")
    for kernel in kernels:
        kernel["policy"] = metadata(kernel["demangled"])
        kernel["opcodes"], kernel["classes"] = instruction_counts(kernel["lines"])
        if not kernel["classes"]["total"]:
            raise ValueError(f"matched function has no parsed SASS instructions: {kernel['symbol']}")
    add_comparisons(kernels)
    kernels.sort(key=lambda kernel: (
        kernel["arch"], *(kernel["policy"][key] or 0 for key in
                         ("family", "input", "counter", "local", "threads", "items", "replicas",
                          "update", "partial", "capacity", "shared_limit")),
        {"scalar": 0, "full_tile": 1, "vector4": 2}[kernel["policy"]["load"]]))
    if options.extract:
        with options.extract.open("w", encoding="utf-8") as destination:
            for kernel in kernels:
                destination.write(f"// {kernel['arch']}: {kernel['demangled']}\nFunction : {kernel['symbol']}\n")
                destination.writelines(kernel["lines"])
                destination.write("\n")
    report = {"source": str(options.sass.resolve()), "source_sha256": file_sha256(options.sass),
              "interpretation": "Static whole-function instruction sites, including fallback/tail paths; not execution counts.",
              "kernels": [{k: v for k, v in kernel.items() if k != "lines"} for kernel in kernels]}
    if options.ptxas_log:
        records = resource_usage(options.ptxas_log, options.cxxfilt)
        by_name = {record["demangled"]: record for record in records}
        for kernel in report["kernels"]:
            if kernel["demangled"] in by_name:
                kernel["ptxas"] = by_name[kernel["demangled"]]
        report["ptxas_log"] = {
            "source": str(options.ptxas_log.resolve()), "sha256": file_sha256(options.ptxas_log),
            "function_count": len(records), "nonzero_spills": [record for record in records
                 if record.get("spill_store_bytes", 0) or record.get("spill_load_bytes", 0)]}
    content = json.dumps(report, indent=2) + "\n" if options.format == "json" else markdown(report)
    if options.output:
        options.output.write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
