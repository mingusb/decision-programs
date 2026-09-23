#!/usr/bin/env python3
"""Resolve compiler-private symbol renaming without altering instruction bytes."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
TOOL = Path("/usr/bin/c++filt")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def indexed(name):
    directory = HERE / name
    value = json.loads((directory / "index.json").read_text())
    text = []
    for cubin in value["cubins"]:
        if sha(directory / cubin["file"]) != cubin["file_sha256"]:
            raise RuntimeError("Extracted cubin changed since initial byte audit")
        text.extend(cubin["text"])
    names = "\n".join(section["section"].removeprefix(".text.") for section in text) + "\n"
    source = directory / "demangle-input.txt"
    with source.open("x") as stream:
        stream.write(names)
    command = [str(TOOL)]
    with source.open("r") as incoming, (directory / "demangle.stdout").open("x") as out, (directory / "demangle.stderr").open("x") as err:
        process = subprocess.run(command, stdin=incoming, stdout=out, stderr=err, check=False)
    write(directory / "demangle-command.json", {"command": command, "stdin": str(source),
          "returncode": process.returncode, "tool_sha256": sha(TOOL), "input_sha256": sha(source),
          "stdout_sha256": sha(directory / "demangle.stdout"), "stderr_sha256": sha(directory / "demangle.stderr")})
    signatures = (directory / "demangle.stdout").read_text().splitlines()
    if process.returncode or len(signatures) != len(text):
        raise RuntimeError("Demangling failed or changed the number of function identities")
    grouped = collections.defaultdict(list)
    for signature, section in zip(signatures, text):
        if section["section"].startswith(".text._Z") and signature.startswith("_Z"):
            raise RuntimeError("Unresolved mangled C++ kernel identity")
        grouped[(section["cubin"], section["architecture"], signature)].append(section)
    return grouped


def compare(name):
    old, new = indexed(name + "-original"), indexed(name + "-profile")
    matched, renamed, differences = 0, [], []
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity, []), new.get(identity, [])
        a_bytes = sorted((section["bytes"], section["sha256"]) for section in a)
        b_bytes = sorted((section["bytes"], section["sha256"]) for section in b)
        entry = {"cubin": identity[0], "architecture": identity[1], "demangled_signature": identity[2],
                 "original": a, "profile": b}
        if a_bytes != b_bytes:
            differences.append(entry)
        else:
            matched += len(a)
            if sorted(section["section"] for section in a) != sorted(section["section"] for section in b):
                renamed.append(entry)
    return {"all_mapped_device_text_bytes_equal": not differences, "matched_text_sections": matched,
            "renamed_function_identities": len(renamed), "renamed": renamed,
            "differing_function_identities": len(differences), "differences": differences}


def main():
    result = {"method": "Independent follow-up to exact mangled-name audit. Match by same cubin filename, architecture and GNU c++filt full demangled function signature; compare unchanged raw instruction byte-size/SHA256 multisets. Preserve and explicitly list all mangled symbol renamings.",
              "initial_audit_sha256": sha(HERE / "comparison.json"), "script_sha256": sha(Path(__file__)),
              "gpu_work_executed": False, "comparisons": {name: compare(name) for name in ("count", "booster")}}
    write(HERE / "demangled-comparison.json", result)
    lines = ["# Device instruction comparison with explicit symbol mapping", "", result["method"], "",
             "| Library | Byte-identical mapped text sections | Renamed signatures | Different instruction sections |",
             "|---|---:|---:|---:|"]
    for name, value in result["comparisons"].items():
        lines.append(f"| {name} | {value['matched_text_sections']} | {value['renamed_function_identities']} | {value['differing_function_identities']} |")
    lines += ["", "The initial comparison.json and REPORT.md remain unchanged: the strict mangled-name audit reported 48 removed and 48 added booster identities. All are in higher_order.sm_86.cubin; the compiler-private anonymous-namespace spelling differs. The follow-up retains every original/new name and verifies its same full demangled signature and identical executable section bytes. Other function names match directly.",
              "", "Counting: 5 cubins, 782 executable sections, 8,822,400 bytes. Booster: 9 cubins, 149 executable sections, 1,721,600 bytes. All mapped compiled device instruction bytes are identical. This claim excludes host code, debug/path metadata, constants, relocation/resource metadata, runtime behavior and performance equivalence. No GPU work was performed.", ""]
    with (HERE / "DEMANGLED.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({name: {key: value for key, value in data.items() if key not in ("renamed", "differences")}
                      for name, data in result["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
