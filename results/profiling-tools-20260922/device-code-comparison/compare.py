#!/usr/bin/env python3
"""CPU-only comparison of every extracted CUDA ELF executable text section."""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOL = Path("/usr/local/cuda/bin/cuobjdump")
CASES = {
    "count-original": ROOT / "build/window-experiment/libgh.a",
    "count-profile": ROOT / "build/profiling-count/libgh.a",
    "booster-original": ROOT / "build/booster-higher-order/libghb.a",
    "booster-profile": ROOT / "build/profiling-booster/libghb.a",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            value.update(part)
    return value.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def extract_elf(path):
    raw = path.read_bytes()
    if raw[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected little-endian ELF64 CUDA cubin: {path}")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    machine, section_offset, section_entry_size, section_count, names_index = header[2], header[6], header[11], header[12], header[13]
    if machine != 190 or section_entry_size != 64 or not section_count or names_index >= section_count:
        raise ValueError("Unsupported ELF machine or extended section numbering")
    if section_offset + section_entry_size * section_count > len(raw):
        raise ValueError("Truncated ELF section headers")
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, section_offset + i * section_entry_size)
                for i in range(section_count)]
    names_header = sections[names_index]
    names = raw[names_header[4]:names_header[4] + names_header[5]]
    architecture = re.search(r"sm_([0-9]+[a-z]?)", path.name)
    if architecture is None:
        raise ValueError(f"Missing architecture in extracted filename: {path.name}")
    result = []
    for section in sections:
        name_at, kind, flags, address, offset, size, link, info, alignment, entry_size = section
        if name_at >= len(names):
            raise ValueError("Invalid section name extent")
        end = names.index(0, name_at)
        name = names[name_at:end].decode("utf-8")
        if not (name == ".text" or name.startswith(".text.")):
            continue
        if kind != 1 or not (flags & 4) or offset + size > len(raw):
            raise ValueError(f"Unexpected executable text section: {name}")
        result.append({"cubin": path.name, "architecture": "sm_" + architecture.group(1),
                       "section": name, "offset": offset, "bytes": size,
                       "sha256": sha(raw[offset:offset + size]),
                       "elf_flags": flags, "elf_info": info, "alignment": alignment})
    if not result:
        raise ValueError(f"No executable CUDA text found in {path}")
    return {"file": path.name, "file_sha256": sha(raw), "file_bytes": len(raw),
            "architecture": "sm_" + architecture.group(1), "text": result}


def capture(name, library):
    destination = HERE / name
    destination.mkdir()
    before = file_sha(library)
    command = [str(TOOL), "--extract-elf", "all", str(library)]
    record = {"command": command, "cwd": str(destination), "library": str(library),
              "library_sha256_before": before, "tool_sha256": file_sha(TOOL),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(destination / "command.json", record)
    started = time.perf_counter()
    with (destination / "stdout").open("x") as out, (destination / "stderr").open("x") as err:
        process = subprocess.run(command, cwd=destination, stdout=out, stderr=err, check=False)
    record.update(returncode=process.returncode, wall_seconds=time.perf_counter() - started,
                  library_sha256_after=file_sha(library), stdout_sha256=file_sha(destination / "stdout"),
                  stderr_sha256=file_sha(destination / "stderr"))
    write(destination / "receipt.json", record)
    if process.returncode or before != record["library_sha256_after"]:
        raise RuntimeError(f"Extraction failed or archive changed: {name}")
    cubins = sorted(destination.glob("*.cubin"))
    if not cubins:
        raise RuntimeError(f"No CUDA ELFs extracted: {name}")
    parsed = [extract_elf(path) for path in cubins]
    snapshots = {}
    for filename in ("CMakeCache.txt", "build.ninja"):
        source = library.parent / filename
        if source.exists():
            target = destination / filename
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
            snapshots[filename] = file_sha(target)
    text = [section for cubin in parsed for section in cubin["text"]]
    result = {"library": str(library), "library_sha256": before, "cubins": parsed,
              "cubin_count": len(parsed), "text_section_count": len(text),
              "text_bytes": sum(s["bytes"] for s in text), "build_snapshot_sha256": snapshots}
    write(destination / "index.json", result)
    return result


def compare(original, profile):
    def group(value):
        result = collections.defaultdict(list)
        for cubin in value["cubins"]:
            for text in cubin["text"]:
                result[(text["architecture"], text["section"])].append(text)
        return result
    old, new = group(original), group(profile)
    differences, matched = [], 0
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity, []), new.get(identity, [])
        a_bytes = sorted((x["bytes"], x["sha256"]) for x in a)
        b_bytes = sorted((x["bytes"], x["sha256"]) for x in b)
        if a_bytes == b_bytes:
            matched += len(a)
        else:
            differences.append({"architecture": identity[0], "section": identity[1], "original": a, "profile": b})
    return {"all_executable_device_text_bytes_equal": not differences,
            "matched_text_sections": matched, "differing_function_identities": len(differences),
            "original_cubins": original["cubin_count"], "profile_cubins": profile["cubin_count"],
            "original_text_sections": original["text_section_count"], "profile_text_sections": profile["text_section_count"],
            "original_text_bytes": original["text_bytes"], "profile_text_bytes": profile["text_bytes"],
            "original_library_sha256": original["library_sha256"], "profile_library_sha256": profile["library_sha256"],
            "differences": differences}


def main():
    (HERE / ".incomplete").touch(exist_ok=False)
    version = subprocess.run([str(TOOL), "--version"], text=True, capture_output=True, check=False)
    write(HERE / "tool-version.json", {"command": [str(TOOL), "--version"], "returncode": version.returncode,
                                       "stdout": version.stdout, "stderr": version.stderr, "tool_sha256": file_sha(TOOL)})
    if version.returncode:
        raise RuntimeError("cuobjdump version failed")
    values = {name: capture(name, path) for name, path in CASES.items()}
    comparisons = {name: compare(values[name + "-original"], values[name + "-profile"])
                   for name in ("count", "booster")}
    result = {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "method": "Extract all CUDA ELF cubins; parse ELF64 section tables; compare byte-size/SHA256 multisets of every executable .text section grouped by exact mangled function name and architecture. No metadata/whitespace normalization of instruction bytes.",
              "scope": "SASS machine instruction sections only. Host code/debug info and complete cubin/archive hashes may differ. This does not independently certify constant-data/resource metadata, source semantics or performance equivalence.",
              "gpu_work_executed": False, "script_sha256": file_sha(Path(__file__)), "comparisons": comparisons}
    write(HERE / "comparison.json", result)
    lines = ["# Compiled device instruction comparison", "", result["method"], "", result["scope"], "",
             "| Library | Original / profile cubins | Original / profile text sections | Matched sections | Text bytes, original / profile | All device instruction bytes equal |",
             "|---|---:|---:|---:|---:|---|"]
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['original_cubins']} / {value['profile_cubins']} | {value['original_text_sections']} / {value['profile_text_sections']} | {value['matched_text_sections']} | {value['original_text_bytes']} / {value['profile_text_bytes']} | {value['all_executable_device_text_bytes_equal']} |")
    lines += ["", "Exact library/tool/build snapshot hashes, commands, return codes, raw CUDA ELFs, every function hash and any differences are retained in comparison.json and the four extraction directories. No GPU workload or production source edit was performed.", ""]
    with (HERE / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines))
    (HERE / ".incomplete").unlink()
    print(json.dumps({name: {key: value for key, value in result.items() if key != "differences"}
                      for name, result in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
