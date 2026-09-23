#!/usr/bin/env python3
"""CPU-only final snapshot/seal; run once after all evidence and documents close."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = HERE / "final-provenance"
BUILD = ROOT / "build/booster-level-batch"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eligible(path):
    return path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def manifest(path, files):
    observations = [(digest(p), p.relative_to(ROOT).as_posix()) for p in sorted(files)]
    with path.open("x") as stream:
        for sha, name in observations:
            stream.write(f"{sha}  {name}\n")
    failures = [name for sha, name in observations if digest(ROOT / name) != sha]
    if failures:
        raise RuntimeError(f"files changed during seal: {failures}")
    return len(observations)


def main():
    if any(HERE.rglob(".incomplete")):
        raise RuntimeError("an evidence campaign is still incomplete")
    captures = [json.loads(p.read_text()) for p in HERE.glob("synthetic-*-capture.json")]
    if len(captures) != 56 or any(c["returncode"] or not c["executable_unchanged"] for c in captures):
        raise RuntimeError("synthetic campaign identity/completion mismatch")
    if {c["executable_sha256"] for c in captures} != {digest(BUILD / "ghb_bench")}:
        raise RuntimeError("production benchmark differs from measured executable")
    for name, expected in (("validation", 80), ("test", 60)):
        receipt = json.loads((HERE / "real" / name / "summary.json").read_text())
        if receipt["jobs"] != expected or receipt["passed"] != expected or receipt["failed"]:
            raise RuntimeError(f"incomplete or failed real-data {name} campaign")
    for path in (HERE / "artifacts.sha256", HERE / "seal-verification.json", FROZEN):
        if path.exists():
            raise FileExistsError(path)
    FROZEN.mkdir()
    files = [ROOT / "AGENTS.md", ROOT / "CMakeLists.txt"]
    for folder in ("training", "src", "include"):
        files += [p for p in (ROOT / folder).rglob("*") if eligible(p)]
    files += [p for p in BUILD.iterdir() if eligible(p) and
              (p.name.startswith("ghb_") or p.suffix == ".a" or p.name in
               ("CMakeCache.txt", "build.ninja", "CTestTestfile.cmake"))]
    for source in files:
        target = FROZEN / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"snapshot copy differs: {source}")
    frozen_count = manifest(FROZEN / "manifest.sha256", [p for p in FROZEN.rglob("*") if eligible(p)])
    artifact_count = manifest(HERE / "artifacts.sha256", [p for p in HERE.rglob("*") if eligible(p)])
    receipt = {"status": "verified", "source_build_files": frozen_count, "artifacts": artifact_count,
               "source_build_manifest_sha256": digest(FROZEN / "manifest.sha256"),
               "artifact_manifest_sha256": digest(HERE / "artifacts.sha256"),
               "excluded": ["__pycache__", "*.pyc", "artifacts.sha256 itself", "seal-verification.json itself"],
               "production_bench_sha256": digest(BUILD / "ghb_bench"),
               "real_bench_sha256": digest(BUILD / "ghb_real_bench"),
               "quality_status": "strict failures retained; see REPORT.md and quality audits",
               "promotion": "output_batch remains opt-in; preceding root/count defaults preserved"}
    (HERE / "seal-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
