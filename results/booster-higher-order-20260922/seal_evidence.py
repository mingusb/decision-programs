#!/usr/bin/env python3
"""Seal completed observations after all writers and GPU jobs have finished."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
WORKSPACE = EVIDENCE.parents[1]
EXCLUDED = {"artifact-manifest.json", "seal-verification.json"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def artifacts():
    return {str(p.relative_to(EVIDENCE)): digest(p)
            for p in sorted(EVIDENCE.rglob("*"))
            if p.is_file() and p.name not in EXCLUDED and "__pycache__" not in p.parts}


def main():
    audit = read(EVIDENCE / "loss-audit.json")
    assert audit["status"] == "passed" and not audit["errors"] and not audit["pending"]
    assert audit["observed_loss_curves_including_untracked_diagnostics"] == 111
    assert read(EVIDENCE / "profiles-v2/summary.json")["complete"] is True
    assert read(EVIDENCE / "profiles/summary.json")["complete"] is False
    frozen = read(EVIDENCE / "production-provenance/source-sha256.json")
    assert len(frozen) == 47
    assert all(digest(WORKSPACE / name) == expected for name, expected in frozen.items())
    before = read(EVIDENCE / "prechange/source-sha256.json")
    counts = {name: value for name, value in before.items()
              if name == "CMakeLists.txt" or name.startswith(("src/", "include/"))}
    assert len(counts) == 9
    assert all(digest(WORKSPACE / name) == expected for name, expected in counts.items())
    binaries = read(EVIDENCE / "production-provenance/binary-sha256.json")
    assert all(digest(EVIDENCE / "bin" / name) == expected for name, expected in binaries.items())
    for name in ("systems-analysis.json", "systems-analysis.md", "compute-analysis.json", "compute-analysis.md"):
        assert (EVIDENCE / name).is_file(), name
    related = ["training/README.md", "training/ALGORITHM_DECISIONS.md",
               "training/HIGHER_ORDER_EXPERIMENT.md", "training/tools/higher_order_campaign.py"]
    workspace_hashes = {name: digest(WORKSPACE / name) for name in related}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "All retained experiment files except this manifest, its verification receipt and Python bytecode caches; historical failures included.",
                "artifacts_sha256": artifacts(), "related_workspace_sha256": workspace_hashes}
    write_new(EVIDENCE / "artifact-manifest.json", manifest)
    assert artifacts() == manifest["artifacts_sha256"]
    assert all(digest(WORKSPACE / p) == h for p, h in workspace_hashes.items())
    write_new(EVIDENCE / "seal-verification.json", {
        "verified_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "manifest_sha256": digest(EVIDENCE / "artifact-manifest.json"),
        "retained_artifact_count": len(manifest["artifacts_sha256"]),
        "all_retained_artifact_hashes_rechecked": True,
        "related_workspace_files_rechecked": len(workspace_hashes),
        "frozen_production_sources_rechecked": len(frozen),
        "frozen_count_sources_rechecked": len(counts),
        "frozen_binaries_rechecked": len(binaries),
        "loss_audit_status": audit["status"],
        "historical_profiler_failure_preserved": True,
        "quality_regressions_remain_failures": True,
        "limits": "File identity and captured evidence verification; not a universal performance or generalization certificate."})
    print(json.dumps({"passed": True, "artifacts": len(manifest["artifacts_sha256"])}))


if __name__ == "__main__":
    main()
