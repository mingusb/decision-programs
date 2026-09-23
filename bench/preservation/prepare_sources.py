#!/usr/bin/env python3
"""Validate and stage the two frozen source archives, without touching the GPU."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


BACKENDS = (
    ("old", "results/a5000-custom-only/source-manifest.json", "gh_old"),
    ("new", "results/a5000-large-bins/final-source-manifest.json", "gh_new"),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or not path.parts or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe or noncanonical archive path: {name!r}")
    return name


def read_archive(root, label, manifest_relative, namespace):
    manifest_path = root / manifest_relative
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    archive_relative = safe_name(manifest["archive"])
    archive_path = root / archive_relative
    archive_data = archive_path.read_bytes()
    actual_archive_hash = sha256(archive_data)
    if actual_archive_hash != manifest["archive_sha256"]:
        raise ValueError(f"Archive SHA256 mismatch: {archive_path}")

    expected = {}
    for entry in manifest["files"]:
        name = safe_name(entry["path"])
        if name in expected:
            raise ValueError(f"Duplicate manifest path: {name}")
        expected[name] = entry["sha256"]
    contents = {}
    # Parse the exact byte sequence just hashed, even if a source file is later replaced.
    with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = safe_name(member.name)
            if not member.isfile():
                raise ValueError(f"Only regular archived files are accepted: {name}")
            if name in contents or name not in expected:
                raise ValueError(f"Duplicate or undeclared archive member: {name}")
            with archive.extractfile(member) as stream:
                data = stream.read()
            if sha256(data) != expected[name]:
                raise ValueError(f"Archived source SHA256 mismatch: {name}")
            contents[name] = data
    if set(contents) != set(expected):
        raise ValueError(f"Missing archive members: {sorted(set(expected) - set(contents))}")
    provenance = {
        "backend": label,
        "namespace": namespace,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_data),
        "archive": str(archive_path),
        "archive_sha256": actual_archive_hash,
        "files": [{"path": name, "sha256": expected[name]} for name in sorted(expected)],
    }
    return contents, provenance


def stage_archive(destination, contents):
    if destination.is_symlink():
        raise ValueError(f"Source directory must not be a symlink: {destination}")
    if destination.exists():
        actual = set()
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Extracted sources contain a symlink: {path}")
            if path.is_file():
                actual.add(path.relative_to(destination).as_posix())
            elif not path.is_dir():
                raise ValueError(f"Unexpected extracted source entry: {path}")
        if actual != set(contents):
            raise ValueError(f"Existing source directory has a different file set: {destination}")
        for name, data in contents.items():
            if (destination / name).read_bytes() != data:
                raise ValueError(f"Existing extracted source has changed: {destination / name}")
        return
    destination.mkdir(parents=True)
    for name, data in contents.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    output = args.output_dir.absolute()
    if output.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {output}")

    # Validate both inputs completely before staging either one.
    validated = [read_archive(root, *backend) for backend in BACKENDS]
    provenance = {
        "schema": 1,
        "purpose": "Same-process comparison of exact archived custom histogram sources",
        "repo_root": str(root),
        "source_directory": str(output),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256(Path(__file__).read_bytes()),
        "backends": [],
    }
    for contents, entry in validated:
        destination = output / entry["backend"]
        stage_archive(destination, contents)
        entry["source_directory"] = str(destination)
        provenance["backends"].append(entry)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated and staged {sum(len(c) for c, _ in validated)} archived files in {output}")


if __name__ == "__main__":
    main()
