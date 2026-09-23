"""Seal completed evidence without overwriting raw observations or previous seals."""
from pathlib import Path
import hashlib,json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""): value.update(block)
    return value.hexdigest()

manifest = OUT / "artifacts.sha256"
receipt = OUT / "seal-verification.json"
if manifest.exists() or receipt.exists(): raise SystemExit("refusing to overwrite evidence seal")
for name in ("REPORT.md", "QUALITY.md", "SYSTEMS.md", "COMPUTE.md", "deeper-results.md",
             "root-split-primitives.md", "metadata-verification.json", "trace-audit.json"):
    if not (OUT / name).is_file(): raise SystemExit("required artifact missing: " + name)
if list(OUT.rglob(".incomplete")): raise SystemExit("incomplete evidence remains")
frozen = OUT / "final-provenance"
for line in (frozen / "manifest.sha256").read_text().splitlines():
    digest, relative = line.split("  ", 1)
    if sha(frozen / relative) != digest or sha(ROOT / relative) != digest:
        raise SystemExit("current/frozen artifact changed: " + relative)
allowed_failures = {"configure-command.json", "quality-audit-command.json", "previous-repeat-controls-command.json"}
for path in OUT.glob("*-command.json"):
    record = json.loads(path.read_text())
    if record["returncode"] and path.name not in allowed_failures:
        raise SystemExit("unexpected diagnostic failure: " + path.name)
entries = []
for path in sorted(OUT.rglob("*")):
    if path.is_file() and "__pycache__" not in path.parts:
        entries.append((sha(path), path.relative_to(ROOT)))
with manifest.open("x") as target:
    for digest, relative in entries: target.write(f"{digest}  {relative}\n")
for digest, relative in entries:
    if sha(ROOT / relative) != digest: raise SystemExit("artifact changed during sealing: " + str(relative))
value = dict(status="verified", artifacts=len(entries), manifest_sha256=sha(manifest),
             quality_status=json.loads((OUT / "quality/summary.json").read_text())["status"],
             note="Manifest excludes itself, this receipt, and Python bytecode caches.")
with receipt.open("x") as target: json.dump(value,target,indent=2);target.write("\n")
print(json.dumps(value))
