#!/usr/bin/env python3
"""Offline experiment fixtures and quality references; never production training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import numpy as np

MAGIC = b"GHBDS001"
HEADER = struct.Struct("<8s6I")
ROOT = Path(__file__).resolve().parents[1]
SEED = 2026092207

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated fixture header")
        magic, version, rows, features, targets, objective, classes = HEADER.unpack(header)
        if magic != MAGIC or version != 1 or not rows or not features or not targets or objective > 2:
            raise ValueError("invalid fixture header")
        if objective == 2 and (targets != 1 or classes < 2):
            raise ValueError("invalid multiclass fixture header")
        if path.stat().st_size != HEADER.size + 4 * rows * (features + targets):
            raise ValueError("fixture extent mismatch")
        x = np.fromfile(stream, dtype="<f4", count=rows * features).reshape(rows, features)
        y = np.fromfile(stream, dtype="<f4", count=rows * targets).reshape(rows, targets)
    validate(x, y, objective, classes)
    return x, y, {"rows": rows, "features": features, "targets": targets,
                  "objective": objective, "classes": classes}

def validate(x, y, objective, classes):
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid array shapes")
    if np.isinf(x).any() or not np.isfinite(y).all():
        raise ValueError("invalid feature/target finite contract")
    if objective == 1 and not np.isin(y, [0, 1]).all():
        raise ValueError("binary target domain mismatch")
    if objective == 2 and ((y < 0).any() or (y >= classes).any() or (y != np.floor(y)).any()):
        raise ValueError("class target domain mismatch")

def write(path, x, y, objective, classes):
    x, y = np.asarray(x, dtype="<f4", order="C"), np.asarray(y, dtype="<f4", order="C")
    validate(x, y, objective, classes)
    with Path(path).open("xb") as stream:
        stream.write(HEADER.pack(MAGIC, 1, x.shape[0], x.shape[1], y.shape[1], objective, classes))
        x.tofile(stream)
        y.tofile(stream)

def stratified(indices, y, validation_fraction, rng):
    training, validation = [], []
    for label in np.unique(y[indices]):
        group = indices[y[indices] == label].copy()
        rng.shuffle(group)
        cut = max(1, int(len(group) * validation_fraction))
        validation.extend(group[:cut])
        training.extend(group[cut:])
    return np.sort(training), np.sort(validation)

def sparse_arff(path, feature_count, label_names):
    attributes, rows = [], []
    data = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if not data:
            if line.lower().startswith("@attribute "):
                fields = line.split(maxsplit=2)
                attributes.append(fields[1].strip("'\""))
                if fields[2].replace(" ", "") != "{0,1}":
                    raise ValueError("expected binary nominal ARFF attributes")
            if line.lower() == "@data":
                data = True
            continue
        if not (line.startswith("{") and line.endswith("}")):
            raise ValueError("expected sparse ARFF row")
        row = np.zeros(len(attributes), dtype=np.float32)
        seen = set()
        for field in line[1:-1].split(","):
            if not field.strip():
                continue
            index, value = field.split()
            index = int(index)
            if index in seen or index < 0 or index >= len(attributes) or value not in ("0", "1"):
                raise ValueError("invalid sparse ARFF cell")
            seen.add(index)
            row[index] = float(value)
        rows.append(row)
    if attributes[feature_count:] != label_names or len(attributes) != feature_count + len(label_names):
        raise ValueError("ARFF/XML label order mismatch")
    matrix = np.asarray(rows, dtype=np.float32)
    return matrix[:, :feature_count], matrix[:, feature_count:]

def prepare():
    data = ROOT / "data"
    out = data / "fixtures"
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(SEED)
    datasets = {}
    wine = np.loadtxt(data / "winequality-white.csv", delimiter=";", skiprows=1, dtype=np.float32)
    wx, wy = wine[:, :-1], wine[:, -1:]
    # Hash groups rather than rows: identical feature vectors never straddle splits.
    wine_groups = np.asarray([int.from_bytes(hashlib.sha256(row.tobytes() + str(SEED).encode()).digest()[:8], "little") % 10 for row in wx])
    datasets["wine"] = (wx, wy, 0, 0, {"train": np.flatnonzero(wine_groups < 6),
        "validation": np.flatnonzero((wine_groups >= 6) & (wine_groups < 8)), "test": np.flatnonzero(wine_groups >= 8)},
        "hashed feature-vector groups, 60/20/20 expected; identical features stay together", ["winequality-white.csv"])
    magic = np.loadtxt(data / "magic04.data", delimiter=",", dtype=str)
    mx, my = magic[:, :-1].astype(np.float32), (magic[:, -1:] == "g").astype(np.float32)
    mtrain, mtest = stratified(np.arange(len(mx)), my[:, 0], .2, rng)
    mtrain, mval = stratified(mtrain, my[:, 0], .25, rng)
    datasets["magic"] = (mx, my, 1, 2, {"train": mtrain, "validation": mval, "test": mtest},
        "deterministic class-stratified 60/20/20; gamma=1, hadron=0", ["magic04.data"])
    letter = np.loadtxt(data / "letter-recognition.data", delimiter=",", dtype=str)
    lx = letter[:, 1:].astype(np.float32)
    ly = np.asarray([[ord(a) - ord("A")] for a in letter[:, 0]], dtype=np.float32)
    ltrain, lval = stratified(np.arange(16000), ly[:, 0], .2, rng)
    datasets["letter"] = (lx, ly, 2, 26, {"train": ltrain, "validation": lval, "test": np.arange(16000, 20000)},
        "official final 4000 rows as test; stratified 80/20 within first 16000", ["letter-recognition.data"])
    source = data / "delicious-source"
    label_names = [e.attrib["name"] for e in ET.parse(source / "delicious.xml").getroot()]
    # The original Mulan XML has exactly this spelling discrepancy relative to
    # both ARFF files. Column order is retained; no target is dropped or moved.
    assert label_names[612] == "TTAG_m\\'usica"
    label_names[612] = "TAG_m\\'usica"
    dtx, dty = sparse_arff(source / "delicious-train.arff", 500, label_names)
    dex, dey = sparse_arff(source / "delicious-test.arff", 500, label_names)
    dtrain = rng.permutation(len(dtx))
    cut = int(len(dtrain) * .2)
    dx, dy = np.concatenate([dtx, dex]), np.concatenate([dty, dey])
    datasets["delicious"] = (dx, dy, 1, 2, {"train": np.sort(dtrain[cut:]), "validation": np.sort(dtrain[:cut]),
        "test": np.arange(len(dtx), len(dx))}, "official train/test; seeded 80/20 within official train; all 983 labels",
        ["delicious.rar", "delicious-source/delicious-train.arff", "delicious-source/delicious-test.arff", "delicious-source/delicious.xml"])
    manifest = {"schema": 1, "split_seed": SEED, "numpy_version": np.__version__, "format": "GHBDS001 little-endian raw float32", "datasets": {}}
    for name, (x, y, objective, classes, splits, description, sources) in datasets.items():
        directory = out / name
        directory.mkdir()
        entry = {"objective": objective, "classes": classes, "features": x.shape[1], "targets": y.shape[1],
                 "total_rows": len(x), "split_rule": description, "sources": {p: sha256(data / p) for p in sources}, "splits": {}}
        if name == "delicious":
            entry["labels"] = label_names
            entry["source_xml_spelling_correction"] = {"label_column": 612, "xml": "TTAG_m\\'usica", "arff": "TAG_m\\'usica"}
        combined = np.concatenate(list(splits.values()))
        assert len(combined) == len(x) and len(np.unique(combined)) == len(x)
        for split, indices in splits.items():
            path = directory / f"{split}.ghb"
            write(path, x[indices], y[indices], objective, classes)
            idx = directory / f"{split}-source-indices.u32"
            np.asarray(indices, dtype="<u4").tofile(idx)
            summary = {"rows": len(indices), "sha256": sha256(path), "bytes": path.stat().st_size,
                       "row_indices_sha256": sha256(idx), "target_mean": np.mean(y[indices], axis=0, dtype=np.float64).tolist()}
            if objective == 1:
                summary["positive_count_by_label"] = np.sum(y[indices], axis=0, dtype=np.float64).astype(int).tolist()
            if objective == 2:
                summary["class_counts"] = np.bincount(y[indices, 0].astype(int), minlength=classes).tolist()
            entry["splits"][split] = summary
            # Round-trip catches layout/target mistakes before a GPU job is authorized.
            xx, yy, _ = load(path)
            assert np.array_equal(xx, x[indices]) and np.array_equal(yy, y[indices])
        manifest["datasets"][name] = entry
        print(name, {s: len(i) for s, i in splits.items()}, x.shape[1], y.shape[1])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
