"""CPU-only, zero-allowance audit of this experiment; preserve all failures."""
from pathlib import Path
import importlib.util
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPERS = ROOT / "results/booster-resident-20260922/audit_optimized_quality.py"
spec = importlib.util.spec_from_file_location("retained_quality", HELPERS)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.OUT = OUT
a = old.a

def plan():
    pairs = [(f"{case}-base-{suffix}", f"{case}-{mode}-{suffix}")
             for case in ("scalar", "129", "1024", "4096") for suffix in ("a", "b")
             for mode in ("counts", "split", "both", "deep-shared")]
    pairs += [(f"scalar-original-{s}", f"scalar-base-{s}") for s in ("a", "b")]
    pairs += [(f"{case}-base-{s}", f"{case}-counts-shared-{s}") for case in ("129", "4096") for s in ("a", "b")]
    for case in ("129", "1024", "4096"):
        for suffix in ("b", "c", "d"):
            pair=(f"{case}-base-a",f"{case}-base-{suffix}")
            pairs += [pair,pair[::-1]]
    pairs += [(f"confirm-{case}-base",f"confirm-{case}-{mode}") for case in ("129","4096") for mode in ("both","deep-shared")]
    pairs += [("multiclass17-base", "multiclass17-"+mode) for mode in ("both","deep-shared")]
    return sorted({name for pair in pairs for name in pair}), pairs


def mode_for(name):
    if name.startswith('scalar-original'): return 'original'
    for mode in ('counts-shared','deep-shared','counts','split','both','base'):
        if name.endswith('-'+mode) or any(name.endswith('-'+mode+'-'+s) for s in ('a','b','c','d')):
            return mode
    raise ValueError('unknown experiment case '+name)


def check_mode(name, meta, capture, model):
    mode=mode_for(name)
    expected=dict(root_histogram='batched',split_policy='warp32',root_counts='per-output',split_batch='per-tree')
    if mode=='original': expected.update(root_histogram='per-tree',split_policy='block256')
    if mode in ('counts','both','deep-shared'): expected['root_counts']='reuse-global'
    if mode=='counts-shared': expected['root_counts']='reuse-shared'
    if mode in ('split','both','deep-shared'): expected['split_batch']='root'
    expected['histogram']='shared' if name.startswith('scalar-') or mode=='deep-shared' else 'global'
    for key,value in expected.items():
        if meta[key]!=value: raise ValueError('unexpected '+key+' in '+name)
    command=capture['command']; flags=dict(zip(command[1::2],command[2::2]))
    for flag,key in (('--root-histogram','root_histogram'),('--split-policy','split_policy'),('--root-counts','root_counts'),('--split-batch','split_batch'),('--histogram','histogram')):
        if flag in flags and flags[flag]!=meta[key]: raise ValueError('command/result policy mismatch')
    if (model['outputs'],len(model['features']),len(model['trees']))!=(meta['outputs'],meta['features'],meta['trees']):
        raise ValueError('model/benchmark dimensions differ')

def main():
    destination = OUT / "quality"
    destination.mkdir(exist_ok=False)
    marker = destination / ".incomplete"
    marker.touch(exist_ok=False)
    evaluator = a.load_evaluator()
    names, pairs = plan()
    summary = dict(started_unix_seconds=time.time(), allowance=0.0,
                   cases=[], comparisons=[], errors=[], artifacts=[],
                   sources=[a.artifact(path) for path in (Path(__file__), HELPERS, old.HELPERS, a.EVALUATOR)])
    metadata, captures, reports, models = {}, {}, {}, {}
    for name in names:
        row = dict(name=name)
        summary["cases"].append(row)
        try:
            meta, capture, artifacts = old.validate_capture(name)
            metadata[name], captures[name] = meta, capture
            summary["artifacts"].extend(artifacts)
            objective, dataset, split = a.identity(meta)
            directory = OUT / name
            paths = {}
            for label, filename in (("model", "predictions.csv"), ("base", "baseline.csv")):
                report = evaluator.evaluate(directory / "targets.csv", directory / filename, objective, dataset, split,
                                            meta["outputs"] if objective == "multiclass" else None)
                if (report["sample_count"], report["outputs"]) != (meta["test_rows"], meta["outputs"]):
                    raise ValueError("CSV dimensions differ from benchmark")
                paths[label] = destination / (name + "-" + label + ".json")
                a.write_json(paths[label], report)
            reports[name] = paths["model"]
            models[name] = a.read_model(directory / "model.ghb")
            check_mode(name, meta, capture, models[name])
            comparison = evaluator.compare(paths["base"], paths["model"], 0.0)
            a.write_json(destination / (name + "-model-vs-base.json"), comparison)
            row.update(a.compact_comparison(comparison))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            row.update(status="invalid_evidence", error=str(error))
            summary["errors"].append(dict(case=name, error=str(error)))
        print(name, row["status"], flush=True)
    for reference, candidate in pairs:
        row = dict(reference=reference, candidate=candidate)
        summary["comparisons"].append(row)
        try:
            for key in a.IDENTITY:
                if key != 'histogram' and metadata[reference][key] != metadata[candidate][key]:
                    raise ValueError('unmatched protocol '+key)
            if metadata[reference]['histogram'] != metadata[candidate]['histogram'] and mode_for(candidate)!='deep-shared':
                raise ValueError('undeclared deeper histogram policy change')
            for key in ("tree_execution", "quantize_policy", "tree_export_batch_requested", "tree_export_batch_effective"):
                if metadata[reference][key] != metadata[candidate][key]:
                    raise ValueError("unmatched " + key)
            if captures[reference]["executable_sha256"] != captures[candidate]["executable_sha256"]:
                raise ValueError("not a same-binary comparison")
            before, after = OUT / reference, OUT / candidate
            if a.sha(before / "targets.csv") != a.sha(after / "targets.csv"):
                raise ValueError("target/weight bytes differ")
            if old.feature_bytes(models[reference]) != old.feature_bytes(models[candidate]):
                raise ValueError("quantization metadata differs")
            comparison = evaluator.compare(reports[reference], reports[candidate], 0.0)
            row.update(a.compact_comparison(comparison))
            row["predictions"] = a.prediction_changes(before / "predictions.csv", after / "predictions.csv", before / "targets.csv", a.identity(metadata[candidate])[0])
            row["model"] = a.model_changes(before / "model.ghb", after / "model.ghb")
            row["encoded_domain"] = old.exact_region_comparison(models[reference], models[candidate])
            a.write_json(destination / (candidate + "-vs-" + reference + ".json"), dict(comparison=comparison, **row))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            row.update(status="invalid_evidence", error=str(error))
            summary["errors"].append(dict(candidate=candidate, error=str(error)))
        print(candidate, row["status"], flush=True)
    for artifact in summary["sources"] + summary["artifacts"]:
        try:
            if a.sha(artifact["path"]) != artifact["sha256"]:
                summary["errors"].append(dict(artifact_changed=artifact["path"]))
        except OSError as error:
            summary["errors"].append(dict(artifact_missing=artifact["path"],error=str(error)))
    summary["status"] = "invalid_evidence" if summary["errors"] else "regression" if any(r["status"] == "regression" for r in summary["cases"] + summary["comparisons"]) else "pass"
    summary["finished_unix_seconds"] = time.time()
    a.write_json(destination / "summary.json", summary)
    lines = ["# Strict quality audit", "", "Status: **" + summary["status"] + "**; allowance remains zero.", "",
             "| Candidate | Gate | Regressed metrics | Max prediction difference | Decision changes | All-bin margin bound |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary["comparisons"]:
        if "predictions" in row and "encoded_domain" in row and "regressions" in row:
            p = row["predictions"]
            lines.append(f"| {row['candidate']} | {row['status']} | {len(row['regressions'])} | {p['max_abs_difference']:.17g} | {p['decision_changes']} | {row['encoded_domain']['maximum_raw_margin_bound_upward']:.17g} |")
        else:
            lines.append(f"| {row['candidate']} | invalid evidence | — | — | — | — |")
    lines += ["", "All-bin bounds use exact-real sums of stored leaves over intersecting encoded leaf regions. They exclude runtime arithmetic rounding. Measured quality failures remain failures.", ""]
    (destination / "summary.md").write_text("\n".join(lines))
    marker.unlink()
    return 2 if summary["errors"] else int(summary["status"] == "regression")

if __name__ == "__main__":
    raise SystemExit(main())
