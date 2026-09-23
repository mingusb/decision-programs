"""Same-policy repeat controls from frozen previous evidence; CPU only."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'results/booster-root-split-20260922'
helper=ROOT/'results/booster-resident-20260922/audit_optimized_quality.py'
spec=importlib.util.spec_from_file_location('old_helpers',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.OUT=OLD
a=h.a

def main():
    dest=OUT/'previous-repeat-controls';dest.mkdir(exist_ok=False)
    marker=dest/'.incomplete';marker.touch()
    ev=a.load_evaluator()
    summary=dict(allowance=0.0,comparisons=[],sources=[a.artifact(Path(__file__)),a.artifact(helper)],errors=[])
    for case in ('scalar','129','1024','4096'):
        for mode in ('base','both'):
            names=[f'{case}-{mode}-{s}' for s in ('a','b')]
            metadata=[h.validate_capture(name)[0] for name in names]
            a.matching(*metadata)
            for key in ('root_histogram','split_policy','tree_execution','quantize_policy','tree_export_batch_requested'):
                if metadata[0][key]!=metadata[1][key]:raise ValueError('unmatched control '+key)
            for before,after in (names,names[::-1]):
                comparison=ev.compare(OLD/'quality'/(before+'-model.json'),OLD/'quality'/(after+'-model.json'),0.0)
                change=a.prediction_changes(OLD/before/'predictions.csv',OLD/after/'predictions.csv',OLD/before/'targets.csv',a.identity(metadata[0])[0])
                row=dict(reference=before,candidate=after,**a.compact_comparison(comparison),predictions=change)
                a.write_json(dest/(after+'-vs-'+before+'.json'),dict(comparison=comparison,**row))
                summary['comparisons'].append(row)
                print(before,'->',after,row['status'],flush=True)
    summary['status']='regression' if any(x['status']=='regression' for x in summary['comparisons']) else 'pass'
    a.write_json(dest/'summary.json',summary)
    marker.unlink()
    return int(summary['status']=='regression')

if __name__=='__main__':raise SystemExit(main())
