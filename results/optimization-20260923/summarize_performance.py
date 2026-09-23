#!/usr/bin/env python3
"""Offline paired timing summary. Profiler timings are deliberately absent."""
import json
import argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--idle',action='store_true')
args=parser.parse_args()
suffix='-idle' if args.idle else ''
rng = np.random.default_rng(2026092301)

def summary(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    assert a.shape == b.shape and len(a) and np.all(a>0) and np.all(b>0)
    ratios = b/a
    sampled = np.median(ratios[rng.integers(0,len(ratios),size=(20000,len(ratios)))],axis=1)
    low, high = np.quantile(sampled,[.025,.975])
    return {'pairs':len(a), 'reference_median':float(np.median(a)), 'candidate_median':float(np.median(b)),
            'paired_ratios':ratios.tolist(), 'median_paired_ratio':float(np.median(ratios)),
            'bootstrap_95_percentile_interval':[float(low),float(high)],
            'timing_gate_upper_below_one':bool(high<1), 'median_paired_speedup':float(1/np.median(ratios))}

result = {'method':'20,000 paired-bootstrap median-ratio resamples, seed 2026092301; percentile 95%; descriptive casewise intervals, no multiplicity-adjusted/global claim',
          'performance_promotion_valid':False,
          'timing_scope':'idle_confirmation' if args.idle else 'gaming_time_exploratory',
          'limitation':('Idle GPU, no concurrent heavy CPU analysis/builds. Casewise timing intervals do not waive quality failures or regressions elsewhere in the workload matrix.' if args.idle else 'Exploratory campaign; confirmed concurrent game activity and some overlapping CPU work. Bootstrap intervals do not remove this confound.'),
          'split_training':{}, 'prediction':[], 'split_operations':[]}
campaign = HERE/('split-campaign'+suffix)/'observations.json'
if campaign.exists():
    observations = json.loads(campaign.read_text())
    for metric in ('training_wall','training','prediction_wall'):
        a,b=[],[]
        for pair in range(5):
            values={x['policy']:x['metrics']['timing_ms'][metric] for x in observations if x['pair']==pair}
            if len(values)==2:a.append(values['warp32']);b.append(values['warp-wide'])
        result['split_training'][metric] = summary(a,b)
    a,b=[],[]
    for pair in range(5):
        values={x['policy']:sum(x['metrics']['timing_ms'][key] for key in ('training_wall','prediction_wall','serialization'))
                for x in observations if x['pair']==pair}
        if len(values)==2:a.append(values['warp32']);b.append(values['warp-wide'])
    result['split_training']['train_predict_serialize_ms'] = summary(a,b)

for folder in ('prediction-actual'+suffix,'prediction-matrix'+suffix):
    for path in sorted((HERE/folder).glob('*.json')):
        report=json.loads(path.read_text())
        if not isinstance(report,dict) or report.get('schema')!='ghb.prediction_benchmark.v1':continue
        a,b=[],[]
        for pair in range(report['pairs']):
            values={x['policy']:x['milliseconds'] for x in report['samples'] if x['pair']==pair}
            if len(values)==2:a.append(values['per_tree']);b.append(values['fused_output'])
        if a:
            result['prediction'].append({'artifact':str(path.relative_to(HERE)),
                 **{k:report[k] for k in ('rows','features','outputs','trees','nodes','bitwise_frozen_reference_passed')},
                 'complete_call_ms':summary(a,b), 'memory':report['payload_excluding_quantizer']})

bench=HERE/('split-wide-benchmark'+suffix+'.json')
if bench.exists():
    try:
        reports=[json.loads(line) for line in bench.read_text().splitlines() if line.startswith('{')]
    except json.JSONDecodeError:
        status=HERE/('split-wide-benchmark'+suffix+'-status.json')
        if not status.exists():raise
        result['split_operation_failure']=json.loads(status.read_text())
        reports=[]
    for report in reports:
        for case in report['cases']:
            cell={k:v for k,v in case.items() if k!='raw'}
            for metric in ('device_us','host_us'):
                a,b=[],[]
                for pair in range(report['samples']):
                    values={x['variant']:x[metric] for x in case['raw'] if x['sample']==pair}
                    a.append(values[0]);b.append(values[1])
                cell[metric]=summary(a,b)
            result['split_operations'].append(cell)
destination=HERE/('performance'+suffix+'-summary.json')
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('prediction','split_operations')},indent=2))
for p in result['prediction']:print(p['artifact'],p['complete_call_ms'])
print('split operation cells',len(result['split_operations']))
