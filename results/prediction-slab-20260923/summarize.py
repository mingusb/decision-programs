#!/usr/bin/env python3
"""Offline, casewise paired ratios; never starts CUDA or removes samples."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'results/optimization-20260923/prediction-slab-resumed'
SEED = 2026092302
BOOTSTRAPS = 20000

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    completion = read(CAMPAIGN / 'completion.json')
    identity = read(CAMPAIGN / 'identity-comparison.json')
    combined = read(CAMPAIGN / 'combined-results.json')
    assert completion['status'] == 'complete' and completion['completed_new_processes'] == 13
    assert completion['accepted_original_records'] == 37 and completion['verified_new_records'] == 13
    assert identity['all_unchanged'] and combined['status'] == 'complete'
    assert len(combined['records']) == 50 and [r['original_job_index'] for r in combined['records']] == list(range(50))
    rng = np.random.default_rng(SEED)
    rows = []
    for record in combined['records']:
        observation = read(Path(record['observations_path']))[record['observation_index']]
        assert observation['name'] == record['name'] and observation['mode'] == record['mode']
        assert observation['returncode'] == 0 and observation['result_validation_passed']
        path = Path(record['result_path'])
        assert digest(path) == observation['result_sha256'] == record['result_sha256']
        result = read(path)
        assert result['bitwise_frozen_reference_passed'] and result['slab_reference_checked']
        reference = 'fused_output' if observation['mode'] == 'compare-slab' else 'per_tree'
        by_pair = {}
        for sample in result['samples']:
            pair = by_pair.setdefault(sample['pair'], {})
            assert sample['policy'] not in pair
            pair[sample['policy']] = sample['milliseconds']
        assert len(by_pair) == result['pairs'] >= 15
        ratios = np.array([pair['fused_output_slab'] / pair[reference] for _, pair in sorted(by_pair.items())])
        assert np.isfinite(ratios).all() and (ratios > 0).all()
        bootstrap = np.median(ratios[rng.integers(0, len(ratios), size=(BOOTSTRAPS, len(ratios)))], axis=1)
        lower, upper = np.quantile(bootstrap, [.025, .975])
        ratio = float(np.median(ratios))
        classification = 'faster' if upper < 1 else 'slower' if lower > 1 else 'inconclusive'
        row = {'case': observation['name'].removesuffix('-' + observation['mode']),
               'reference': reference, 'candidate': 'fused_output_slab',
               'median_ms': result['median_ms'], 'median_paired_candidate_over_reference': ratio,
               'median_paired_speedup': 1 / ratio, 'paired_bootstrap_95_interval': [float(lower), float(upper)],
               'casewise_timing_classification': classification, 'paired_ratios': ratios.tolist(),
               'exact_raw_and_transformed_passed': True, 'payload': result['payload_excluding_quantizer'],
               'source': str(path.relative_to(ROOT)), 'sha256': digest(path), 'campaign_segment': record['origin']}
        rows.append(row)
        print(row['case'], reference, f'{ratio:.5f}', f'[{lower:.5f},{upper:.5f}]', classification)
    counts = {reference: {label: sum(row['reference'] == reference and row['casewise_timing_classification'] == label
                                    for row in rows) for label in ('faster', 'slower', 'inconclusive')}
              for reference in ('fused_output', 'per_tree')}
    document = {'schema': 1, 'scope': '25 frozen shapes, two same-binary paired comparisons each; complete synchronous calls',
                'seed': SEED, 'bootstrap_resamples': BOOTSTRAPS,
                'uncertainty': 'Percentile bootstrap of median paired candidate/reference ratios; casewise 95%, no multiplicity correction, no universal ranking.',
                'all_source_identities_unchanged': True, 'all_exactness_gates_passed': True,
                'conditions': 'Recorded residual desktop rendering; no zero-background-load claim. First37 complete comparisons retained;13 comparisons resumed after a process interruption. Unverified original job38 results retained separately and excluded from ranking, irrespective of timing.',
                'default_promoted': False, 'classifications': counts, 'cases': rows,
                'script_sha256': digest(Path(__file__)),
                'campaign_completion_sha256': digest(CAMPAIGN / 'completion.json'),
                'combined_manifest_sha256': digest(CAMPAIGN / 'combined-results.json'),
                'campaign_observations_sha256': digest(CAMPAIGN / 'observations.json')}
    with (OUT / 'performance-summary.json').open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('COUNTS', counts)

if __name__ == '__main__':
    main()
