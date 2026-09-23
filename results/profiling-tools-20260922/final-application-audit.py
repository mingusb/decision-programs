"""CPU-only validation of retained application JSON and recorded process identities."""
import datetime
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ['nsys', 'ncu', 'memcheck', 'racecheck', 'initcheck', 'synccheck',
         'cupti-trace', 'cupti-range', 'cupti-pc', 'nvbit-count', 'nvbit-memory', 'nvbit-graph']
EXPECTED = {f'driver61692-final-{kind}-{tool}' for kind in ['count', 'booster'] for tool in TOOLS}
CONFIG = ['objective', 'generator_version', 'rows', 'test_rows', 'features', 'outputs', 'seed',
          'test_seed', 'rounds', 'max_depth', 'max_bins', 'output_tile_size', 'histogram',
          'tree_build', 'effective_deeper_histogram', 'tree_batch_size', 'root_histogram',
          'split_policy', 'root_counts', 'split_batch', 'quantize_policy', 'learning_rate',
          'l2', 'min_leaf_rows', 'min_child_hessian', 'min_gain', 'max_leaf_value', 'optimization_order']

def artifact(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def application(path):
    text = path.read_text(errors='replace')
    decoder, records, consumed = json.JSONDecoder(), [], -1
    for match in re.finditer(r'^\s*\{', text, re.MULTILINE):
        start = text.index('{', match.start(), match.end())
        if start < consumed:
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        consumed = end
        if isinstance(value, dict) and value.get('kind') in ['ghb.instrumentation', 'ghb.training']:
            records.append(value)
    if len(records) != 1:
        raise ValueError(f'{path}: expected one application JSON, found {len(records)}')
    return records[0]

def identities(value, location=''):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ['observed_processes', 'owned_processes', 'survivors'] and isinstance(child, list):
                for item in child:
                    if isinstance(item, dict) and 'pid' in item and 'start_ticks' in item:
                        yield item, location + '/' + key
            yield from identities(child, location + '/' + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from identities(child, location + '/' + str(i))

def current_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as error:
        return {'pid': pid, 'read_error': str(error)}

def write_new(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
reference_path = ROOT / 'capture-validation/booster-capture0.stdout.log'
reference = application(reference_path)
manifest_paths = sorted(ROOT.glob('driver61692-final-*/manifest.json'))
errors, runs, recorded = [], [], {}
found = {p.parent.name for p in manifest_paths}
if found != EXPECTED:
    errors.append({'missing_runs': sorted(EXPECTED - found), 'unexpected_runs': sorted(found - EXPECTED)})
if reference['validation'] != {'cpu_gpu_max_abs_error': 0, 'serialization_equal': True}:
    errors.append({'reference_validation_failed': reference['validation']})
for path in manifest_paths:
    manifest = json.loads(path.read_text())
    name, local_errors = path.parent.name, []
    row = {'run': name, 'manifest': artifact(path), 'stdout': artifact(path.parent / 'stdout.log'),
           'tool': manifest['tool'], 'runner_status': manifest['status'],
           'requested_command': manifest['requested_command']}
    if manifest['status'] != 'passed' or manifest.get('runner_exit_code') != 0:
        local_errors.append('runner did not pass')
    for identity, location in identities(manifest):
        key = (identity['pid'], identity['start_ticks'])
        item = recorded.setdefault(key, {'recorded_identity': identity, 'observations': []})
        item['observations'].append({'run': name, 'manifest_location': location})
    try:
        app = application(path.parent / 'stdout.log')
        row.update(application_kind=app['kind'], validation=app['validation'])
        if '-count-' in name:
            workload, validation = app['workload'], app['validation']
            row['workload'] = workload
            if app['kind'] != 'ghb.instrumentation' or app.get('benchmark') != 'count_histogram_component':
                local_errors.append('wrong counting application kind')
            if validation.get('passed') is not True or validation.get('mismatched_bins') != 0:
                local_errors.append('histogram bin validation failed')
            if validation.get('checked_bins') != workload['bins'] or validation.get('checked_repetitions') != workload['repetitions']:
                local_errors.append('histogram validation coverage differs from workload')
        else:
            row['configuration'] = {key: app[key] for key in CONFIG}
            row.update(tree_execution=app['tree_execution'], training_loss=app['training_loss'], heldout=app['heldout'])
            if app['kind'] != 'ghb.training' or app['validation'].get('cpu_gpu_max_abs_error') != 0 or app['validation'].get('serialization_equal') is not True:
                local_errors.append('booster CPU/GPU or serialization validation failed')
            losses = app['training_loss'] + list(app['heldout'].values())
            if len(app['training_loss']) != app['rounds'] + 1 or not all(math.isfinite(x) for x in losses):
                local_errors.append('invalid or incomplete loss sequence')
            differences = {key: {'reference': reference[key], 'observed': app[key]} for key in CONFIG if app[key] != reference[key]}
            if name == 'driver61692-final-booster-ncu':
                row['comparison_category'] = 'larger_8192_row_NCU_workload_separate_from_reference'
                row['reference_loss_comparison'] = 'not_applicable_different_training_rows'
                if differences != {'rows': {'reference': 128, 'observed': 8192}}:
                    local_errors.append('unexpected larger NCU configuration difference')
            else:
                row['comparison_category'] = 'same_128_row_reference_workload'
                row['training_loss_exactly_equal'] = app['training_loss'] == reference['training_loss']
                row['heldout_loss_exactly_equal'] = app['heldout'] == reference['heldout']
                row['max_abs_training_loss_difference'] = max(abs(x-y) for x,y in zip(app['training_loss'],reference['training_loss']))
                row['max_abs_heldout_loss_difference'] = max(abs(app['heldout'][key]-reference['heldout'][key]) for key in reference['heldout'])
                if differences or not row['training_loss_exactly_equal'] or not row['heldout_loss_exactly_equal']:
                    local_errors.append('zero-allowance reference comparison failed')
            row['reference_configuration_differences'] = differences
    except (KeyError, ValueError, TypeError) as error:
        local_errors.append(str(error))
    row.update(passed=not local_errors, errors=local_errors)
    if local_errors:
        errors.append({'run': name, 'errors': local_errors})
    runs.append(row)

processes = []
for (pid, start_ticks), item in sorted(recorded.items()):
    current = current_identity(pid)
    if current is None:
        outcome = 'not_present'
    elif 'read_error' in current:
        outcome = 'unknown_read_error'
    elif current['start_ticks'] != start_ticks:
        outcome = 'pid_reused_original_identity_gone'
    elif current['state'] in ['Z', 'X', 'x']:
        outcome = 'exited_nonexecuting_identity_still_present'
    else:
        outcome = 'still_alive'
    item.update(current_identity=current, outcome=outcome)
    processes.append(item)

validation = {'schema': 'ghb.final_application_validation.v1', 'created_utc': now,
              'method': 'JSONDecoder.raw_decode from line-start objects, skipping nested objects; exact numeric comparisons with zero allowance.',
              'auditor': artifact(Path(__file__).resolve()), 'reference': {'artifact': artifact(reference_path),
              'configuration': {key: reference[key] for key in CONFIG}, 'training_loss': reference['training_loss'],
              'heldout': reference['heldout'], 'validation': reference['validation']},
              'expected_runs': 24, 'observed_runs': len(runs), 'passed_runs': sum(r['passed'] for r in runs),
              'small_booster_exact_comparisons': sum(r.get('comparison_category') == 'same_128_row_reference_workload' for r in runs),
              'separate_larger_booster_cases': sum(r.get('comparison_category') == 'larger_8192_row_NCU_workload_separate_from_reference' for r in runs),
              'passed': not errors, 'errors': errors, 'runs': runs,
              'limits': ['Application-reported CPU/GPU and serialization checks are audited from retained logs; models were not regenerated.',
                         'Graph and stream execution modes are retained explicitly and may differ across tools.',
                         'Instrumented timings are excluded from correctness comparisons and performance ranking.']}
process_check = {'schema': 'ghb.post_matrix_process_check.v1', 'created_utc': now,
                 'method': 'Read /proc/PID/stat; compare PID and starttime field22 against all observed/cleanup identities recursively retained in final manifests. No signals or GPU calls.',
                 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'auditor': artifact(Path(__file__).resolve()), 'manifests': [artifact(p) for p in manifest_paths],
                 'unique_recorded_identities': len(processes),
                 'outcome_counts': {state: sum(p['outcome'] == state for p in processes) for state in sorted({p['outcome'] for p in processes})},
                 'passed': found == EXPECTED and bool(processes) and all(p['outcome'] not in ['still_alive','unknown_read_error'] for p in processes),
                 'processes': processes,
                 'limits': ['Point-in-time absence/liveness check of recorded identities; does not prove no unobserved process existed between ownership snapshots.']}
write_new(ROOT / 'final-application-validation.json', validation)
write_new(ROOT / 'post-matrix-process-check.json', process_check)
print(json.dumps({'application_passed': validation['passed'], 'passed_runs': validation['passed_runs'],
                  'exact_small_booster_comparisons': validation['small_booster_exact_comparisons'],
                  'larger_booster_cases': validation['separate_larger_booster_cases'],
                  'process_check_passed': process_check['passed'], 'unique_process_identities': len(processes),
                  'process_outcomes': process_check['outcome_counts'], 'errors': errors}))
raise SystemExit(0 if validation['passed'] and process_check['passed'] else 1)
