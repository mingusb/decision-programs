"""Serial uninstrumented whole-training comparisons; run only after GPU validation."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import time

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]
BINARY = EVIDENCE / 'bin/ghb_bench'
CASES = {
    'scalar': dict(rows=262144, features=16, outputs=1, rounds=10, depth=5, bins=32),
    'multi33': dict(rows=32768, features=16, outputs=33, rounds=10, depth=5, bins=32),
    'wide129': dict(rows=8192, features=65, outputs=129, rounds=5, depth=3, bins=64),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    output = EVIDENCE / 'synthetic'
    output.mkdir(exist_ok=False)
    identity = sha(BINARY)
    jobs = [('warmup', order, 0, CASES['multi33']) for order in (2, 3, 4)]
    for name, shape in CASES.items():
        for repetition in range(3):
            for order in (2, 3, 4)[repetition:] + (2, 3, 4)[:repetition]:
                jobs.append((name, order, repetition, shape))
    records = []
    for name, order, repetition, shape in jobs:
        tag = f'{name}-o{order}-r{repetition}'
        command = [str(BINARY), '--objective', 'binary', '--test-rows', '512',
                   '--tree-build', 'output-batch', '--tree-execution', 'graph',
                   '--histogram', 'global', '--output-tile', '16', '--tree-export-batch', '16',
                   '--instrumentation', 'off', '--max-leaf-value', '1', '--optimization-order', str(order),
                   '--output-dir', str(output / tag)]
        for key, value in shape.items():
            command += ['--' + key, str(value)]
        records.append(dict(case=name, order=order, repetition=repetition, tag=tag, command=command))
    (output / 'protocol.json').write_text(json.dumps(dict(binary_sha256=identity, script_sha256=sha(Path(__file__)),
        cases=CASES, repetitions=3, order='rotated', timing='uninstrumented total_train_ms including preparation and cleanup',
        warmups='one unranked multi33 job per order', jobs=records), indent=2)+'\n')
    for record in records:
        assert sha(BINARY) == identity
        start = time.perf_counter()
        with (output / (record['tag']+'.stdout')).open('x') as out, (output / (record['tag']+'.stderr')).open('x') as err:
            process = subprocess.run(record['command'], cwd=ROOT, stdout=out, stderr=err)
        record.update(returncode=process.returncode, wall_seconds=time.perf_counter()-start)
        (output / (record['tag']+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(record['tag'], process.returncode, flush=True)
        if process.returncode:
            raise RuntimeError('failed synthetic run; raw failure retained')
        record['result'] = json.loads((output / record['tag'] / 'result.json').read_text())
    summary = {}
    for name in CASES:
        summary[name] = {}
        for order in (2,3,4):
            runs = [r['result'] for r in records if r['case']==name and r['order']==order]
            summary[name][order] = dict(
                total_train_ms=[r['timing']['total_train_ms'] for r in runs],
                training_ms=[r['timing']['training_ms'] for r in runs],
                total_train_median_ms=statistics.median(r['timing']['total_train_ms'] for r in runs),
                training_median_ms=statistics.median(r['timing']['training_ms'] for r in runs),
                heldout_loss=[r['heldout']['loss'] for r in runs],
                memory=runs[0]['memory'],
                max_prediction_reference_error=max(r['validation']['cpu_gpu_max_abs_error'] for r in runs),
                training_loss_increases=[sum(b>a for a,b in zip(r['training_loss'],r['training_loss'][1:])) for r in runs])
    (output / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__ == '__main__':
    main()
