#!/usr/bin/env python3
"""Serial, unprofiled paired experiment. No production preprocessing/training."""
import datetime
import argparse
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / 'split-campaign'
BIN = ROOT / 'build/optimization-20260923/ghb_real_bench'
DATA = ROOT / 'results/booster-level-batch-20260922/data/fixtures/delicious'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def telemetry():
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate', '--format=csv'], capture_output=True, text=True)
    return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}

def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if args.tag:
        if not all(c.isalnum() or c in '-_' for c in args.tag):raise ValueError('invalid tag')
        OUT = OUT.with_name(OUT.name+'-'+args.tag)
    OUT.mkdir(exist_ok=False)
    identities = {str(p.relative_to(ROOT)): digest(p) for p in [BIN, DATA/'train.ghb', DATA/'validation.ghb']}
    for p in sorted((ROOT/'training').rglob('*')):
        if p.is_file() and p.suffix in ('.cpp', '.cu', '.cuh', '.hpp', '.inc', '.txt'):
            identities[str(p.relative_to(ROOT))] = digest(p)
    (OUT/'identities.json').write_text(json.dumps(identities, indent=2)+'\n')
    observations = []
    for pair in range(-1, 5):
        policies = ['warp32', 'warp-wide'] if pair % 2 == 0 else ['warp-wide', 'warp32']
        for position, policy in enumerate(policies):
            name = f'{"warmup" if pair < 0 else "pair"+str(pair)}-{policy}'
            dest = OUT/name
            argv = [str(BIN), '--train', str(DATA/'train.ghb'), '--evaluation', str(DATA/'validation.ghb'),
                    '--output-dir', str(dest), '--rounds', '1', '--depth', '3', '--bins', '32',
                    '--output-tile', '32', '--tree-build', 'output-batch', '--tree-execution', 'graph',
                    '--histogram', 'global', '--instrumentation', 'off', '--split-policy', policy,
                    '--tree-export-batch', '16']
            observation = {'name': name, 'pair': pair, 'position': position, 'policy': policy,
                           'argv': argv, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           'telemetry_before': telemetry()}
            (OUT/(name+'.invocation.json')).write_text(json.dumps(observation, indent=2)+'\n')
            start = time.monotonic()
            with (OUT/(name+'.stdout')).open('w') as stdout, (OUT/(name+'.stderr')).open('w') as stderr:
                result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180)
            observation.update(returncode=result.returncode, process_wall_seconds=time.monotonic()-start,
                               telemetry_after=telemetry())
            if (dest/'metrics.json').exists():
                observation['metrics'] = json.loads((dest/'metrics.json').read_text())
                observation['artifact_sha256'] = {p.name: digest(p) for p in dest.iterdir() if p.is_file()}
            observations.append(observation)
            (OUT/'observations.json').write_text(json.dumps(observations, indent=2)+'\n')
            print(name, result.returncode, observation.get('metrics', {}).get('timing_ms'), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
    post = {name: digest(ROOT/name) for name in identities}
    (OUT/'identities-after.json').write_text(json.dumps(post, indent=2)+'\n')
    (OUT/'identity-comparison.json').write_text(json.dumps({'all_unchanged':post == identities,
        'changed':[name for name in identities if identities[name] != post[name]]}, indent=2)+'\n')

if __name__ == '__main__':
    main()
