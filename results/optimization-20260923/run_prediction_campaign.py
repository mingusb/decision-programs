#!/usr/bin/env python3
"""Frozen-model complete-call benchmarks, GPU jobs strictly serial."""
import argparse
import json
import pathlib
import struct
import subprocess
import time
from run_split_campaign import ROOT, digest, telemetry

HERE = pathlib.Path(__file__).resolve().parent
BIN = ROOT/'build/optimization-20260923/ghb_prediction_bench'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['actual', 'matrix'], required=True)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if args.tag and not all(c.isalnum() or c in '-_' for c in args.tag):raise ValueError('invalid tag')
    out = HERE/('prediction-'+args.phase+('-'+args.tag if args.tag else ''))
    out.mkdir(exist_ok=False)
    cases = []
    if args.phase == 'actual':
        fixture = ROOT/'results/booster-level-batch-20260922/data/fixtures/delicious/validation.ghb'
        with fixture.open('rb') as source:
            magic, version, rows, features, targets, objective, classes = struct.unpack('<8s6I', source.read(32))
            assert magic == b'GHBDS001' and version == 1
            data = source.read(rows*features*4)
            assert len(data) == rows*features*4
        features_file = out/'validation-features.f32'
        features_file.write_bytes(data)
        (out/'input.json').write_text(json.dumps({'fixture': str(fixture), 'fixture_sha256': digest(fixture),
            'features_sha256': digest(features_file), 'extraction': 'verbatim row-major feature bytes; no transform'}, indent=2)+'\n')
        cases = [('delicious-validation', HERE/'nsys-delicious-baseline-model/model.ghb', rows, features_file)]
    else:
        # Host-model setup only: preserve exact serialized metadata/trees for
        # the first 65 independent outputs of an existing frozen binary model.
        original=ROOT/'results/booster-trainer-20260922/multilabel-1024-tile16/model.ghb'
        raw=original.read_bytes()
        magic, version, objective, outputs, features, tree_count=struct.unpack_from('<8s4IQ',raw)
        assert magic==b'GHBMODEL' and version==1 and objective==1 and outputs>=65
        metadata_begin=32+8*outputs
        position=metadata_begin
        for feature in range(features):
            kind,cuts,categories=struct.unpack_from('<3I',raw,position)
            position+=12+4*(cuts+categories)
        metadata=raw[metadata_begin:position]
        selected=[]
        for tree in range(tree_count):
            start=position
            output,nodes=struct.unpack_from('<2I',raw,position)
            position+=8+28*nodes
            if output<65:selected.append(raw[start:position])
        assert position==len(raw)
        model65=out/'frozen-prefix65.ghb'
        model65.write_bytes(struct.pack('<8s4IQ',magic,version,objective,65,features,len(selected))+
                            raw[32:32+65*8]+metadata+b''.join(selected))
        (out/'frozen-prefix65-provenance.json').write_text(json.dumps({'source':str(original),
            'source_sha256':digest(original),'derived_sha256':digest(model65),'outputs':65,
            'trees':len(selected),'operation':'byte-preserving selection of first65 independent outputs; benchmark host-model setup, no training or quality claim'},indent=2)+'\n')
        models = [
            ('regression1', ROOT/'results/booster-trainer-20260922/regression-global-b/model.ghb'),
            ('binary1', ROOT/'results/booster-trainer-20260922/binary/model.ghb'),
            ('independent3', ROOT/'results/profiling-expansion-20260923/model-before/model.ghb'),
            ('multiclass5', ROOT/'results/booster-trainer-20260922/multiclass/model.ghb'),
            ('independent65', model65),
            ('independent1024', ROOT/'results/booster-trainer-20260922/multilabel-1024-tile16/model.ghb')]
        cases = [(f'{name}-rows{rows}', model, rows, None) for name, model in models for rows in (32,4096,65536)]
    observations = []
    for name, model, rows, features in cases:
        destination = out/(name+'.json')
        argv = [str(BIN), '--model', str(model), '--rows', str(rows), '--pairs', '15', '--warmup', '3', '--output', str(destination)]
        if features:
            argv += ['--features-bin', str(features)]
        observation = {'name': name, 'argv': argv, 'binary_sha256': digest(BIN), 'model_sha256': digest(model),
                       'telemetry_before': telemetry()}
        (out/(name+'.invocation.json')).write_text(json.dumps(observation, indent=2)+'\n')
        start = time.monotonic()
        with (out/(name+'.stdout')).open('w') as stdout, (out/(name+'.stderr')).open('w') as stderr:
            result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=600)
        observation.update(returncode=result.returncode, process_wall_seconds=time.monotonic()-start,
                           telemetry_after=telemetry())
        if destination.exists():
            observation['result_sha256'] = digest(destination)
            observation['medians'] = json.loads(destination.read_text())['median_ms']
        observations.append(observation)
        (out/'observations.json').write_text(json.dumps(observations, indent=2)+'\n')
        print(name, result.returncode, observation.get('medians'), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
