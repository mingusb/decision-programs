#!/usr/bin/env python3
"""Read isolated whole-call Nsight captures; counts only, never timing rankings."""
from pathlib import Path
import hashlib
import json
import sqlite3

HERE = Path(__file__).resolve().parent

def main():
    policies = {}
    for name, directory in [('per_tile', 'nsys-per-tile-v2'), ('final_status', 'nsys-final-status')]:
        folder = HERE / directory
        manifest = json.loads((folder / 'manifest.json').read_text())
        benchmark = json.loads((folder / 'benchmark.json').read_text())
        assert manifest['status'] == 'passed' and benchmark['bitwise_frozen_reference_passed']
        assert benchmark['pairs'] == 1 and benchmark['warmup_per_policy'] == 0
        path = folder / 'profile.1.sqlite'
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
            api = dict(db.execute('SELECT s.value,COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId GROUP BY s.value'))
            copies = db.execute('SELECT copyKind,COUNT(*),SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').fetchall()
            kernels = db.execute('SELECT s.value,COUNT(*),k.blockX,k.blockY,k.blockZ,k.registersPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value,k.blockX,k.blockY,k.blockZ,k.registersPerThread ORDER BY s.value').fetchall()
            sequence = db.execute('SELECT s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
        policies[name] = {'sqlite': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'api_counts': api, 'copy_counts_and_bytes': copies, 'kernels': kernels,
                          'kernel_sequence': sequence, 'payload': benchmark['payload_excluding_quantizer']}
    a, b = policies.values()
    document = {'scope': 'Same-binary isolated complete prediction calls, same model/features; profiler counts only, no timing ranking.',
                'policies': policies, 'kernel_sequences_equal': a['kernel_sequence'] == b['kernel_sequence']}
    assert document['kernel_sequences_equal']
    with (HERE / 'transfer-comparison.json').open('x') as stream:
        json.dump(document, stream, indent=2); stream.write('\n')
    for name, policy in policies.items():
        print(name, {k:v for k,v in policy['api_counts'].items() if any(k.startswith(p) for p in
            ('cudaMalloc_', 'cudaFree_', 'cudaMemcpyAsync_', 'cudaMemcpy2DAsync_', 'cudaStreamSynchronize_', 'cudaLaunchKernel_'))})
        print('copies', policy['copy_counts_and_bytes'], 'kernel_count', sum(row[1] for row in policy['kernels']))
    print('kernel_sequences_equal', document['kernel_sequences_equal'])

if __name__ == '__main__':
    main()
