"""Serial uninstrumented whole-trainer comparison; refuse artifact overwrite."""
from pathlib import Path
import hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXE=ROOT/'build/booster-level-batch/ghb_bench'
SHAPES={
 'scalar': ['--rows','65536','--test-rows','2048','--features','32','--rounds','10','--depth','5','--bins','64'],
 '33deep':['--outputs','33','--rows','8192','--test-rows','256','--features','16','--rounds','3','--depth','5','--bins','32'],
 '129':['--outputs','129','--rows','4096','--test-rows','256','--features','16','--rounds','3','--depth','2','--bins','32'],
 '1024':['--objective','binary','--outputs','1024','--rows','4096','--test-rows','256','--features','16','--rounds','2','--depth','2','--bins','16'],
 '4096':['--objective','binary','--outputs','4096','--rows','1024','--test-rows','64','--features','16','--rounds','1','--depth','2','--bins','16'],
 'multiclass17':['--objective','multiclass','--classes','17','--rows','4096','--test-rows','512','--features','16','--rounds','3','--depth','3','--bins','32'],
 'fallback33':['--outputs','33','--rows','2048','--test-rows','128','--features','40','--rounds','2','--depth','3','--bins','256']}
def telemetry():
 c=['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu','--format=csv']
 r=subprocess.run(c,capture_output=True,text=True);return dict(command=c,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def run(name,args):
 dest=OUT/name;capture=OUT/(name+'-capture.json')
 if dest.exists() or capture.exists():raise RuntimeError('refusing overwrite '+name)
 cmd=[str(EXE),*args,'--output-dir',str(dest)]
 digest=hashlib.sha256(EXE.read_bytes()).hexdigest();rec=dict(command=cmd,executable_sha256=digest,before=telemetry())
 start=time.monotonic();r=subprocess.run(cmd,cwd=ROOT,capture_output=True)
 rec.update(returncode=r.returncode,wall_seconds=time.monotonic()-start,executable_unchanged=digest==hashlib.sha256(EXE.read_bytes()).hexdigest(),after=telemetry())
 (OUT/(name+'-stdout.json')).write_bytes(r.stdout);(OUT/(name+'-stderr.txt')).write_bytes(r.stderr);capture.write_text(json.dumps(rec,indent=2)+'\n')
 if r.returncode or not rec['executable_unchanged']:raise RuntimeError(name+' failed')
 data=json.loads(r.stdout);assert data==json.loads((dest/'result.json').read_text());assert not (dest/'.incomplete').exists()
 print(name,data['timing'],flush=True)
if __name__=='__main__':
 for case,args in SHAPES.items():
  for execution in ['stream','graph']:
   for suffix,modes in [('a',['per-output','output-batch']),('b',['output-batch','per-output'])]:
    for mode in modes:run(f'synthetic-{case}-{execution}-{mode}-{suffix}',args+['--seed','20260922701','--instrumentation','off','--tree-execution',execution,'--tree-build',mode])
