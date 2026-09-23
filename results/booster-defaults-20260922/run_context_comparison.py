"""Serial same-binary comparison under otherwise actual API defaults."""
from pathlib import Path
import hashlib,json,subprocess,time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXE=ROOT/'build/booster-reuse/ghb_bench'
POLICIES={
 'old':['--root-histogram','per-tree','--split-policy','block256','--root-counts','per-output','--split-batch','per-tree'],
 'candidate':['--root-histogram','batched','--split-policy','warp32','--root-counts','reuse-global','--split-batch','root']}
SHAPES={
 'scalar':['--rows','65536','--test-rows','8192','--features','32','--rounds','10','--depth','5','--bins','64'],
 '129':['--outputs','129','--rows','4096','--test-rows','256','--features','16','--rounds','3','--depth','2','--bins','32'],
 '1024':['--objective','binary','--outputs','1024','--rows','4096','--test-rows','256','--features','16','--rounds','2','--depth','2','--bins','16'],
 '4096':['--objective','binary','--outputs','4096','--rows','1024','--test-rows','64','--features','16','--rounds','1','--depth','2','--bins','16'],
 'multiclass17':['--objective','multiclass','--classes','17','--rows','4096','--test-rows','512','--features','16','--rounds','3','--depth','2','--bins','32'],
 'fallback33':['--outputs','33','--rows','2048','--test-rows','128','--features','40','--rounds','2','--depth','2','--bins','256']}
def telemetry():
 c=['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu','--format=csv']
 r=subprocess.run(c,capture_output=True,text=True);return dict(command=c,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def run(name,args):
 destination=OUT/name;capture=OUT/(name+'-capture.json')
 if destination.exists() or capture.exists():raise RuntimeError('refusing overwrite '+name)
 command=[str(EXE),*args,'--output-dir',str(destination)]
 digest=hashlib.sha256(EXE.read_bytes()).hexdigest();record=dict(command=command,executable_sha256=digest,before=telemetry())
 start=time.monotonic();r=subprocess.run(command,cwd=ROOT,capture_output=True)
 record.update(returncode=r.returncode,wall_seconds=time.monotonic()-start,executable_unchanged=digest==hashlib.sha256(EXE.read_bytes()).hexdigest(),after=telemetry())
 (OUT/(name+'-stdout.json')).write_bytes(r.stdout);(OUT/(name+'-stderr.txt')).write_bytes(r.stderr);capture.write_text(json.dumps(record,indent=2)+'\n')
 if r.returncode or not record['executable_unchanged']:raise RuntimeError(name+' failed')
 data=json.loads(r.stdout);assert data==json.loads((destination/'result.json').read_text())
 assert not (destination/'.incomplete').exists()
 print(name,data['timing'],flush=True)
if __name__=='__main__':
 for case,args in SHAPES.items():
  for suffix,modes in [('a',['old','candidate']),('b',['candidate','old'])]:
   for mode in modes:run(f'context-{case}-{mode}-{suffix}',args+['--seed','20260922605','--instrumentation','off']+POLICIES[mode])
