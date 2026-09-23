"""Root-only separate device-wide memory diagnostics; excluded from rankings."""
from pathlib import Path
import json,os,subprocess,sys
from campaign import HERE,WORKSPACE,IMPLEMENTATIONS,command
root=HERE/'memory'
root.mkdir(exist_ok=False)
selection=json.loads((HERE/'test/selection.json').read_text())
env=os.environ.copy()
env.update(OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
libs=[WORKSPACE/'build/benchmark-env/lib/python3.12/site-packages/nvidia/nccl/lib',WORKSPACE/'build/benchmark-env/lib/python3.12/site-packages/nvidia/cu13/lib',Path('/usr/local/cuda/lib64')]
env['LD_LIBRARY_PATH']=':'.join(map(str,libs))+':'+env.get('LD_LIBRARY_PATH','')
observations=[]
for implementation in IMPLEMENTATIONS:
 name='delicious-'+implementation
 config=selection['delicious'][implementation]['config']
 cmd=command(implementation,config,'delicious','test',root/(name+'-result'),WORKSPACE/'build/booster-level-batch/ghb_real_bench')
 wrapped=[sys.executable,str(HERE/'memory_observation.py'),'--output',str(root/name),'--',*cmd]
 with (root/(name+'.stdout')).open('x') as stdout,(root/(name+'.stderr')).open('x') as stderr:
  result=subprocess.run(wrapped,env=env,cwd=WORKSPACE,stdout=stdout,stderr=stderr)
 observations.append(dict(implementation=implementation,command=wrapped,returncode=result.returncode,config=config))
 (root/'campaign.json').write_text(json.dumps(observations,indent=2)+'\n')
 print(name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
