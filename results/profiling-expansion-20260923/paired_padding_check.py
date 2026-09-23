"""Unprofiled paired end-to-end observation, not a tuning promotion gate."""
from pathlib import Path
import subprocess,time,json,hashlib,statistics,datetime
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent/'padding-paired';out.mkdir(exist_ok=False)
binaries={'before':root/'build/profiling-booster/ghb_bench','after':root/'build/diagnostics-runtime-booster/ghb_bench'}
args=['--objective','binary','--rows','4096','--test-rows','1024','--features','16','--outputs','3','--output-tile','2','--rounds','3','--depth','3','--bins','33','--histogram','global','--tree-build','output-batch','--tree-execution','graph','--tree-export-batch','0','--instrumentation','off']
records=[]
for pair in range(8):
 for name in (['before','after'] if pair%2==0 else ['after','before']):
  command=[str(binaries[name])]+args;stem=out/f'{pair:02d}-{name}';started=time.perf_counter()
  p=subprocess.run(command,capture_output=True,text=True,timeout=90)
  stem.with_suffix('.stdout').write_text(p.stdout);stem.with_suffix('.stderr').write_text(p.stderr)
  record={'pair':pair,'variant':name,'argv':command,'exit':p.returncode,'elapsed':time.perf_counter()-started,'binary_sha256':hashlib.sha256(binaries[name].read_bytes()).hexdigest(),'warmup_pair':pair==0}
  if p.returncode==0:record['result']=json.loads(next(z for z in reversed(p.stdout.splitlines()) if z.startswith('{')))
  records.append(record);(out/'raw.json').write_text(json.dumps(records,indent=2));print(pair,name,p.returncode,flush=True)
measure=[x for x in records if not x['warmup_pair']];results={}
for field in ['training_ms','total_train_ms','gpu_predict_wall_ms']:
 ratios=[];diff=[]
 for pair in range(1,8):
  a=next(x['result']['timing'][field] for x in measure if x['pair']==pair and x['variant']=='before');b=next(x['result']['timing'][field] for x in measure if x['pair']==pair and x['variant']=='after');ratios.append(b/a);diff.append(b-a)
 results[field]={'before_median_ms':statistics.median(x['result']['timing'][field] for x in measure if x['variant']=='before'),'after_median_ms':statistics.median(x['result']['timing'][field] for x in measure if x['variant']=='after'),'paired_ratio_median':statistics.median(ratios),'paired_ratio_min':min(ratios),'paired_ratio_max':max(ratios),'paired_delta_ms':diff}
quality=[]
for pair in range(1,8):
 a=next(x['result'] for x in measure if x['pair']==pair and x['variant']=='before');b=next(x['result'] for x in measure if x['pair']==pair and x['variant']=='after')
 quality.append({'pair':pair,'validation_equal':a['validation']==b['validation'],'training_loss_exact_equal':a['training_loss']==b['training_loss'],'heldout_exact_equal':a['heldout']==b['heldout'],'trees_equal':a['trees']==b['trees'],'heldout_loss_delta':b['heldout']['loss']-a['heldout']['loss']})
summary={'scope':'Seven alternating-order process pairs after one discarded warmup pair; same workload; uninstrumented runtime; no statistical proof of zero overhead or universal quality gate','timings':results,'quality':quality}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
