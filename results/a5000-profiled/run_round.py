#!/usr/bin/env python3
"""Serial, recorded experiments for the third A5000 pass; root owns GPU execution."""
import argparse, csv, datetime, hashlib, io, json, pathlib, platform, shutil, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/a5000-profiled'
EXE = ROOT / 'build/histogram_bench'
ACTIVE_ENVIRONMENT = None
CASES = [
 ('small8', 1<<20, 8, 'u32', 'u32', 'uniform', 'shuffled', 'warm'),
 ('smallbyte', 4096, 256, 'u8', 'u32', 'uniform', 'shuffled', 'warm'),
 ('cachedbyte', 1<<20, 256, 'u8', 'u32', 'uniform', 'shuffled', 'warm'),
 ('byte', 1<<24, 256, 'u8', 'u32', 'uniform', 'shuffled', 'warm'),
 ('hot99', 1<<20, 256, 'u32', 'u32', 'hot99', 'shuffled', 'warm'),
 ('sortedhot99', 1<<20, 256, 'u32', 'u32', 'hot99', 'sorted', 'warm'),
 ('single', 1<<20, 256, 'u32', 'u32', 'single', 'shuffled', 'warm'),
 ('large4096', 1<<24, 4096, 'u32', 'u32', 'uniform', 'shuffled', 'warm'),
 ('large4096-u64', 1<<24, 4096, 'u32', 'u64', 'uniform', 'shuffled', 'warm'),
 ('large8192-u64', 1<<24, 8192, 'u32', 'u64', 'uniform', 'shuffled', 'warm'),
 ('large16384-u64', 1<<24, 16384, 'u32', 'u64', 'uniform', 'shuffled', 'warm'),
 ('cold4096-u64', 1<<20, 4096, 'u32', 'u64', 'uniform', 'shuffled', 'cold'),
]
def args(case):
 _, n,b,i,c,d,o,k = case
 return ['--n',str(n),'--bins',str(b),'--input',i,'--counter',c,'--distribution',d,'--order',o,'--cache',k,'--launch','graph']
def telemetry():
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv'],capture_output=True,text=True)
 return p.stdout or p.stderr

def sha256(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def gpu_environment():
 # UUID is unavailable on some nvidia-smi interfaces. Driver/name remain required.
 for fields in ('driver_version,name,uuid', 'driver_version,name'):
  p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=False)
  if p.returncode:
   if fields.endswith(',uuid'): continue
   raise RuntimeError('cannot record GPU environment: '+p.stderr.strip())
  rows=list(csv.reader(io.StringIO(p.stdout),skipinitialspace=True))
  names=fields.split(',')
  if not rows or any(len(row)!=len(names) for row in rows):
   raise RuntimeError('invalid nvidia-smi environment output')
  devices=[]
  for row in rows:
   device=dict(zip(names,(value.strip() for value in row)))
   if not device['driver_version'] or not device['name']:
    raise RuntimeError('missing GPU driver/name in nvidia-smi output')
   device.setdefault('uuid',None)
   if device['uuid'] in ('N/A','[N/A]','Not Supported','[Not Supported]'):
    device['uuid']=None
   devices.append(device)
  return devices

def ensure_environment(output_root):
 global ACTIVE_ENVIRONMENT
 current={
  'schema':1,
  'session_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'gpus':gpu_environment(),
  'binary':str(EXE.resolve()),
  'binary_sha256':sha256(EXE),
  'uname':dict(platform.uname()._asdict()),
 }
 output_root.mkdir(parents=True,exist_ok=True)
 path=output_root/'environment.json'
 if path.exists():
  recorded=json.loads(path.read_text())
  stable=('schema','gpus','binary','binary_sha256','uname')
  changed=[field for field in stable if recorded.get(field)!=current[field]]
  if changed:
   raise RuntimeError(f'{path}: environment changed ({", ".join(changed)}); use a new --output-root')
  if not recorded.get('session_started_utc'):
   raise RuntimeError(f'{path}: missing session_started_utc')
  ACTIVE_ENVIRONMENT=recorded
  return recorded
 # Do not attach a new environment identity to measurements collected earlier.
 existing=[str(item) for directory in ('plans','confirmation','timing-after','reconfirmation')
           if (output_root/directory).exists() for item in (output_root/directory).rglob('*') if item.is_file()]
 if existing:
  raise RuntimeError(f'{path}: refusing to adopt existing artifacts without environment metadata: {existing[0]}')
 with path.open('x') as output:
  json.dump(current,output,indent=2)
  output.write('\n')
 ACTIVE_ENVIRONMENT=current
 return current

def executable_path(value):
 path=pathlib.Path(value)
 if not path.is_absolute():
  resolved=shutil.which(str(value)) if '/' not in str(value) else str(ROOT/path)
  if resolved is None: raise RuntimeError(f'executable not found: {value}')
  path=pathlib.Path(resolved)
 return path.resolve()

def run(stem, command, suffix='.csv'):
 stem.parent.mkdir(parents=True,exist_ok=True)
 command=[str(x) for x in command]
 artifacts=[stem.with_suffix(extension) for extension in (suffix,'.log','.command.json')]
 existing=[str(path) for path in artifacts if path.exists()]
 if existing: raise RuntimeError('refusing overwrite: '+', '.join(existing))
 executable=executable_path(command[0])
 # The direct executable is Python for autotune; --exe identifies the benchmark.
 benchmark=executable_path(command[command.index('--exe')+1]) if '--exe' in command else executable
 executable_hash=sha256(executable)
 binary_hash=sha256(benchmark)
 gpus_before=gpu_environment()
 if ACTIVE_ENVIRONMENT is not None and gpus_before!=ACTIVE_ENVIRONMENT['gpus']:
  raise RuntimeError('GPU/driver environment changed before command: '+str(stem))
 before=telemetry(); start=time.monotonic()
 p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=3600)
 stem.with_suffix('.log').write_text(p.stderr)
 unchanged=sha256(executable)==executable_hash and sha256(benchmark)==binary_hash
 gpus_after=gpu_environment()
 stem.with_suffix('.command.json').write_text(json.dumps(dict(command=command,seconds=time.monotonic()-start,exit_code=p.returncode,executable=str(executable),executable_sha256=executable_hash,binary=str(benchmark),binary_sha256=binary_hash,executables_unchanged=unchanged,gpus_before=gpus_before,gpus_after=gpus_after,telemetry_before=before,telemetry_after=telemetry()),indent=2)+'\n')
 if not unchanged: raise RuntimeError('executable changed during command: '+str(stem))
 if gpus_before!=gpus_after: raise RuntimeError('GPU/driver environment changed during command: '+str(stem))
 if p.returncode: raise RuntimeError(p.stderr[-4000:])
 stem.with_suffix(suffix).write_text(p.stdout)
 print('Completed',stem.name,flush=True)

def key(row):
 value=f"{row['algorithm']}:{row['tuning']}:{row['blocks']}:{row['local_counter']}"
 return value+':'+row['clear_policy'] if 'clear_policy' in row else value

def plan_schema(plan):
 schema=plan.get('schema')
 if schema not in (3,4,5): raise RuntimeError('unsupported plan schema')
 if schema==5 and plan['chosen']['algorithm'] in ('cub','nvidia_sample256'):
  raise RuntimeError('schema5 plans must select a custom histogram, not a NVIDIA reference')
 return schema

def homogeneous_schema(directory):
 schemas={plan_schema(json.loads(path.read_text())) for case in CASES
          if (path:=directory/'plans'/f'{case[0]}.json').exists()}
 if len(schemas)>1:
  raise RuntimeError(f'{directory}: plans must have a single supported schema (3, 4 or 5)')
 return next(iter(schemas),None)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('stage',choices=('ladder','tune','confirm','reconfirm'))
 parser.add_argument('cases',nargs='*',help='selected case names; defaults to all cases for the stage')
 parser.add_argument('--output-root',type=pathlib.Path,default=OUT,help='session directory (must match its recorded environment on resume)')
 parser.add_argument('--source-root',type=pathlib.Path,help='reconfirm only: prior plans to evaluate with the current binary and explicit kernel clear; does not rerun selection')
 options=parser.parse_intermixed_args(argv)
 stage=options.stage
 selected=set(options.cases)
 allowed={case[0] for case in (CASES[:3] if stage=='ladder' else CASES)}
 if len(selected)!=len(options.cases): parser.error('duplicate case selection')
 if selected-allowed: parser.error('unknown cases for '+stage+': '+', '.join(sorted(selected-allowed)))
 if (stage=='reconfirm') != (options.source_root is not None):
  parser.error('--source-root is required exactly for the reconfirm stage')
 output_root=options.output_root.resolve()
 environment=ensure_environment(output_root)
 if stage in ('tune','confirm'):
  if (output_root/'reconfirmation').exists(): raise RuntimeError('cannot mix new selection with prior-selection reconfirmation')
  schema=homogeneous_schema(output_root)
  if schema is not None and schema!=5: raise RuntimeError('current tune/confirm requires schema5; use reconfirm in a new output directory for historical plans')
 if stage=='ladder':
  byte='cub:2:192,shared:2:192,shared_partial:3:96,nvidia_sample256:2:192'
  configs={'small8':'cub:2:192,shared:4:192,shared_partial:2:96,bitplane:1:192','smallbyte':byte,'cachedbyte':byte}
  for case in CASES[:3]:
   if selected and case[0] not in selected: continue
   for batch in (1,2,5,20,100):
    for seed in (67890,24680):
     name=f'{case[0]}-b{batch}-s{seed}'
     run(output_root/'timing-after'/name,[EXE]+args(case)+['--variants',configs[case[0]],'--batch',str(batch),'--samples','15','--seed',str(seed)])
 elif stage=='tune':
  for case in CASES:
   if selected and case[0] not in selected: continue
   plan=output_root/'plans'/f'{case[0]}.json'
   existing=[path for path in (plan,plan.with_suffix('.search.csv'),plan.with_suffix('.validation.csv')) if path.exists()]
   if existing: raise RuntimeError('refusing overwrite: '+', '.join(map(str,existing)))
   run(output_root/'plans'/f'{case[0]}-run',[sys.executable,ROOT/'tools/autotune.py','--exe',EXE,'--output',plan,'--batch','32','--search-samples','5','--validation-samples','15']+args(case),'.stdout')
 elif stage in ('confirm','reconfirm'):
  source_root=options.source_root.resolve() if stage=='reconfirm' else output_root
  if stage=='reconfirm':
   if source_root==output_root or (output_root/'plans').exists():
    raise RuntimeError('reconfirm requires a separate output directory without plans')
   source_schema=homogeneous_schema(source_root)
   for existing_path in (output_root/'reconfirmation').glob('*.json'):
    existing=json.loads(existing_path.read_text())
    if (pathlib.Path(existing['source_root']).resolve()!=source_root
        or existing['prior_plan_schema']!=source_schema):
     raise RuntimeError(f'{existing_path}: cannot mix prior selection directories or schemas')
  for case in CASES:
   if selected and case[0] not in selected: continue
   plan_path=source_root/'plans'/f'{case[0]}.json'
   plan=json.loads(plan_path.read_text())
   plan_schema(plan)
   if plan['validation']['batch']!=32: raise RuntimeError(f'{case[0]}: prior validation did not use batch32')
   if {plan['search']['seed'],*plan['validation']['seeds']} & {424242,987654}:
    raise RuntimeError(f'{case[0]}: confirmation seeds overlap selection seeds')
   if stage=='confirm' and plan['build']['sha256']!=sha256(EXE): raise RuntimeError(f'{case[0]}: plan benchmark hash differs from current executable')
   search_path=source_root/'plans'/f'{case[0]}.search.csv'
   if plan['search']['csv_sha256']!=sha256(search_path): raise RuntimeError(f'{case[0]}: search CSV hash differs from plan')
   search=list(csv.DictReader(search_path.open()))
   scalar=min((r for r in search if int(r['tuning'])<6 and r['algorithm'] not in ('cub','nvidia_sample256')),key=lambda r:float(r['median_us']))
   configs=[plan['chosen'],scalar]+[r for r in search if r['algorithm'] in ('cub','nvidia_sample256')]
   if stage=='reconfirm': configs=[dict(row,clear_policy='kernel') for row in configs]
   variants=[key(row) for row in configs]
   variants=list(dict.fromkeys(variants))
   if stage=='reconfirm':
    provenance={
     'schema':1,'kind':'frozen_prior_selection','case':case[0],
     'interpretation':'Evaluate prior chosen/scalar/reference configurations on the current binary with explicit kernel clearing. No new search or validation selection was performed.',
     'source_root':str(source_root),'prior_plan':str(plan_path),'prior_plan_sha256':sha256(plan_path),
     'prior_plan_schema':plan['schema'],'prior_binary_sha256':plan['build']['sha256'],
     'measurement_schema':5,
     'prior_search_csv':str(search_path),'prior_search_csv_sha256':sha256(search_path),
     'binary_sha256':environment['binary_sha256'],'clear_policy':'kernel',
     'expected_variants':variants,
    }
    provenance_path=output_root/'reconfirmation'/f'{case[0]}.json'
    provenance_path.parent.mkdir(parents=True,exist_ok=True)
    if provenance_path.exists():
     if json.loads(provenance_path.read_text())!=provenance:
      raise RuntimeError(f'{provenance_path}: prior-selection provenance changed')
    else:
     with provenance_path.open('x') as output:
      json.dump(provenance,output,indent=2); output.write('\n')
   for repeat,seed in ((1,424242),(2,424242),(3,987654)):
    workload=[]
    if stage=='reconfirm':
     for name,value in plan['workload'].items(): workload+=['--'+name.replace('_','-'),str(value)]
    else: workload=args(case)
    run(output_root/'confirmation'/f'{case[0]}-r{repeat}',[EXE]+workload+['--variants',','.join(variants),'--samples','21','--batch','32','--seed',str(seed)])
 return 0

if __name__ == '__main__':
 try:
  raise SystemExit(main())
 except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
  print('ERROR:',error,file=sys.stderr)
  raise SystemExit(2)
