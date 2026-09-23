import argparse,json,pathlib,re,subprocess,time
out=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True,help='new run label; historical evidence is never overwritten')
args=parser.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.label): parser.error('label must contain only letters, digits, underscores or hyphens')
cases=json.loads((out/'matrix-plan.json').read_text())
for case in cases:
 target=out/(args.label+'-'+case['name'])
 if target.exists(): parser.error('evidence already exists: '+str(target))
 case['command'][case['command'].index('--output')+1]=str(target)
 case['output']=str(target)
preflight=out/(args.label+'-preflight.json')
if preflight.exists(): parser.error('preflight already exists: '+str(preflight))
p=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=15)
preflight.write_text(json.dumps({'command':['nvidia-smi'],'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n')
if p.returncode:
 print('Driver preflight failed; no GPU workloads launched. See '+str(preflight),flush=True)
 raise SystemExit(1)
results=[]
for case in cases:
 start=time.monotonic()
 p=subprocess.run(case['command'],capture_output=True,text=True)
 item={'name':case['name'],'exit_code':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr}
 results.append(item);(out/(args.label+'-results.json')).write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(item),flush=True)
raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)
