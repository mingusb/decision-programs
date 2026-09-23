import argparse, datetime, hashlib, json, math, sqlite3, sys
from pathlib import Path
API=Path('/opt/nvidia/nsight-compute/2026.3.0/extras/python')
sys.path.insert(0,str(API))
import ncu_report
ROOT=Path('/home/b/gpu_histogram/results/profiling-tools-20260922')
def identity(p):
 p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def scalar(m):
 if m is None:return None
 return {'aggregate':m.as_double(),'unit':m.unit(),'num_instances':m.num_instances()}
def timeline(m,a,full=False):
 c=m.correlation_ids();vs=[m.as_double(i) for i in range(m.num_instances())];ts=[c.as_uint64(i) for i in range(c.num_instances())] if c else []
 # Retain the ITimelines owner while reading its returned metric proxy.
 timelines_owner=a.timelines();aligned=timelines_owner.aligned_metric(m)
 result={'name':m.name(),**scalar(m),'finite_samples':sum(math.isfinite(v) for v in vs),'positive_samples':sum(v>0 for v in vs),'zero_samples':sum(v==0 for v in vs),'sample_min':min(vs) if vs else None,'sample_max':max(vs) if vs else None,'timestamp_instances':len(ts),'timestamps_strictly_increasing':all(x<y for x,y in zip(ts,ts[1:])),'first_timestamp_ns':ts[0] if ts else None,'last_timestamp_ns':ts[-1] if ts else None,'aligned_instances':aligned.num_instances() if aligned else None}
 if full:result.update(timestamps_ns=ts,sample_values=vs)
 return result

def ncu(kind):
 d=ROOT/f'{args.ncu_prefix}-{kind}-ncu';p=d/'profile.ncu-repz';r=ncu_report.load_report(str(p));actions=[]
 for ri in range(r.num_ranges()):
  rg=r.range_by_idx(ri)
  for ai in range(rg.num_actions()):
   a=rg.action_by_idx(ai);names=a.metric_names();groups=a.metric_by_name('profiler__pmsampler_pass_groups');registered=[]
   for i in range(groups.num_instances()):registered.extend((groups.as_string(i) or '').split(','))
   pm_names=[name for name in names if name.startswith('pmsampling:') or 'Triage' in name]
   pm=[timeline(a.metric_by_name(name),a, name in ['SM_A.TriageAC.sm__cycles_active.avg.per_cycle_elapsed','SM_A.TriageAC.sm__inst_executed_realtime.avg.per_cycle_elapsed']) for name in pm_names]
   m=a.metric_by_name('inst_executed');ids=m.correlation_ids();sass=[]
   for i in range(m.num_instances()):
    pc=ids.as_uint64(i);s=a.source_info(pc)
    sass.append({'pc':pc,'warp_instruction_executions':m.as_uint64(i),'sass':a.sass_by_pc(pc),'source_file':s.file_name() if s else None,'source_line':s.line() if s else None})
   op=a.metric_by_name('sass__inst_executed_per_opcode');oi=op.correlation_ids()
   active=next(x for x in pm if x['name']=='SM_A.TriageAC.sm__cycles_active.avg.per_cycle_elapsed')
   pm_pass=active['positive_samples']>0 and active['timestamps_strictly_increasing'] and active['finite_samples']==active['num_instances'] and active['name'] in registered
   sass_pass=m.as_uint64()>0 and any(x['warp_instruction_executions']>0 and x['sass'] and x['source_file'] for x in sass)
   actions.append({'kernel':a.name(ncu_report.IAction.NameBase_DEMANGLED),'sections':[s.identifier() for s in a.sections()],'pm_sampling':{'passed':pm_pass,'timeline_count':len(pm),'pass_group_count':groups.as_uint64(),'pass_group_metrics':[groups.as_string(i) for i in range(groups.num_instances())],'interval_time':scalar(a.metric_by_name('profiler__pmsampler_interval_time')),'merged_samples':scalar(a.metric_by_name('profiler__pmsampler_merged_samples')),'timelines':pm},'source_counters':{'passed':sass_pass,'inst_executed':scalar(m),'sum_pc_instruction_executions':sum(x['warp_instruction_executions'] for x in sass),'source_correlated_pc_count':sum(bool(x['source_file']) for x in sass),'sass_pc_count':sum(bool(x['sass']) for x in sass),'embedded_source_files':list(a.source_files()),'opcode_counts':{oi.as_string(i):op.as_uint64(i) for i in range(op.num_instances())},'pc_samples':scalar(a.metric_by_name('smsp__pcsamp_sample_count')),'pc_sampling_dropped_bytes':scalar(a.metric_by_name('smsp__pcsamp_dropped_bytes')),'pc_records':sass}})
 return {'report':identity(p),'capture_manifest':identity(d/'manifest.json'),'passed':all(a['pm_sampling']['passed'] and a['source_counters']['passed'] for a in actions),'actions':actions}

def nsys(dirname):
 d=ROOT/dirname;manifest=json.loads((d/'manifest.json').read_text());result={'directory':str(d),'capture_manifest':identity(d/'manifest.json'),'runner_status':manifest.get('status','passed' if manifest.get('exit_code')==0 and not manifest.get('timed_out') else 'failed'),'runner_exit_code':manifest.get('runner_exit_code',manifest.get('exit_code')),'process':manifest.get('process',{'exit_code':manifest.get('exit_code'),'timed_out':manifest.get('timed_out'),'seconds':manifest.get('seconds')}),'reports':[identity(p) for p in sorted(d.glob('profile*.nsys-rep'))],'sqlite':[]}
 for p in sorted(d.glob('profile*.sqlite')):
  c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
  tables={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  nvtx=[dict(x) for x in c.execute('SELECT COALESCE(n.text,s.value) AS name,COUNT(*) AS events FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id GROUP BY 1')] if 'NVTX_EVENTS' in tables else []
  counts=dict(c.execute('SELECT COUNT(*) AS kernels,SUM(graphNodeId IS NOT NULL AND graphNodeId!=0) AS graph_node_kernels,COUNT(DISTINCT CASE WHEN graphNodeId IS NOT NULL AND graphNodeId!=0 THEN graphId END) AS distinct_kernel_graphs FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()) if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else {}
  kernels=[dict(x) for x in c.execute('SELECT s.value AS name,COUNT(*) AS instances,SUM(graphNodeId IS NOT NULL AND graphNodeId!=0) AS graph_instances FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id GROUP BY 1')] if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else []
  result['sqlite'].append({'artifact':identity(p),'nvtx_events':nvtx,'counts':counts,'kernels':kernels});c.close()
 result['populated_capture']=bool(result['reports']) and any(x['counts'].get('kernels',0)>0 for x in result['sqlite'])
 return result

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--nsys',action='append',default=[]);ap.add_argument('--ncu-prefix',default='driver61692');args=ap.parse_args()
proof={'schema':'ghb.section_specific_proof.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':{'ncu':'Offline ncu_report.load_report; IAction.metric_names/metric_by_name, IMetric.num_instances/as_double/correlation_ids, IAction.timelines().aligned_metric, IAction.source_info/sass_by_pc. PM includes pmsampling: names AND Triage names enumerated in profiler__pmsampler_pass_groups.','nsys':'Read-only SQLite exported by the existing wrapper; NVTX_EVENTS joined to StringIds and CUPTI_ACTIVITY_KIND_KERNEL graphNodeId/graphId; failed original capture retained separately.','command':sys.argv,'extractor':identity(__file__),'ncu_python_api':identity(API/'ncu_report.py'),'official_local_references':[{'path':'/opt/nvidia/nsight-compute/2026.3.0/docs/ProfilingGuide/index.html','lines':[2745,2762,2838,2843,2891],'supports':['merged_samples counts merging due to hardware backpressure, not total timeline samples','PM sampling names may use pmsampling prefix or valid Triage group','sample correlation IDs are GPU timestamps in ns','WSL does not support PM context-switch trace']},{'path':'/opt/nvidia/nsight-compute/2026.3.0/extras/python/ncu_report.py','lines':[2567,2594,3100,3130,3254,3299]}]},'ncu':{k:ncu(k) for k in ['count','booster']},'nsys':[nsys(d) for d in args.nsys],'limits':['Diagnostic profiler captures are not uninstrumented performance ranking evidence.','Only the selected kernels and recorded launches are established here; this is not complete-kernel coverage.','PM sampling is device-wide; WSL lacks context-switch filtering, so isolated serial captures do not mathematically exclude unrelated GPU activity.','Zero XU/tensor timeline values remain zero; positive proof uses SM-active/SM-instruction Triage timelines.','Sample values and report aggregates are preserved exactly; report aggregate is not relabeled as arithmetic mean of raw samples.','SASS/source PC and line correlation are present; source file text was not embedded in these NCU reports.']}
with args.output.open('x') as f:json.dump(proof,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'output':str(args.output),'ncu_passed':{k:v['passed'] for k,v in proof['ncu'].items()},'nsys':[(x['directory'],x['runner_status'],x['populated_capture']) for x in proof['nsys']]}))
