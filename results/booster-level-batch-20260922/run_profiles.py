"""Run profiler diagnostics after separately completed sanitizer checks."""
from run_gpu_diagnostics import checked, PREFIX, BUILD
if __name__ == "__main__":
 common=[f'{BUILD}/ghb_bench','--rows','4096','--test-rows','128','--features','16','--depth','3','--bins','32','--output-tile','16','--tree-export-batch','16']
 for mode,execution in [('per-output','graph'),('output-batch','graph'),('output-batch','stream')]:
  name=f'nsys-{mode}-{execution}'
  args=common+['--outputs','129','--rounds','3','--instrumentation','nvtx','--tree-build',mode,'--tree-execution',execution,'--output-dir',f'{PREFIX}/{name}-benchmark']
  checked(name,['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--cuda-graph-trace=node','--sample=none','--output',f'{PREFIX}/{name}',*args])
  checked(name+'-stats',['/usr/local/bin/nsys','stats','--report','cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum','--format','csv','--output',f'{PREFIX}/{name}-stats',f'{PREFIX}/{name}.nsys-rep'])
 ncu='/opt/nvidia/nsight-compute/2026.3.0/ncu'
 for name,kernel in [('deeper','regex:.*global_accumulate.*'),('materialize','regex:.*materialize_small.*'),('split','regex:.*warp_candidates.*')]:
  tag='ncu-'+name
  args=common+['--outputs','16','--rounds','1','--instrumentation','off','--tree-build','output-batch','--tree-execution','stream','--output-dir',f'{PREFIX}/{tag}-benchmark']
  checked(tag,[ncu,'--set','full','--kernel-name',kernel,'--launch-count','1','--clock-control','none','--export',f'{PREFIX}/{tag}',*args])
  for page in ['details','raw']:
   checked(tag+'-'+page,[ncu,'--import',f'{PREFIX}/{tag}.ncu-repz','--page',page,'--csv'])
