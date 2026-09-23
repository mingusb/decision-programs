from pathlib import Path
import json,subprocess,datetime,os,hashlib
p=Path(__file__).resolve().parent
records=[]
def run(name,path,extra=()):
 output=p/'objects'/f'{name}.ptx'
 cmd=['/usr/local/cuda/bin/nvcc','-std=c++23','-O2','-arch=sm_86','-ccbin=/usr/bin/g++','-I/usr/local/cuda/include/cccl','-I/usr/local/cuda/include',*extra,'--ptx',str(path),'-o',str(output)]
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r=subprocess.run(cmd,cwd=p,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45,env={**os.environ,'PATH':'/usr/local/cuda/bin:/usr/bin:/bin'})
 (p/'logs'/f'{name}.stdout').write_text(r.stdout);(p/'logs'/f'{name}.stderr').write_text(r.stderr)
 records.append({'name':name,'command':cmd,'started_utc':stamp,'exit_code':r.returncode,'stdout':f'logs/{name}.stdout','stderr':f'logs/{name}.stderr'})
 print(name,r.returncode,flush=True)
for ns in ['std','cuda_std']:
 for feature in ['function_ref','fold_left','ranges_view','expected_monadic','bind_back']:
  old=p/'sources'/f'nvcc_device_{ns}_{feature}.cu'
  name=f'nvcc_device_implicit_lambda_{ns}_{feature}'
  src=old.read_text().replace('[] __device__','[]')
  path=p/'sources'/f'{name}.cu';path.write_text(src)
  run(name,path)
# Explicit annotations need the extended-lambda flag; keep separate flag cell.
for feature in ['ranges_view','expected_monadic']:
 name=f'nvcc_device_extended_lambda_cuda_std_{feature}'
 run(name,p/'sources'/f'nvcc_device_cuda_std_{feature}.cu',['--extended-lambda'])
# Bounded unsigned values, constexpr semantics and a runtime-dependent device
# expression exercise the actual backport, not just a header or static assert.
name='nvcc_device_saturation_bounded'
source='''#include <cuda/std/numeric>
#include <cuda/std/limits>
static_assert(cuda::std::saturating_add<unsigned>(~0u,1u)==~0u);
static_assert(cuda::std::saturating_sub<unsigned>(0u,1u)==0u);
static_assert(cuda::std::saturating_mul<unsigned>(~0u,2u)==~0u);
static_assert(cuda::std::saturating_div<int>(cuda::std::numeric_limits<int>::min(),-1)==cuda::std::numeric_limits<int>::max());
static_assert(cuda::std::saturating_cast<unsigned char>(1000u)==255u);
extern "C" __global__ void probe_kernel(const unsigned* in,unsigned* out){
 const unsigned x=*in;
 *out=cuda::std::saturating_add(x,1u)^cuda::std::saturating_sub(x,1u)^cuda::std::saturating_mul(x,2u)^cuda::std::saturating_div(x,2u)^cuda::std::saturating_cast<unsigned>(static_cast<unsigned long long>(x));
}
'''
path=p/'sources'/f'{name}.cu';path.write_text(source);run(name,path)
# Bounded capacity failure-returning API, no device exception recovery assumed.
name='nvcc_device_inplace_vector_try'
source='''#include <cuda/std/inplace_vector>
extern "C" __global__ void probe_kernel(const int* in,int* out){
 cuda::std::inplace_vector<int,1> v;
 const auto inserted=v.try_push_back(*in);
 const auto full=v.try_push_back(1);
 *out=inserted && !full ? v[0] : 0;
}
'''
path=p/'sources'/f'{name}.cu';path.write_text(source);run(name,path)
(p/'followup-results.json').write_text(json.dumps({'schema':'ghb.compile_only_feature_probe.v1','gpu_used':False,'binary_execution':False,'initial_failures_preserved':True,'cases':records},indent=2)+'\n')
print('DONE',len(records),flush=True)
