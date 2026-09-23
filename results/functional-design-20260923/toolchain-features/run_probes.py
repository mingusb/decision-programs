from pathlib import Path
import subprocess,json,hashlib,datetime,os
root=Path(__file__).resolve().parent
(root/'sources').mkdir(exist_ok=True); (root/'logs').mkdir(exist_ok=True); (root/'objects').mkdir(exist_ok=True)
incs=['-I/usr/local/cuda/include/cccl','-I/usr/local/cuda/include']
features={
 'saturation_current': ('numeric', 'return NS::saturating_add(x,1)+NS::saturating_sub(x,1)+NS::saturating_mul(x,2)+NS::saturating_div(x,2)+NS::saturating_cast<int>(static_cast<long long>(x));'),
 'saturation_legacy': ('numeric', 'return NS::add_sat(x,1)+NS::sub_sat(x,1)+NS::mul_sat(x,2)+NS::div_sat(x,2)+NS::saturate_cast<int>(static_cast<long long>(x));'),
 'function_ref': ('functional', 'auto fn = [] ANNOT (int y){return y+1;}; NS::function_ref<int(int)> ref(fn); return ref(x);'),
 'inplace_vector': ('inplace_vector', 'NS::inplace_vector<int,4> v; v.push_back(x); v.push_back(2); return v[0]+v[1];'),
 'fold_left': ('algorithm', 'const int a[]{x,2,3}; return NS::ranges::fold_left(a,0,[] ANNOT (int acc,int y){return acc+y;});'),
 'ranges_view': ('ranges', 'const int a[]{x,2,3}; auto v=a | NS::views::transform([] ANNOT (int y){return y+1;}); return v[0]+v[1]+v[2];'),
 'expected_monadic': ('expected', 'const NS::expected<int,int> v(x); const auto r=v.transform([] ANNOT (int y){return y+1;}).and_then([] ANNOT (int y){return NS::expected<int,int>(y+2);}); return *r;'),
 'forward_like': ('utility', 'return NS::forward_like<const int&>(x);'),
 'bind_back': ('functional', 'const auto fn=NS::bind_back([] ANNOT (int a,int b){return a+b;},2); return fn(x);'),
}
language={
 'pack_index': 'template<class... Ts> ANNOT constexpr auto first(Ts... xs){return xs...[0];}\nANNOT int probe(int x){return first(x,2,3);}',
 'deducing_this': 'struct F { ANNOT constexpr int operator()(this auto self,int x){return x<=0?0:1+self(x-1);} };\nANNOT int probe(int x){return F{}(x);}',
 'static_call_operator': 'struct F { ANNOT static constexpr int operator()(int x){return x+1;} };\nANNOT int probe(int x){return F{}(x);}',
 'if_consteval': 'ANNOT constexpr int f(int x){if consteval {return x+1;} else {return x+2;}}\nstatic_assert(f(1)==2);\nANNOT int probe(int x){return f(x);}',
 'fold_expression': 'template<class... T> ANNOT constexpr auto sum(T... xs){return (0+...+xs);}\nANNOT int probe(int x){return sum(x,2,3);}',
}
records=[]
def run(name,cmd):
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  p=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,env={**os.environ,'PATH':'/usr/local/cuda/bin:/usr/bin:/bin'})
  stdout,stderr,code=p.stdout,p.stderr,p.returncode
 except subprocess.TimeoutExpired as e:
  stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else (e.stdout or '')
  stderr=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else (e.stderr or '')
  code='timeout'
 (root/'logs'/f'{name}.stdout').write_text(stdout)
 (root/'logs'/f'{name}.stderr').write_text(stderr)
 records.append({'name':name,'command':cmd,'started_utc':stamp,'exit_code':code,'stdout':f'logs/{name}.stdout','stderr':f'logs/{name}.stderr'})
 (root/'partial-results.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name,code,flush=True)
 return code
run('nvcc_version',['/usr/local/cuda/bin/nvcc','--version']);run('gcc_version',['/usr/bin/g++','--version']);run('nvcc_help',['/usr/local/cuda/bin/nvcc','--help'])
for mode in ['gcc_host','nvcc_host','nvcc_device']:
 for ns in ['std','cuda_std']:
  for feature,(header,body) in features.items():
   name=f'{mode}_{ns}_{feature}'
   anno='__device__' if mode=='nvcc_device' else ''
   source=f'#include <'+ ('cuda/std/' if ns=='cuda_std' else '')+header+'>\n'
   source+=f'{anno} int probe(int x){{'+body.replace('NS','cuda::std' if ns=='cuda_std' else 'std').replace('ANNOT',anno)+'}\n'
   if mode=='nvcc_device': source+='extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}\n'
   suffix='.cpp' if mode=='gcc_host' else '.cu'
   path=root/'sources'/f'{name}{suffix}';path.write_text(source)
   output=root/'objects'/f'{name}'+Path('') if False else root/'objects'/f'{name}{".ptx" if mode=="nvcc_device" else ".o"}'
   if mode=='gcc_host': cmd=['/usr/bin/g++','-std=c++23','-O2','-pedantic-errors',*incs,'-c',str(path),'-o',str(output)]
   else: cmd=['/usr/local/cuda/bin/nvcc','-std=c++23','-O2','-arch=sm_86','-ccbin=/usr/bin/g++',*incs,'--ptx' if mode=='nvcc_device' else '-c',str(path),'-o',str(output)]
   run(name,cmd)
 for feature,body in language.items():
  name=f'{mode}_language_{feature}';anno='__device__' if mode=='nvcc_device' else ''
  source=body.replace('ANNOT',anno)+'\n'
  if mode=='nvcc_device':source+='extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}\n'
  path=root/'sources'/f'{name}{".cpp" if mode=="gcc_host" else ".cu"}';path.write_text(source)
  output=root/'objects'/f'{name}{".ptx" if mode=="nvcc_device" else ".o"}'
  if mode=='gcc_host':cmd=['/usr/bin/g++','-std=c++23','-O2','-pedantic-errors','-c',str(path),'-o',str(output)]
  else:cmd=['/usr/local/cuda/bin/nvcc','-std=c++23','-O2','-arch=sm_86','-ccbin=/usr/bin/g++','--ptx' if mode=='nvcc_device' else '-c',str(path),'-o',str(output)]
  run(name,cmd)
# Explicit dialect rejection and host-only C++26 controls; never executes binaries.
small=root/'sources'/'dialect.cu';small.write_text('int probe(int x){return x+1;}\n')
run('nvcc_cxx26_dialect',['/usr/local/cuda/bin/nvcc','-std=c++26','-c',str(small),'-o',str(root/'objects'/'dialect.o')])
for f in ['saturation_legacy','saturation_current','function_ref','inplace_vector']:
 path=root/'sources'/f'gcc_host_std_{f}.cpp'
 run('gcc_cxx26_'+f,['/usr/bin/g++','-std=c++26','-O2','-pedantic-errors','-c',str(path),'-o',str(root/'objects'/f'gcc_cxx26_{f}.o')])
run('gcc_cxx26_pack_index',['/usr/bin/g++','-std=c++26','-O2','-pedantic-errors','-c',str(root/'sources'/'gcc_host_language_pack_index.cpp'),'-o',str(root/'objects'/'gcc_cxx26_pack_index.o')])
result={'schema':'ghb.compile_only_feature_probe.v1','gpu_used':False,'binary_execution':False,'dialect':'c++23 except explicitly named cxx26 controls','cases':records}
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print('DONE',len(records),flush=True)
