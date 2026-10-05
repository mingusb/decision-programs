// Public host transport/metadata dispatcher. Numerical work stays in the
// maintained CUDA backends. No shell interpolation or checkout paths are used.
#include "class_io.hpp"
#include "decision_programs_csv.hpp"
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <functional>
#include <map>
#include <set>
#include <sys/wait.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

namespace {
namespace fs=std::filesystem;
using J=nlohmann::json;
using Args=std::vector<std::string>;
void need(bool ok,const std::string& why){if(!ok)throw std::invalid_argument(why);}
const std::set<std::string> switches={"help","dry-run","check","gpu","stop-after-oof","all","list","diagnostic-trajectories"};
struct Options {
  std::map<std::string,Args> values; Args tail;
  Options(const Args& arguments){
    for(std::size_t i=0;i<arguments.size();++i){
      auto token=arguments[i];
      if(token=="--"){tail.assign(arguments.begin()+i+1,arguments.end());break;}
      need(token.starts_with("--"),"unexpected positional argument: "+token+"; use --help");
      token.erase(0,2);auto equal=token.find('=');std::string key=token.substr(0,equal),value;
      need(!key.empty(),"empty option");if(key=="output")key="out";
      if(equal!=std::string::npos)value=token.substr(equal+1);
      else if(switches.contains(key))value="true";
      else {need(i+1<arguments.size()&&!arguments[i+1].starts_with("--"),"--"+key+" requires a value");value=arguments[++i];}
      need(key=="set"||key=="trial"||key=="tool-arg"||!values.contains(key),"--"+key+" may occur only once");
      values[key].push_back(std::move(value));
    }
  }
  bool has(const std::string& key)const{return values.contains(key);}
  std::string get(const std::string& key,const std::string& fallback={})const{auto it=values.find(key);return it==values.end()?fallback:it->second.front();}
  std::string required(const std::string& key)const{auto v=get(key);need(!v.empty(),"missing --"+key+"; use --help");return v;}
  void allow(std::set<std::string> keys)const{
    keys.insert("help");keys.insert("dry-run");keys.insert("save-plan");keys.insert("set");
    for(const auto& [key,value]:values)need(keys.contains(key),"unknown option --"+key+"; use --help");
  }
};
fs::path absolute(const std::string& path){need(!path.empty(),"empty path");return fs::absolute(path).lexically_normal();}
J parse(const std::string& text){return J::parse(text);}
J file_json(const std::string& path){return parse(dpnative::read_text(absolute(path)));}
std::string pin(const fs::path& path){return dpnative::sha256(dpnative::read_text(path));}
J scalar(const std::string& text){return parse(text);}
J number(const Options&o,const char*key,std::uint64_t fallback){return o.has(key)?scalar(o.get(key)):J(fallback);}
void overrides(J& plan,const Options&o){
  if(!o.has("set"))return;
  for(const auto& assignment:o.values.at("set")){
    auto equal=assignment.find('=');need(equal!=std::string::npos&&equal>0,"--set requires /json/pointer=JSON_VALUE");
    auto name=assignment.substr(0,equal);need(name.front()=='/',"--set path must begin with / (JSON Pointer)");
    plan[J::json_pointer(name)]=parse(assignment.substr(equal+1));
  }
}
fs::path executable(){return fs::canonical("/proc/self/exe");}
fs::path model_file(const std::string&value){
  auto path=absolute(value);if(!fs::is_directory(path))return path;
  for(const char*name:{"model.canonical","model.clsgdag"})if(fs::is_regular_file(path/name))return path/name;
  std::vector<fs::path> candidates;for(const auto&e:fs::directory_iterator(path))if(e.is_regular_file()&&e.path().extension()==".canonical")candidates.push_back(e.path());
  need(candidates.size()==1,"model directory must contain model.canonical/model.clsgdag or exactly one .canonical file; select a trial explicitly");return candidates.front();
}
fs::path backend(const std::string& name,bool required=true){
  Args directories;if(const char*env=std::getenv("DECISION_PROGRAMS_BACKEND_DIR"))directories.emplace_back(env);
  auto bin=executable().parent_path();directories.push_back(bin.string());directories.push_back((bin/"../libexec/decision-programs").lexically_normal().string());
  for(const auto& directory:directories){auto path=absolute(directory)/name;if(fs::is_regular_file(path)&&access(path.c_str(),X_OK)==0)return path;}
  if(required)throw std::runtime_error("backend "+name+" is missing; build/install the CUDA targets or set DECISION_PROGRAMS_BACKEND_DIR");
  return {};
}
fs::path program(const std::string& name){
  if(name.find('/')!=std::string::npos){auto path=absolute(name);if(access(path.c_str(),X_OK)==0)return path;return {};}
  const char*env=std::getenv("PATH");std::stringstream paths(env?env:"");std::string dir;
  while(std::getline(paths,dir,':')){auto path=fs::path(dir.empty()?".":dir)/name;if(fs::is_regular_file(path)&&access(path.c_str(),X_OK)==0)return absolute(path.string());}
  return {};
}
volatile std::sig_atomic_t child_pid=0;
void forward_signal(int signal){if(child_pid>0)kill(pid_t(child_pid),signal);}
int run(const Args& command,std::string* captured=nullptr){
  need(!command.empty(),"empty command");int pipefd[2]{-1,-1};if(captured&&pipe(pipefd)!=0)throw std::runtime_error("pipe failed");
  pid_t child=fork();if(child<0)throw std::runtime_error("fork failed");
  if(child==0){
    if(captured){close(pipefd[0]);dup2(pipefd[1],STDOUT_FILENO);close(pipefd[1]);}
    std::vector<char*> argv;for(const auto&word:command)argv.push_back(const_cast<char*>(word.c_str()));argv.push_back(nullptr);
    execv(argv[0],argv.data());std::cerr<<"cannot execute "<<command[0]<<": "<<std::strerror(errno)<<'\n';_exit(127);
  }
  child_pid=child;auto old_int=std::signal(SIGINT,forward_signal),old_term=std::signal(SIGTERM,forward_signal),old_usr=std::signal(SIGUSR1,forward_signal);
  if(captured){close(pipefd[1]);char data[8192];for(;;){ssize_t n=read(pipefd[0],data,sizeof(data));if(n>0)captured->append(data,std::size_t(n));else if(n<0&&errno==EINTR)continue;else break;}close(pipefd[0]);}
  int status=0;while(waitpid(child,&status,0)<0){if(errno!=EINTR)throw std::runtime_error("waitpid failed");}
  child_pid=0;std::signal(SIGINT,old_int);std::signal(SIGTERM,old_term);std::signal(SIGUSR1,old_usr);
  return WIFEXITED(status)?WEXITSTATUS(status):WIFSIGNALED(status)?128+WTERMSIG(status):1;
}
struct Inputs {
  fs::path directory;J generated=J::object();
  Inputs(){std::string pattern=(fs::temp_directory_path()/"decision-programs-XXXXXX").string();std::vector<char> text(pattern.begin(),pattern.end());text.push_back(0);char* made=mkdtemp(text.data());if(!made)throw std::runtime_error("cannot create temporary input directory");directory=made;}
  ~Inputs(){std::error_code ignored;fs::remove_all(directory,ignored);}
  fs::path json(const std::string& name,const J& value){auto path=directory/name;dpnative::atomic_json(path,value);generated[name]=value;return path;}
  fs::path bytes(const std::string&name,const std::string&value){auto path=directory/name;dpnative::atomic_text(path,value);return path;}
};
int launch(const Args& command,const Options&o,const Inputs&inputs){
  if(o.has("save-plan")){
    auto out=absolute(o.get("save-plan"));need(!fs::exists(out),"--save-plan output already exists");
    need(inputs.generated.contains("plan.json"),"this command does not generate a plan");auto saved=inputs.generated.at("plan.json");
    dpnative::atomic_json(out,saved);
  }
  if(o.has("dry-run")){std::cout<<J{{"command",command},{"generated_inputs",inputs.generated},{"numerical_execution",false}}.dump(2)<<'\n';return 0;}
  const bool fresh_output=o.has("out")&&!fs::exists(absolute(o.get("out")));auto status=run(command);
  if(status==0&&fresh_output&&inputs.generated.contains("input-transport.json")&&o.has("out")){auto out=absolute(o.get("out"));if(fs::is_directory(out)&&!fs::exists(out/"input-transport.json"))dpnative::atomic_json(out/"input-transport.json",inputs.generated.at("input-transport.json"));}
  return status;
}
struct CsvFiles{fs::path values,labels;};
CsvFiles retain_csv(const std::string&original,const decision_programs_csv::Dense&dense,const std::string&identity){
  fs::path root;if(const char*env=std::getenv("DECISION_PROGRAMS_CACHE_DIR"))root=absolute(env);else if(const char*env=std::getenv("XDG_CACHE_HOME"))root=absolute(env)/"decision-programs";else if(const char*env=std::getenv("HOME"))root=absolute(env)/".cache/decision-programs";else root=fs::temp_directory_path()/"decision-programs-cache";
  auto dir=root/"inputs"/dpnative::sha256(identity);fs::create_directories(dir);int fd=open((dir/".lock").c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0600);need(fd>=0&&flock(fd,LOCK_EX)==0,"cannot lock retained CSV input");
  struct Guard{int fd;~Guard(){flock(fd,LOCK_UN);close(fd);}}guard{fd};
  auto retain=[&](const char*name,const std::string&bytes){auto path=dir/name;if(fs::exists(path))need(dpnative::read_text(path)==bytes,"retained CSV input changed/corrupt: "+path.string());else dpnative::atomic_text(path,bytes);return path;};
  retain("source.csv",original);return {retain("values.fp32",dense.values),retain("labels.u32",dense.labels)};
}
void remap_labels(decision_programs_csv::Dense&dense,const J&document){
  const auto&mapping=document.is_array()?document:document.at("class_label_mapping");need(mapping.is_array(),"label map must be an array or input-transport.json");if(dense.label_mapping.is_null()||dense.label_mapping.empty())return;
  std::map<std::string,std::uint32_t>required;std::set<std::uint32_t>used_ids;std::uint32_t classes=0;for(const auto&entry:mapping){const auto&value=entry.at("class");need(value.is_number_integer()&&(value.is_number_unsigned()||value.get<std::int64_t>()>=0),"label-map class must be a nonnegative integer");auto number=value.get<std::uint64_t>();need(number<UINT32_MAX,"label map class exceeds capacity");auto id=std::uint32_t(number);need(required.emplace(entry.at("label").get<std::string>(),id).second,"duplicate label-map label");need(used_ids.insert(id).second,"duplicate label-map class ID");classes=std::max(classes,id+1);}
  std::vector<std::uint32_t>translation(dense.classes);for(const auto&entry:dense.label_mapping){auto found=required.find(entry.at("label"));need(found!=required.end(),"evaluation CSV target is absent from the trained label map");translation.at(entry.at("class").get<std::uint32_t>())=found->second;}
  for(std::size_t at=0;at<dense.labels.size();at+=4){std::uint32_t id;std::memcpy(&id,dense.labels.data()+at,4);id=translation.at(id);std::memcpy(dense.labels.data()+at,&id,4);}dense.classes=classes;dense.label_mapping=mapping;
}
J dataset(const Options&o,bool fit_only,Inputs&input){
  if(o.has("data")){
    auto path=absolute(o.get("data"));if(path.extension()!=".csv")return file_json(path.string());
    auto original=dpnative::read_text(path);auto dense=decision_programs_csv::decode(original,o.required("target"),o.get("header","auto"));if(o.has("label-map"))remap_labels(dense,file_json(o.get("label-map")));
    if(o.has("classes")){auto declared=scalar(o.get("classes")).get<std::uint32_t>();need(declared>=dense.classes,"declared classes do not cover CSV labels");dense.classes=declared;}
    if(fit_only)need(dense.classes>=2,"classification training requires at least two declared classes");
    auto cached=retain_csv(original,dense,original+"\n"+o.get("target")+"\n"+o.get("header","auto")+"\n"+dense.label_mapping.dump());auto values=cached.values,labels=cached.labels;
    auto fit=fit_only?dense.rows:number(o,"fit-rows",0).get<std::uint64_t>();need(fit>0&&fit<=dense.rows,"CSV --fit-rows must be positive and not exceed total rows");
    J provenance={{"format","CSV-input-transport-1"},{"source_path",path.string()},{"source_sha256",dpnative::sha256(original)},{"header",dense.header},{"feature_names",dense.names},{"target",o.get("target")},{"class_label_mapping",dense.label_mapping},{"host_input_format_conversion",true},{"CPU_model_numerics",false}};input.json("input-transport.json",provenance);
    return {{"format","dense-fp32-u32-class-labels-1"},{"features",dense.features},{"classes",dense.classes},{"rows",dense.rows},{"row_stride",dense.features},{"FIT_rows",fit},{"VALID_rows",dense.rows-fit},{"values_path",values.string()},{"values_sha256",pin(values)},{"labels_path",labels.string()},{"labels_sha256",pin(labels)},{"preprocessing","CSV-to-FP32/u32 input transport; source="+path.string()+"; source_sha256="+dpnative::sha256(original)+"; target="+o.get("target")},{"TEST_read",false}};
  }
  auto values=absolute(o.required("values")),labels=absolute(o.required("labels"));
  auto rows=scalar(o.required("rows")),features=scalar(o.required("features")),classes=scalar(o.required("classes"));
  need(rows.is_number_unsigned()||rows.is_number_integer(),"--rows must be an integer");
  auto fit=fit_only?rows:number(o,"fit-rows",0);need(fit_only||o.has("fit-rows"),"--fit-rows is required for evaluation/search");
  need(rows.get<std::int64_t>()>0&&fit.get<std::int64_t>()>0&&fit.get<std::uint64_t>()<=rows.get<std::uint64_t>(),"invalid FIT/total row counts");
  return {{"format","dense-fp32-u32-class-labels-1"},{"features",features},{"classes",classes},{"rows",rows},
    {"row_stride",o.has("row-stride")?scalar(o.get("row-stride")):features},{"FIT_rows",fit},{"VALID_rows",rows.get<std::uint64_t>()-fit.get<std::uint64_t>()},
    {"values_path",values.string()},{"values_sha256",pin(values)},{"labels_path",labels.string()},{"labels_sha256",pin(labels)},
    {"preprocessing",o.get("preprocessing","caller-supplied model-input FP32; no CLI preprocessing")},{"TEST_read",false}};
}
J hyperparameters(const Options&o){
  J hp={{"rounds",number(o,"rounds",1)},{"max_depth",number(o,"depth",2)}};
  for(const char* key:{"eta","lambda","alpha","gamma","min-child-weight","max-bin","seed","subsample","colsample-bytree","colsample-bylevel","colsample-bynode"})if(o.has(key)){std::string name=key;std::replace(name.begin(),name.end(),'-','_');hp[name]=scalar(o.get(key));}
  return hp;
}
void bind_library(J& plan,const Options&o){
  if(o.has("library")){auto path=absolute(o.get("library"));plan["native_library_path"]=path.string();plan["native_library_sha256"]=pin(path);}
  if(!plan.contains("native_library_path")){
    fs::path path;if(const char*env=std::getenv("XGBOOST_LIBRARY"))path=absolute(env);
    else {
      auto python=program("python3");if(python.empty())python=program("python");if(!python.empty()){
        std::string found;auto status=run({python.string(),"-c","import importlib.util,pathlib; s=importlib.util.find_spec('xgboost'); p=pathlib.Path(s.origin).parent/'lib'/'libxgboost.so' if s and s.origin else pathlib.Path('/nonexistent'); print(p if p.is_file() else '')"},&found);
        if(status==0){while(!found.empty()&&(found.back()=='\n'||found.back()=='\r'))found.pop_back();if(!found.empty())path=absolute(found);}
      }
    }
    need(!path.empty()&&fs::is_regular_file(path),"XGBoost library not found; install CUDA-enabled xgboost==3.4.1, set XGBOOST_LIBRARY, or pass --library PATH");plan["native_library_path"]=path.string();plan["native_library_sha256"]=pin(path);
  }
  need(plan.contains("native_library_sha256"),"native library SHA binding missing");
}
const std::set<std::string> data_options={"data","target","header","label-map","values","labels","features","classes","rows","row-stride","fit-rows","preprocessing"};
const std::set<std::string> hp_options={"rounds","depth","eta","lambda","alpha","gamma","min-child-weight","max-bin","seed","subsample","colsample-bytree","colsample-bylevel","colsample-bynode"};
std::set<std::string> allowed(std::initializer_list<const char*> keys,bool data=false,bool hp=false){std::set<std::string> out;for(auto key:keys)out.insert(key);if(data)out.insert(data_options.begin(),data_options.end());if(hp)out.insert(hp_options.begin(),hp_options.end());return out;}
void conversion_flags(J&options,const Options&o){
  for(const char* key:{"initial-states","max-states","initial-nodes","max-nodes","gpu-byte-budget","max-expansions","batch-size","max-batch-size","admission-threads","draft-threads","oldest-ready-jobs","completed-cache-limit","refinement-visit-budget","cover-visit-budget","checkpoint-interval-seconds","checkpoint-host-byte-budget"})if(o.has(key)){std::string name=key;std::replace(name.begin(),name.end(),'-','_');options[name]=scalar(o.get(key));}
  for(const char* key:{"split-policy","runtime-residency"})if(o.has(key)){std::string name=key;std::replace(name.begin(),name.end(),'-','_');options[name]=o.get(key);}
  for(auto [key,name]:std::vector<std::pair<const char*,const char*>>{{"checkpoint","checkpoint_path"},{"resume","resume_from"},{"proof-dir","proof_module_directory"},{"proof-request","proof_module_request"}})if(o.has(key))options[name]=absolute(o.get(key)).string();
}
std::set<std::string> conversion_options(){return allowed({"model","library","options","out","initial-states","max-states","initial-nodes","max-nodes","gpu-byte-budget","max-expansions","batch-size","max-batch-size","admission-threads","draft-threads","oldest-ready-jobs","completed-cache-limit","refinement-visit-budget","cover-visit-budget","checkpoint-interval-seconds","checkpoint-host-byte-budget","split-policy","runtime-residency","checkpoint","resume","proof-dir","proof-request"});}
void help(const std::string& command={}){
  if(command.empty()){
    std::cout<<R"(decision-programs — exact decision programs on CUDA

Usage: decision-programs COMMAND [OPTIONS]
       decision-programs COMMAND --help

  doctor       Check installed backends/tools; --gpu queries CUDA
  demo         Create and verify a small self-contained decision model
  train        FIT-only native XGBoost CUDA training
  convert      Convert a saved multiclass XGBoost model
  import-tree  Import a completed CLSTREE1 tree with its origin contract
  study        Fixed trials: train, convert, simplify, then evaluate
  hpo          Declared native trials selected on VALID
  combine      Out-of-fold nonlinear teacher composition (alias: nonlinear)
  rl           Qualified experimental equivalent-encoding policy search
  evaluate     Compare converted runtime classes against native source
  simplify     Exact adjacent-predicate shared-DAG simplification
  predict      Predict class IDs from a completed canonical/compact model
  inspect      Validate and inspect model metadata on CUDA
  explain      Exact decision paths; regional format also has fuzzy gradients
  export       Shared exact decision equations, or compact model bytes
  checkpoint   Request a live checkpoint or inspect a checkpoint directory
  resume       Continue checkpoints; RL resumes policy words into a fresh session
  proofs       List Lean sources or check and replay the maintained module suite
  profile      Run any public command under an installed profiling tool
  capabilities Print supported formats and evidence boundaries

Paths may be relative. Inputs are hashed automatically. Existing outputs are
preserved; use a fresh --out. --dry-run prints the actual backend arguments and
generated plans without CUDA execution. --save-plan FILE retains a generated
plan. --set /json/pointer=JSON_VALUE exposes every supported backend setting.
Version: 0.1.0. Numerical computation: C++23/CUDA.
)";return;
  }
  std::cout<<"Usage: decision-programs "<<command<<' ';
  if(command=="doctor")std::cout<<"[--gpu]\nHost inventory is always available. --gpu calls the CUDA device query backend.\n";
  else if(command=="demo")std::cout<<"[--output NEW_DIRECTORY]\nCreates a fresh temporary output by default. Builds a five-node, three-class model; verifies predictions and exact paths on CUDA.\n";
  else if(command=="convert")std::cout<<R"(--model MODEL.json --library libxgboost.so --out NEW_DIRECTORY
  [--options OPTIONS.json] [--max-nodes N] [--max-states N]
  [--gpu-byte-budget BYTES] [--max-expansions N] [--batch-size N]
  [--split-policy source_order|widest_residual|aggregate_residual|contracting_residual]
  [--runtime-residency dual|canonical_only|compact_only]
  [--checkpoint DIRECTORY] [--resume DIRECTORY] [--proof-dir DIRECTORY]
  [--proof-request FILE] [--set /OPTION=JSON_VALUE]
Requires CUDA-enabled XGBoost 3.4.1. The backend preserves class IDs over its
declared finite-FP32/NaN domain; incomplete capacity-bounded runs stay incomplete.
)";
  else if(command=="import-tree")std::cout<<R"(--model MODEL.clstree --origin ORIGIN.json --output NEW_DIRECTORY
  [--max-nodes N]
Structural CLSTREE1 import preserves active predicate words/classes/NaN routing.
The origin contract must bind the model SHA and runtime_domain. The adapter does
not replay source-equivalence evidence; CUDA qualification uses the shared runtime.
)";
  else if(command=="train")std::cout<<R"(--data TRAIN.csv --target COLUMN --output NEW_DIRECTORY
  [--rounds N] [--depth N] [--library libxgboost.so]
Or:
  (--data FIT_DESCRIPTOR.json | --values X.fp32 --labels Y.u32
    --rows N --features F --classes K [--row-stride S])
  [--rounds N] [--depth N] [--eta NUMBER] [--seed N] [--check]
  [--plan EXISTING_PLAN.json] [--set /hyperparameters/OPTION=JSON_VALUE]
CSV headers are detected automatically; --header yes|no overrides detection.
Targets may be integer IDs or strings (the mapping is retained in the output).
Default: one round, depth two. FIT-only data is required. --check checks the plan
and bindings without CUDA numerical execution. Sampling/regularization options:
--lambda --alpha --gamma --min-child-weight --max-bin --subsample
--colsample-bytree --colsample-bylevel --colsample-bynode.
)";
  else if(command=="study"||command=="hpo"||command=="combine"){
    std::cout<<R"(--library libxgboost.so --out NEW_DIRECTORY
  (--data EVAL_DESCRIPTOR.json | --values X.fp32 --labels Y.u32
    --rows N --features F --classes K --fit-rows N [--row-stride S])
  [--plan PLAN.json] [--rounds N] [--depth N] [--trial JSON_OBJECT]
  [--checkpoint DIRECTORY] [--resume DIRECTORY] [--set /SETTING=JSON_VALUE]
)";
    if(command=="study")std::cout<<"Fixed study additionally needs --train-data FIT_DESCRIPTOR.json; --set /conversion/OPTION=VALUE controls conversion.\n";
    if(command=="hpo")std::cout<<"Each --trial overrides base hyperparameters. VALID errors, then native bytes, then declared order select the winner. No TEST access.\n";
    if(command=="combine")std::cout<<R"(Without --plan, also provide --teachers JSON_ARRAY --meta JSON_ARRAY
--baseline MODEL.json [--folds N] [--oof-gpu-byte-budget BYTES].
Teacher/meta entries are hyperparameter objects. --stop-after-oof requires a
checkpoint. OOF composition and VALID selection do not imply improved TEST accuracy.
)";
  }
  else if(command=="evaluate")std::cout<<R"(--plan PLAN.json --out NEW_DIRECTORY
Or: --model CANONICAL --source MODEL.json --library libxgboost.so --data EVAL_DESCRIPTOR.json --out NEW_DIRECTORY [--compact COMPACT]
Pins and current executable binding are generated automatically. Compares class
IDs and computes FIT/VALID scores; performs no fitting or model selection.
)";
  else if(command=="simplify")std::cout<<R"(--model CANONICAL --source MODEL.json --out NEW_DIRECTORY
  [--max-passes N] [--max-nodes N] [--plan PLAN.json]
Exact immutable adjacent-predicate rewrites; a pass limit can stop before a fixed
point. The output retains the source binding. Native comparison uses evaluate.
)";
  else if(command=="predict"||command=="explain")std::cout<<R"(--model MODEL_DIRECTORY_OR_CANONICAL [--compact COMPACT]
  (--data INPUTS.csv | --input ROWS.json | --values X.fp32 --rows N [--row-stride S])
  [--out NEW_JSON_FILE] [--layout canonical|compact] [--block-size 64|128|256]
ROWS.json is {"rows":[[feature,...],...]}; null means NaN. Predictions are class
IDs. explain returns visited shared nodes including the terminal; use
--max-path-nodes N to bound per-row path storage (default 4096).
Specialized regional explain: --format regional --model CLSRMDL1 --input INPUT.json
--out NEW_DIRECTORY. Requires the original 54-feature/10-temperature contract;
fuzzy membership scores and gradients are a distinct, uncalibrated model.
)";
  else if(command=="inspect")std::cout<<"--model CANONICAL [--compact COMPACT] [--out NEW_JSON_FILE]\nWhole model is validated on CUDA. Hashes/domain/structure/device bytes are reported.\n";
  else if(command=="export")std::cout<<R"(--model CANONICAL --out NEW_FILE [--kind equations|compact]
Shared equations preserve raw FP32 threshold words and NaN routing. They do not
prove new source equivalence or human clarity. --kind compact packs on CUDA.
Specialized regional export: --format regional --model CLSRMDL1 --out NEW_DIRECTORY
[--smooth-numeric 0|1]. Preserve the original regional model contract.
)";
  else if(command=="checkpoint")std::cout<<"(--pid PID | --directory CHECKPOINT_DIRECTORY)\n--pid sends SIGUSR1 to a configured live public command/backend. --directory lists persisted metadata without claiming it is resumable.\n";
  else if(command=="resume")std::cout<<"KIND <KIND_OPTIONS> --resume CHECKPOINT_DIRECTORY\nKIND is convert, study, hpo, combine or rl. Keep semantic declarations unchanged; use a fresh --out.\n";
  else if(command=="proofs")std::cout<<"[--directory PROOF_ROOT] [--check [--output REPORT_DIRECTORY] [--lean LEAN_EXECUTABLE]]\nLists formal sources. --check compiles the maintained module suite in dependency order and replays it with leanchecker, retaining logs/hashes/receipt in a fresh child directory. Requires Lean 4.34.1, leanchecker and CMake. This is no CUDA implementation-refinement claim.\n";
  else if(command=="profile")std::cout<<R"(--tool nsys|ncu|memcheck|initcheck|racecheck|synccheck|cuda-gdb
  [--out REPORT_PREFIX] [--tool-arg ARG] -- COMMAND COMMAND_OPTIONS
An absolute executable path is also accepted as --tool; pass its arguments with
--tool-arg. Exactly one tool wraps the command. Tool availability is not collection
success; backend/tool exit status and retained reports establish completion.
)";
  else if(command=="rl")std::cout<<R"(--model SOURCE.json --output NEW_DIRECTORY [--library libxgboost.so]
  [--episodes N] [--seed N] [--episode N] [--learning-rate NUMBER]
  [--max-native-cells N] [--native-batch-rows N] [--chunk-records N]
  [--warmup-episodes N] [--policy-state POLICY.json | --resume POLICY.json]
  [--diagnostic-trajectories]
Learning rate must be greater than zero and at most one.
Specialized experimental regional policy: 10 numeric features plus wilderness4
and soil40 one-hot groups, seven classes. Orders equivalent encodings; policy
actions do not establish class authority or improved accuracy. --resume warm
starts policy words only; it creates a fresh session, tables, schedule and warmup.
)";
  else if(command=="capabilities")std::cout<<"\nPrint machine-readable per-format support and limitations.\n";
  else throw std::invalid_argument("unknown command: "+command);
  std::cout<<"Common: --output aliases --out; --dry-run --save-plan FILE --set /json/pointer=JSON_VALUE\nXGBoost library: --library PATH, XGBOOST_LIBRARY, or active Python xgboost package.\n";
}
J capabilities(){return {{"format","decision-programs-capabilities-1"},{"generic_runtime_format","CLSGDAG1 / CLSG64B1 with canonical companion"},
 {"supported",{"native_CUDA_training","saved_native_model_conversion","fixed_trials","declared_native_HPO","OOF_nonlinear_composition","adjacent_predicate_simplification","class_prediction","exact_paths","shared_hard_equations","compact_export","conversion_and_experiment_checkpoint_resume","Lean_source_checking","external_profiling"}},
 {"regional_format","CLSRMDL1: specialized Forest rank/category export and fuzzy CUDA explanations; qualified experimental RL ordering"},
 {"generic_pending",{"fuzzy_gradients","counterfactuals","global_feature_importance","causal_effects","adaptive_HPO_policy","in_flight_native_training_resume"}},
 {"conversion","resource-bounded; incomplete runs are not completed models; no global minimum-size or arbitrary-scale guarantee"},
 {"inference","classes only; no native probabilities or calibrated uncertainty"},{"numerical_execution","C++23/CUDA"},{"host_role","argument/metadata/hash/byte transport"}};}
int dispatch(const std::string&cmd,const Options&o){
  if(o.has("help")){help(cmd);return 0;}
  Inputs input;
  if(cmd=="capabilities"){o.allow({});std::cout<<capabilities().dump(2)<<'\n';return 0;}
  if(cmd=="doctor"){
    o.allow({"gpu"});J backends=J::object(),tools=J::object();
    for(const char*name:{"class_model_train","class_model_convert","class_study","class_model_evaluate","class_model_simplify","decision_programs_model_tools","class_rank_xai_export","class_rank_xai_explain","rl_session","class_apply_rl"}){auto p=backend(name,false);backends[name]=p.empty()?J(nullptr):J(p.string());}
    for(const char*name:{"nsys","ncu","compute-sanitizer","cuda-gdb","lean","nvcc","nvidia-smi"}){auto p=program(name);tools[name]=p.empty()?J(nullptr):J(p.string());}
    J result={{"format","decision-programs-doctor-1"},{"executable",executable().string()},{"backends",backends},{"tools",tools},{"CUDA_numerical_execution",false},{"tool_collection_verified",false}};
    std::cout<<result.dump(2)<<'\n';if(o.has("gpu"))return launch({backend("decision_programs_model_tools").string(),"doctor"},o,input);return 0;
  }
  if(cmd=="demo"){
    o.allow({"out"});fs::path out;if(o.has("out"))out=absolute(o.get("out"));else {out=fs::temp_directory_path()/("decision-programs-demo-"+std::to_string(getpid()));std::uint64_t suffix=0;while(fs::exists(out))out=fs::temp_directory_path()/("decision-programs-demo-"+std::to_string(getpid())+"-"+std::to_string(++suffix));}
    std::cerr<<"Demo output: "<<out<<'\n';return launch({backend("decision_programs_model_tools").string(),"demo",out.string()},o,input);
  }
  if(cmd=="convert"){
    o.allow(conversion_options());J settings=o.has("options")?file_json(o.get("options")):J::object();bind_library(settings,o);conversion_flags(settings,o);overrides(settings,o);
    auto mapping=absolute(o.required("model")).parent_path()/"input-transport.json";if(fs::is_regular_file(mapping))input.generated["input-transport.json"]=file_json(mapping.string());
    auto options=input.json("plan.json",settings);return launch({backend("class_model_convert").string(),absolute(o.required("model")).string(),options.string(),absolute(o.required("out")).string()},o,input);
  }
  if(cmd=="import-tree"){
    o.allow({"model","origin","out","max-nodes"});auto model=absolute(o.required("model")),origin=absolute(o.required("origin"));return launch({backend("class_tree_adapter").string(),model.string(),pin(model),origin.string(),pin(origin),absolute(o.required("out")).string(),number(o,"max-nodes",16777216).dump()},o,input);
  }
  if(cmd=="train"){
    o.allow(allowed({"library","out","plan","check"},true,true));auto target=backend("class_model_train");J plan;
    if(o.has("plan"))plan=file_json(o.get("plan"));else plan={{"format","class-model-training-plan-1"},{"dataset",dataset(o,true,input)},{"hyperparameters",hyperparameters(o)},{"TEST_read",false},{"VALID_read",false},{"selection_performed",false},{"training_performed",true}};
    bind_library(plan,o);plan["executable_sha256"]=pin(target);
    std::string manifest;need(run({target.string(),"source-manifest"},&manifest)==0,"training source manifest failed");plan["source_manifest_sha256"]=parse(manifest).at("sha256");overrides(plan,o);
    auto path=input.json("plan.json",plan);Args command={target.string(),o.has("check")?"--check-plan":"fit",path.string(),pin(path)};if(!o.has("check"))command.push_back(absolute(o.required("out")).string());return launch(command,o,input);
  }
  if(cmd=="study"||cmd=="hpo"||cmd=="combine"){
    o.allow(allowed({"library","out","plan","train-data","trial","checkpoint","resume","stop-after-oof","teachers","meta","baseline","folds","oof-gpu-byte-budget"},true,true));
    auto data=dataset(o,false,input);J plan=o.has("plan")?file_json(o.get("plan")):J{{"TEST_read",false},{"VALID_read",false},{"hyperparameters",hyperparameters(o)}};
    bind_library(plan,o);
    if(!o.has("plan")){
      if(cmd=="study"){plan["dataset"]=file_json(o.required("train-data"));plan["conversion"]=J::object();}
      else {plan["workflow"]=cmd=="hpo"?"native-accuracy-search-1":"native-nonlinear-combination-1";plan["dataset_source"]=data;}
      if(cmd=="combine"){
        plan["teacher_hyperparameters"]=parse(o.required("teachers"));plan["meta_hyperparameters"]=parse(o.required("meta"));plan["required_baseline_sha256"]=pin(absolute(o.required("baseline")));
        plan["oof"]={{"folds",number(o,"folds",3)},{"seed",number(o,"seed",0)},{"gpu_byte_budget",number(o,"oof-gpu-byte-budget",1073741824)}};
      }
    }
    if(cmd=="hpo")need(plan.value("workflow",std::string{})=="native-accuracy-search-1","hpo requires native-accuracy-search-1 workflow");
    if(cmd=="combine")need(plan.value("workflow",std::string{})=="native-nonlinear-combination-1","combine requires native-nonlinear-combination-1 workflow");
    if(o.has("trial")){plan["trials"]=J::array();for(const auto&t:o.values.at("trial"))plan["trials"].push_back(parse(t));}
    if(o.has("checkpoint"))plan["experiment_checkpoint_path"]=absolute(o.get("checkpoint")).string();overrides(plan,o);
    auto path=input.json("plan.json",plan),descriptor=input.json("data.json",data);Args command={backend("class_study").string(),path.string(),descriptor.string(),absolute(o.required("out")).string()};
    if(o.has("resume")){command.push_back("--resume");command.push_back(absolute(o.get("resume")).string());}if(o.has("stop-after-oof"))command.push_back("--stop-after-oof");return launch(command,o,input);
  }
  if(cmd=="evaluate"||cmd=="simplify"){
    o.allow(allowed({"plan","out","model","source","library","compact","max-passes","max-nodes"},cmd=="evaluate"));auto target=backend(cmd=="evaluate"?"class_model_evaluate":"class_model_simplify");J plan;
    if(o.has("plan"))plan=file_json(o.get("plan"));
    else {
      auto model=model_file(o.required("model")),source=absolute(o.required("source"));auto bytes=dpnative::read_text(model);need(bytes.size()>=64&&bytes.substr(0,8)=="CLSGDAG1","--model must be canonical CLSGDAG1");
      if(cmd=="evaluate"){
        auto evaluation_options=o;
        if(o.has("data")&&absolute(o.get("data")).extension()==".csv"&&!o.has("label-map")){
          auto mapping=source.parent_path()/"input-transport.json";if(!fs::is_regular_file(mapping))mapping=model.parent_path()/"input-transport.json";
          if(fs::is_regular_file(mapping))evaluation_options.values["label-map"]={mapping.string()};
          else{auto dense=decision_programs_csv::decode(dpnative::read_text(absolute(o.get("data"))),o.required("target"),o.get("header","auto"));need(dense.label_mapping.empty(),"string-label evaluation needs trained --label-map INPUT_TRANSPORT.json; no mapping was found beside source/model");}
        }
        auto evaluation_data=dataset(evaluation_options,false,input);
        if(o.has("data")&&absolute(o.get("data")).extension()==".csv"){
          auto source_document=file_json(source.string());auto count=source_document.at("learner").at("learner_model_param").at("num_class").get<std::string>();auto classes=parse(count);need(classes.is_number_integer()&&classes.get<std::uint64_t>()>=evaluation_data.at("classes").get<std::uint64_t>(),"source classes do not cover evaluation CSV labels");evaluation_data["classes"]=classes;
        }
        plan={{"format","class-model-evaluation-plan-1"},{"dataset",evaluation_data},{"TEST_read",false},{"selection_performed",false},{"training_performed",false}};
        J row={{"source_path",source.string()},{"source_sha256",pin(source)},{"runtime_path",model.string()},{"runtime_sha256",pin(model)}};
        if(o.has("compact")){auto compact=absolute(o.get("compact"));row["canonical_runtime_path"]=model.string();row["canonical_runtime_sha256"]=pin(model);row["runtime_path"]=compact.string();row["runtime_sha256"]=pin(compact);}plan["evaluations"]=J::array({row});
      }else {
        auto word=[&](std::size_t at){std::uint32_t v;std::memcpy(&v,bytes.data()+at,4);return v;};
        plan={{"format","adjacent-predicate-simplification-plan-1"},{"TEST_read",false},{"training_performed",false},{"selection_performed",false},{"features",word(12)},{"classes",word(16)},{"max_passes",number(o,"max-passes",4)},{"max_nodes",number(o,"max-nodes",word(24))},{"canonical_runtime_path",model.string()},{"canonical_runtime_sha256",pin(model)},{"source_path",source.string()},{"source_sha256",pin(source)},{"case_id","public-cli"},{"Lean_prefix_law_receipt",nullptr}};
      }
    }
    if(cmd=="evaluate")bind_library(plan,o);plan["executable_sha256"]=pin(target);overrides(plan,o);auto path=input.json("plan.json",plan);
    return launch({target.string(),cmd,path.string(),pin(path),absolute(o.required("out")).string()},o,input);
  }
  if(cmd=="predict"||cmd=="inspect"||cmd=="explain"||cmd=="export"){
    o.allow({"model","compact","source-sha","model-sha","compact-sha","input","data","header","values","rows","row-stride","out","layout","block-size","max-path-nodes","format","smooth-numeric","kind"});
    auto model=model_file(o.required("model"));
    if(o.get("format")=="regional"){
      need(cmd=="explain"||cmd=="export","regional format is available through explain/export only");Args command={backend(cmd=="explain"?"class_rank_xai_explain":"class_rank_xai_export").string(),model.string(),o.get("model-sha",pin(model))};
      if(cmd=="explain")command.push_back(absolute(o.required("input")).string());command.push_back(absolute(o.required("out")).string());
      if(cmd=="export"&&o.has("smooth-numeric")){command.push_back("--smooth-numeric");command.push_back(o.get("smooth-numeric"));}return launch(command,o,input);
    }
    need(o.get("format","generic")=="generic","unknown model format; use generic or regional");
    J plan={{"canonical_path",model.string()},{"canonical_sha256",o.get("model-sha",pin(model))}};
    if(o.has("source-sha"))plan["source_sha256"]=o.get("source-sha");
    if(o.has("compact")){auto compact=absolute(o.get("compact"));plan["compact_path"]=compact.string();plan["compact_sha256"]=o.get("compact-sha",pin(compact));}
    else if(fs::is_directory(absolute(o.get("model")))){
      for(const auto&candidate:std::vector<fs::path>{model.parent_path()/"model.compact",model.parent_path()/"model.clsg64",fs::path(model).replace_extension(".compact")})if(fs::is_regular_file(candidate)){plan["compact_path"]=candidate.string();plan["compact_sha256"]=pin(candidate);break;}
    }
    if(o.has("data")){
      auto path=absolute(o.get("data"));if(path.extension()==".json")plan["input"]=path.string();else{
        auto original=dpnative::read_text(path);auto dense=decision_programs_csv::decode(original,{},o.get("header","auto"));auto canonical=dpnative::read_text(model);need(canonical.size()>=64,"canonical header truncated");std::uint32_t features;std::memcpy(&features,canonical.data()+12,4);need(dense.features==features,"CSV feature column count differs from model; omit the target column");auto cached=retain_csv(original,dense,original+"\nfeatures\n"+o.get("header","auto"));plan["values"]=cached.values.string();plan["rows"]=dense.rows;plan["row_stride"]=dense.features;plan["input_transport"]="host CSV-to-FP32 input-format conversion; source_sha256="+dpnative::sha256(original);input.json("input-transport.json",{{"source_path",path.string()},{"source_sha256",dpnative::sha256(original)},{"feature_names",dense.names},{"header",dense.header},{"CPU_model_numerics",false}});
      }
    }
    for(const char*key:{"input","values","out"})if(o.has(key))plan[key]=absolute(o.get(key)).string();
    for(const char*key:{"rows","row-stride","block-size","max-path-nodes"})if(o.has(key)){std::string name=key;std::replace(name.begin(),name.end(),'-','_');plan[name]=scalar(o.get(key));}
    plan["layout"]=o.get("layout",plan.contains("compact_path")?"compact":"canonical");plan["kind"]=o.get("kind","equations");overrides(plan,o);auto path=input.json("plan.json",plan);
    return launch({backend("decision_programs_model_tools").string(),cmd,path.string()},o,input);
  }
  if(cmd=="checkpoint"){
    o.allow({"pid","directory"});need(o.has("pid")!=o.has("directory"),"choose --pid or --directory");
    if(o.has("pid")){auto value=scalar(o.get("pid"));need(value.is_number_integer()&&value.get<std::int64_t>()>1&&value.get<std::int64_t>()<=INT32_MAX,"PID must be an integer greater than one");auto pid=pid_t(value.get<std::int64_t>());if(!o.has("dry-run"))need(kill(pid,SIGUSR1)==0,std::string("checkpoint signal failed: ")+std::strerror(errno));std::cout<<J{{"signal","SIGUSR1"},{"pid",pid},{"request_sent",!o.has("dry-run")},{"checkpoint_commit_verified",false}}.dump(2)<<'\n';return 0;}
    auto dir=absolute(o.get("directory"));need(fs::is_directory(dir),"checkpoint directory missing");J files=J::array();for(const auto&e:fs::recursive_directory_iterator(dir))if(e.is_regular_file()&&(e.path().extension()=="json"||e.path().extension()==".json"))files.push_back({{"path",fs::relative(e.path(),dir).string()},{"bytes",e.file_size()},{"sha256",pin(e.path())}});
    std::cout<<J{{"directory",dir.string()},{"metadata_files",files},{"resume_validation_performed",false}}.dump(2)<<'\n';return 0;
  }
  if(cmd=="proofs"){
    o.allow({"directory","check","list","out","lean"});fs::path root=o.has("directory")?absolute(o.get("directory")):(executable().parent_path()/"../share/decision-programs/proofs").lexically_normal();
    if(!fs::is_directory(root)&&fs::is_directory(fs::current_path()/"formal"))root=fs::current_path()/"formal";
    need(fs::is_directory(root),"proof source directory missing; supply --directory formal");
    if(o.has("check")){
      auto cmake=program("cmake");need(!cmake.empty(),"CMake missing for proof dependency/replay checks");auto script=(executable().parent_path()/"../share/decision-programs/cmake/CheckLean.cmake").lexically_normal();if(!fs::is_regular_file(script))script=root.parent_path()/"cmake/CheckLean.cmake";need(fs::is_regular_file(script),"portable proof checker missing beside installation/source");
      auto output=o.has("out")?absolute(o.get("out")):fs::temp_directory_path()/"decision-programs-proof-checks";Args command={cmake.string(),"-DGH_PROOF_SOURCE_ROOT="+root.string(),"-DGH_PROOF_OUTPUT_ROOT="+output.string()};if(o.has("lean"))command.push_back("-DGH_LEAN_EXECUTABLE="+absolute(o.get("lean")).string());command.insert(command.end(),{"-P",script.string()});return launch(command,o,input);
    }
    J files=J::array();for(const auto&e:fs::recursive_directory_iterator(root))if(e.is_regular_file()&&e.path().extension()==".lean")files.push_back(fs::relative(e.path(),root).string());std::cout<<J{{"directory",root.string()},{"Lean_sources",files},{"freshly_checked",false},{"CUDA_implementation_refinement_claimed",false}}.dump(2)<<'\n';return 0;
  }
  if(cmd=="profile"){
    o.allow({"tool","out","tool-arg"});need(!o.tail.empty(),"profile requires -- COMMAND OPTIONS");auto tool=o.required("tool");std::string program_name=tool;
    if(tool=="memcheck"||tool=="initcheck"||tool=="racecheck"||tool=="synccheck")program_name="compute-sanitizer";
    auto path=program(program_name);need(!path.empty(),"profiling tool missing: "+program_name);Args command={path.string()};
    if(program_name=="compute-sanitizer")command.insert(command.end(),{"--tool",tool,"--error-exitcode","99"});
    else if(tool=="nsys")command.insert(command.end(),{"profile","--force-overwrite=false","--output",absolute(o.required("out")).string()});
    else if(tool=="ncu")command.insert(command.end(),{"--export",absolute(o.required("out")).string()});
    else if(tool=="cuda-gdb")command.push_back("--args");
    if(o.has("tool-arg"))command.insert(command.end(),o.values.at("tool-arg").begin(),o.values.at("tool-arg").end());command.push_back(executable().string());command.insert(command.end(),o.tail.begin(),o.tail.end());return launch(command,o,input);
  }
  if(cmd=="rl"){
    o.allow({"model","library","out","episodes","seed","episode","learning-rate","max-native-cells","native-batch-rows","chunk-records","warmup-episodes","policy-state","resume","diagnostic-trajectories"});need(!fs::exists(absolute(o.required("out"))),"RL output must be a fresh directory");J plan=J::object();bind_library(plan,o);auto model=absolute(o.required("model"));double rate=o.has("learning-rate")?scalar(o.get("learning-rate")).get<double>():0.01;need(std::isfinite(rate)&&rate>0&&rate<=1,"learning rate must be finite and in (0,1]");
    Args command={backend("rl_session").string(),model.string(),pin(model),plan.at("native_library_path"),absolute(o.required("out")).string(),"-",number(o,"episodes",32).dump(),number(o,"seed",4100).dump(),number(o,"episode",900).dump(),std::to_string(std::bit_cast<std::uint64_t>(rate)),number(o,"max-native-cells",536870912).dump(),number(o,"native-batch-rows",16384).dump(),number(o,"chunk-records",32).dump(),number(o,"warmup-episodes",32).dump()};
    need(!o.has("policy-state")||!o.has("resume"),"choose --policy-state or --resume");if(o.has("policy-state")||o.has("resume"))command.push_back(absolute(o.get("policy-state",o.get("resume"))).string());if(o.has("diagnostic-trajectories"))command.push_back("--diagnostic-trajectories");input.generated["RL_semantics"]={{"specialized_regional",true},{"resume_scope","policy-word warm start only; fresh session/table/schedule/warmup"}};return launch(command,o,input);
  }
  throw std::invalid_argument("unknown command: "+cmd+"; run decision-programs --help");
}
} // namespace
int main(int argc,char**argv){try{
  if(argc==1||(argc==2&&(std::string(argv[1])=="--help"||std::string(argv[1])=="-h"))){help();return 0;}
  if(argc==2&&std::string(argv[1])=="--version"){std::cout<<"decision-programs 0.1.0\n";return 0;}
  std::string command=argv[1];int begin=2;if(command=="help"){help(argc>2?argv[2]:"");return 0;}
  if(command=="resume"){need(argc>2,"resume requires a kind; use resume --help");if(std::string(argv[2])=="--help"){help("resume");return 0;}command=argv[2];need(command=="convert"||command=="study"||command=="hpo"||command=="combine"||command=="rl","unsupported resume kind");begin=3;}
  if(command=="nonlinear")command="combine";Args args;for(int i=begin;i<argc;++i)args.emplace_back(argv[i]);Options options(args);if(std::string(argv[1])=="resume"&&!options.has("help"))need(options.has("resume"),"resume requires --resume CHECKPOINT_DIRECTORY");return dispatch(command,options);
}catch(const std::exception&e){std::cerr<<"decision-programs: "<<e.what()<<'\n';return 2;}}
