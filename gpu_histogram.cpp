// GH_SOURCE_CATEGORY: tooling
#include "gpu_histogram.hpp"
// Runtime contract: the CPU selects an executable entry, transports opaque
// external bytes, owns CUDA resources and observes runtime completion. All
// application computation stays in the selected GPU entry. Child launches drain
// before resident storage is released; an earlier CUDA error wins over cleanup.
// Existing driver/quality source is the behavioral reference. Consolidation
// preserves their bootstrap and completion topology; validate every GPU suite,
// quality gate and uninstrumented complete-operation benchmark after integration.
#include <cstdio>

#ifndef GH_MODE
#define GH_MODE 0
#endif

#if GH_MODE == 0
#include <exception>
#include <string>
#include <vector>

// Development infrastructure only: argv/environment/log bytes enter POSIX child
// processes; receipts preserve every attempt and artifact identity. No application
// inputs, statistics, fixtures or GPU decisions are computed by these helpers.
// A child owns a new process group; timeout/interrupt sends TERM, waits three
// seconds, then KILL, and Nsight sessions receive their explicit shutdown command.
// Diagnostic activity parsers retain the Python collector's positive evidence
// gates. Profiler timing never ranks an operation. Fair validation: the same raw
// exports/logs and harmless process/timeout fixtures must produce equal receipts
// except elapsed time, collector identity and explicitly variable session names.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <openssl/evp.h>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
extern char **environ;
namespace dev {
using namespace std;
namespace fs = std::filesystem;
using json = nlohmann::json;
using Env = map<string, string>;
string read(const fs::path& path) {
  ifstream in(path, ios::binary);
  if (!in) throw runtime_error("cannot read: " + path.string());
  string result((istreambuf_iterator<char>(in)), {});
  if (in.bad()) throw runtime_error("read failed: " + path.string());
  return result;
}
void write(const fs::path& path, const string& bytes) {
  ofstream out(path, ios::binary);
  if (!out || !out.write(bytes.data(), bytes.size())) throw runtime_error("cannot write: " + path.string());
}
struct Hash {
  unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context{EVP_MD_CTX_new(), EVP_MD_CTX_free};
  Hash() { if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) throw runtime_error("SHA256 initialization failed"); }
  void add(string_view bytes) { if (EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) throw runtime_error("SHA256 update failed"); }
  string finish() {
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned length = 0;
    if (EVP_DigestFinal_ex(context.get(), bytes, &length) != 1) throw runtime_error("SHA256 finalization failed");
    string out; constexpr char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < length; ++i) { out += hex[bytes[i] >> 4]; out += hex[bytes[i] & 15]; }
    return out;
  }
};
string digest(string_view bytes) { Hash hash; hash.add(bytes); return hash.finish(); }
json identity(const fs::path& input) {
  const auto path = fs::canonical(input); ifstream in(path, ios::binary);
  if (!in) throw runtime_error("cannot read: " + path.string());
  Hash hash; array<char, 65536> buffer;
  while (in) { in.read(buffer.data(), buffer.size()); hash.add({buffer.data(), size_t(in.gcount())}); }
  if (in.bad()) throw runtime_error("read failed: " + path.string());
  return {{"path", path.string()}, {"bytes", fs::file_size(path)}, {"sha256", hash.finish()}};
}
Env environment() {
  Env env;
  for (auto p = environ; *p; ++p) { string value = *p; auto equal = value.find('='); if (equal != string::npos) env[value.substr(0, equal)] = value.substr(equal + 1); }
  return env;
}
string executable(const string& name) {
  if (name.find('/') != string::npos) {
    if (access(name.c_str(), X_OK) == 0 && fs::is_regular_file(name)) return fs::canonical(name).string();
  } else {
    istringstream paths(getenv("PATH") ? getenv("PATH") : ""); string part;
    while (getline(paths, part, ':')) { fs::path path = fs::path(part.empty() ? "." : part) / name;
      if (access(path.c_str(), X_OK) == 0 && fs::is_regular_file(path)) return fs::canonical(path).string(); }
  }
  throw runtime_error("missing executable: " + name);
}
namespace subprocess {
volatile sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }
struct SignalScope {
  struct sigaction before{}, action{};
  SignalScope() { interrupted = 0; action.sa_handler = interrupt; sigemptyset(&action.sa_mask); if (sigaction(SIGINT, &action, &before)) throw runtime_error("sigaction failed"); }
  ~SignalScope() { sigaction(SIGINT, &before, nullptr); }
};
struct Child { int exit; bool timeout; };
Child process(const vector<string>& command, const Env& env, const fs::path& output, const string& label, double timeout, bool append = false) {
  if (command.empty()) throw runtime_error("empty command");
  const auto program = executable(command.front());
  vector<string> assignments; vector<char*> argv, envp;
  for (const auto& [k,v] : env) assignments.push_back(k + "=" + v);
  for (const auto& value : command) argv.push_back(const_cast<char*>(value.c_str()));
  argv.push_back(nullptr);
  for (auto& value : assignments) envp.push_back(value.data());
  envp.push_back(nullptr);
  int flags = O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC);
  int out = open((output / (label + ".stdout.log")).c_str(), flags, 0666);
  int err = open((output / (label + ".stderr.log")).c_str(), flags, 0666);
  if (out < 0 || err < 0) { if (out >= 0) close(out); if (err >= 0) close(err); throw runtime_error("cannot open process logs"); }
  int startup[2];
  if (pipe2(startup, O_CLOEXEC)) { close(out); close(err); throw runtime_error("startup pipe failed"); }
  const pid_t pid = fork();
  if (pid == 0) {
    close(startup[0]);
    auto failed = [&](int code) { (void)::write(startup[1], &code, sizeof(code)); _exit(126); };
    signal(SIGINT, SIG_DFL);
    if (setsid() < 0 || chdir(output.c_str()) || dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0) failed(errno);
    close(out); close(err); execve(program.c_str(), argv.data(), envp.data()); failed(errno);
  }
  close(out); close(err); close(startup[1]);
  if (pid < 0) { close(startup[0]); throw runtime_error("fork failed"); }
  int startup_error = 0; ssize_t bytes;
  do { bytes = ::read(startup[0], &startup_error, sizeof(startup_error)); } while (bytes < 0 && errno == EINTR);
  close(startup[0]);
  if (bytes != 0) { int status; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} throw runtime_error(string("process startup failed: ") + strerror(startup_error)); }
  auto wait_until = [&](double seconds, bool heed_interrupt) {
    const auto deadline = chrono::steady_clock::now() + chrono::duration<double>(seconds);
    int status;
    while (true) {
      pid_t got = waitpid(pid, &status, WNOHANG);
      if (got == pid) return pair{true, WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status)};
      if (got < 0 && errno != EINTR) throw runtime_error("waitpid failed");
      if ((heed_interrupt && interrupted) || chrono::steady_clock::now() >= deadline) return pair{false, 0};
      poll(nullptr, 0, 10);
    }
  };
  auto [done, status] = wait_until(timeout, true);
  if (done) return {status, false};
  if (kill(-pid, SIGTERM) && errno != ESRCH) throw runtime_error("process group TERM failed");
  tie(done, status) = wait_until(3, false);
  if (!done) {
    if (kill(-pid, SIGKILL) && errno != ESRCH) throw runtime_error("process group KILL failed");
    int raw; while (waitpid(pid, &raw, 0) < 0) if (errno != EINTR) throw runtime_error("waitpid failed");
    status = WIFEXITED(raw) ? WEXITSTATUS(raw) : -WTERMSIG(raw);
  }
  return {status, true};
}
}
json execute(const vector<string>& argv, const Env& env, const fs::path& output, const string& label, double timeout) {
  const auto started = chrono::steady_clock::now(); subprocess::SignalScope signals;
  const auto child = subprocess::process(argv, env, output, label, timeout); json cleanup = nullptr;
  if (child.timeout) for (const auto& arg : argv) if (arg.starts_with("--session-new=")) {
    vector<string> shutdown{argv[0], "shutdown", "--session=" + arg.substr(14), "--kill=sigkill"};
    subprocess::interrupted = 0; const auto result = subprocess::process(shutdown, env, output, label, 10, true);
    cleanup = {{"argv", shutdown}};
    if (result.timeout) cleanup["timeout"] = true; else cleanup["exit"] = result.exit;
    break;
  }
  return {{"argv", argv}, {"exit", child.exit}, {"timeout", child.timeout}, {"cleanup", cleanup},
          {"wall_seconds", chrono::duration<double>(chrono::steady_clock::now() - started).count()}};
}
namespace collector {
// CLI patterns use UTF-8 PCRE2 with Unicode properties: groups, alternation,
// classes, inline flags, lookbehind and Python-style named captures are supported.
// Internal fixed parsers retain their separately exercised ECMAScript patterns.
struct Pattern {
  unique_ptr<pcre2_code, decltype(&pcre2_code_free)> code{nullptr, pcre2_code_free};
  explicit Pattern(const string& pattern) {
    int error; PCRE2_SIZE offset;
    code.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr));
    if (!code) { array<PCRE2_UCHAR, 256> message{}; pcre2_get_error_message(error, message.data(), message.size()); throw runtime_error("pattern at byte " + to_string(offset) + ": " + reinterpret_cast<const char*>(message.data())); }
  }
  bool search(const string& text) const {
    unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(pcre2_match_data_create_from_pattern(code.get(), nullptr), pcre2_match_data_free);
    if (!match) throw runtime_error("regex match allocation failed");
    const int result = pcre2_match(code.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), 0, 0, match.get(), nullptr);
    if (result < 0 && result != PCRE2_ERROR_NOMATCH) throw runtime_error("regex match failed: " + to_string(result));
    return result >= 0;
  }
};
const vector<string> sanitizers{"memcheck", "initcheck", "racecheck", "synccheck"};
const vector<string> injectors{"LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH", "NVTX_INJECTION64_PATH"};
const vector<string> collectors{"nsys", "ncu", "memcheck", "initcheck", "racecheck", "synccheck", "cupti-trace", "cupti-range", "cupti-pc", "nvbit-count", "nvbit-memory", "nvbit-graph", "cuda-gdb", "cuobjdump", "nvdisasm", "compiler"};
bool has(const vector<string>& values, const string& value) { return find(values.begin(), values.end(), value) != values.end(); }
string join(const vector<string>& values, string_view separator) {
  string result; for (const auto& value : values) { if (!result.empty()) result += separator; result += value; } return result;
}
vector<fs::path> files(const fs::path& dir) {
  vector<fs::path> paths; for (const auto& entry : fs::directory_iterator(dir)) if (entry.is_regular_file()) paths.push_back(entry.path());
  sort(paths.begin(), paths.end()); return paths;
}
vector<unsigned long long> version(const fs::path& path) {
  vector<unsigned long long> result; const string name = path.parent_path().filename().string();
  const regex number(R"(\d+)"); for (sregex_iterator it(name.begin(), name.end(), number), end; it != end; ++it) result.push_back(stoull(it->str()));
  return result;
}
fs::path sdk(const string& kind) {
  string key = kind; transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return char(toupper(c)); }); key += "_ROOT";
  if (const char* path = getenv(key.c_str()); path && *path) return fs::canonical(path);
  const char* home = getenv("HOME"); if (!home) throw runtime_error("HOME is not set");
  fs::path base = fs::path(home) / ".local/opt/gpu-profiling"; vector<fs::path> choices;
  if (fs::is_directory(base)) for (const auto& parent : fs::directory_iterator(base)) if (parent.is_directory() && parent.path().filename().string().starts_with(kind + "-")) {
    if (kind == "nvbit") { auto path = parent.path() / "nvbit_release_x86_64"; if (fs::exists(path)) choices.push_back(path); }
    else for (const auto& item : fs::directory_iterator(parent)) { const auto name = item.path().filename().string(); if (name.starts_with("cuda_cupti-") && name.ends_with("-archive")) choices.push_back(item.path()); }
  }
  if (choices.empty()) throw runtime_error("missing SDK; set " + key);
  return fs::canonical(*max_element(choices.begin(), choices.end(), [](const auto& a, const auto& b) { return version(a) < version(b); }));
}
string nonce() {
  array<unsigned char, 16> bytes{};
  size_t used = 0; while (used < bytes.size()) { auto n = getrandom(bytes.data() + used, bytes.size() - used, 0); if (n < 0) { if (errno == EINTR) continue; throw runtime_error("getrandom failed"); } used += n; }
  string text; constexpr char hex[] = "0123456789abcdef"; for (auto byte : bytes) { text += hex[byte >> 4]; text += hex[byte & 15]; } return text;
}
struct Collect {
  string tool, workload{"cdp"}, kernel, activity{R"(GH_GPU_ACTIVITY|PASS\s|GPU .*checks completed)"}, api_errors{"explicit"};
  fs::path output; double timeout{120}; int limit{2}; vector<string> command;
};
struct Plan { vector<string> command; Env env; json assets; fs::path decoder; };
Plan plan(const Collect& args) {
  Env env = environment(); vector<string> inherited;
  for (const auto& key : injectors) if (env.contains(key) && !env[key].empty()) inherited.push_back(key);
  if (!inherited.empty()) throw runtime_error("competing inherited injection: " + join(inherited, ", "));
  const auto target = fs::canonical(args.command.front());
  if (!fs::is_regular_file(target)) throw runtime_error("target must be a file");
  Plan p{args.command, env, {{"target", identity(target)}, {"collector", identity("/proc/self/exe")}}, {}};
  p.command[0] = target.string(); const auto& tool = args.tool;
  auto prefix = [&](vector<string> arguments) { arguments.insert(arguments.end(), p.command.begin(), p.command.end()); p.command = move(arguments); };
  if (tool == "nsys") prefix({executable("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--session-new=gh-" + nonce(),
    "--cuda-graph-trace=" + string(args.workload == "graph" ? "graph" : "node"), "--output=" + (args.output / "profile").string()});
  else if (tool == "ncu") prefix({executable("ncu"), "--metrics", "sm__ctas_launched.sum" + string(args.workload == "graph" ? ",launch__graph_exec_cuda_id" : ""),
    "--launch-count", to_string(args.limit), "--target-processes", "all", "--export", (args.output / "profile").string(), "--graph-profiling", args.workload == "graph" ? "graph" : "node"});
  else if (has(sanitizers, tool)) prefix({executable("compute-sanitizer"), "--tool", tool, "--error-exitcode", "97", "--report-api-errors", args.api_errors});
  else if (tool.starts_with("nvbit-")) {
    if (args.workload == "graph" && tool != "nvbit-graph") throw runtime_error("ordinary NVBit count/memory samples do not support graph capture; use graph counter");
    string name = tool == "nvbit-count" ? "instr_count" : tool == "nvbit-memory" ? "mem_trace" : "instr_count_cuda_graph";
    const auto library = sdk("nvbit") / "tools" / name / (name + ".so"); p.assets["injector"] = identity(library);
    Env added{{"LD_PRELOAD", library.string()}, {"ACTIVE_FROM_START", "1"}, {"TOOL_VERBOSE", "0"}, {"INSTR_BEGIN", "0"}, {"INSTR_END", "4294967295"},
      {"MANGLED_NAMES", "0"}, {"START_GRID_NUM", "0"}, {"END_GRID_NUM", to_string(min(args.limit, 100))}, {"COUNT_WARP_LEVEL", "1"}};
    for (const auto& [key, value] : added) p.env[key] = value;
    p.assets["nvdisasm"] = identity(executable("nvdisasm"));
  } else if (tool.starts_with("cupti-")) {
    if (tool == "cupti-range" && args.workload == "graph") throw runtime_error("installed range-injection sample has no graph launch callbacks");
    const auto root = sdk("cupti");
    const auto library = root / "samples" / (tool == "cupti-trace" ? "cupti_trace_injection/libcupti_trace_injection.so" : tool == "cupti-range" ? "profiling_injection/libinjection.so" : "pc_sampling_continuous/libpc_sampling_continuous.so");
    p.assets["injector"] = identity(library);
    for (const string name : {"libcupti.so", "libpcsamplingutil.so", "libnvperf_host.so", "libnvperf_target.so"}) if (fs::exists(root / "lib" / name)) p.assets[name] = identity(root / "lib" / name);
    p.env["CUDA_INJECTION64_PATH"] = library.string(); p.env["LD_LIBRARY_PATH"] = (root / "lib").string() + ":" + p.env["LD_LIBRARY_PATH"];
    if (tool == "cupti-trace") p.env["NVTX_INJECTION64_PATH"] = library.string();
    else if (tool == "cupti-range") p.env["INJECTION_METRICS"] = "sm__ctas_launched.sum";
    else {
      p.env["INJECTION_PARAM"] = "--collection-mode 1 --sampling-period 12 --file-name pcsampling.dat --verbose";
      p.decoder = root / "samples/pc_sampling_utility/pc_sampling_utility"; p.assets["decoder"] = identity(p.decoder);
    }
  } else if (tool == "cuda-gdb") prefix({executable("cuda-gdb"), "--batch", "-ex", "set pagination off", "-ex", "set cuda break_on_launch application", "-ex", "run",
    "-ex", "info cuda kernels", "-ex", "set cuda break_on_launch none", "-ex", "continue", "--args"});
  else if (tool == "cuobjdump") p.command = {executable(tool), "--dump-resource-usage", "--dump-sass", target.string()};
  else if (tool == "nvdisasm") p.command = {executable(tool), target.string()};
  if (tool != "compiler" && !tool.starts_with("nvbit-") && !tool.starts_with("cupti-")) p.assets["tool"] = identity(p.command[0]);
  return p;
}
bool csv_row(istream& in, vector<string>& row) {
  row.clear(); string field; bool quoted = false, seen = false;
  for (int value; (value = in.get()) != EOF;) {
    seen = true; char c = char(value);
    if (quoted) { if (c == '"') { if (in.peek() == '"') { in.get(); field += '"'; } else quoted = false; } else field += c; }
    else if (c == '"' && field.empty()) quoted = true;
    else if (c == ',') { row.push_back(move(field)); field.clear(); }
    else if (c == '\n' || c == '\r') { if (c == '\r' && in.peek() == '\n') in.get(); row.push_back(move(field)); return true; }
    else field += c;
  }
  if (seen) row.push_back(move(field));
  return seen;
}
json ncu_activity(const fs::path& path, bool graph, const string& kernel) {
  ifstream in(path); if (!in) throw runtime_error("cannot read Nsight Compute export");
  vector<string> header, row; using Key = tuple<string,string,string>; map<Key, map<string,double>> rows; vector<Key> order;
  optional<Pattern> filter; if (!kernel.empty()) filter.emplace(kernel);
  while (csv_row(in, row)) {
    if (has(row, "ID") && ((has(row, "Metric Name") && has(row, "Metric Value")) || has(row, "sm__ctas_launched.sum"))) header = row;
    else if (!header.empty() && row.size() == header.size()) {
      map<string,string> value; for (size_t i = 0; i < row.size(); ++i) value[header[i]] = row[i];
      const auto name = value.contains("Kernel Name") ? value["Kernel Name"] : value["Name"];
      if (value["ID"].empty() || !all_of(value["ID"].begin(), value["ID"].end(), [](unsigned char c) { return isdigit(c); }) || (filter && !filter->search(name))) continue;
      const auto key = tuple{value["Process ID"], value["ID"], name}; map<string,string> metrics;
      if (value.contains("Metric Name")) metrics[value["Metric Name"]] = value["Metric Value"];
      else for (const string metric : {"sm__ctas_launched.sum", "launch__graph_exec_cuda_id"}) if (value.contains(metric)) metrics[metric] = value[metric];
      for (auto [metric, raw] : metrics) {
        erase(raw, ','); try { size_t end; const auto number = stod(raw, &end); if (all_of(raw.begin() + end, raw.end(), [](unsigned char c) { return isspace(c); }) && isfinite(number)) { if (!rows.contains(key)) order.push_back(key); rows[key][metric] = number; } } catch (const invalid_argument&) {} catch (const out_of_range&) {}
      }
    }
  }
  vector<string> names;
  for (const auto& key : order) { auto& metrics = rows[key]; if (metrics["sm__ctas_launched.sum"] > 0 && (!graph || metrics["launch__graph_exec_cuda_id"] > 0)) names.push_back(get<2>(key)); }
  return {{"records", names.size()}, {"names", names}};
}
json nsys_activity(const fs::path& path, bool graph, const string& kernel) {
  sqlite3* raw = nullptr;
  if (sqlite3_open_v2(path.c_str(), &raw, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) { string error = raw ? sqlite3_errmsg(raw) : "open failed"; if (raw) sqlite3_close(raw); throw runtime_error(error); }
  unique_ptr<sqlite3, decltype(&sqlite3_close)> db(raw, sqlite3_close);
  auto query = [&](const string& sql) {
    sqlite3_stmt* prepared = nullptr;
    if (sqlite3_prepare_v2(db.get(), sql.c_str(), -1, &prepared, nullptr) != SQLITE_OK) throw runtime_error(sqlite3_errmsg(db.get()));
    unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(prepared, sqlite3_finalize);
    vector<vector<string>> rows; int status;
    while ((status = sqlite3_step(statement.get())) == SQLITE_ROW) {
      vector<string> row; for (int i = 0; i < sqlite3_column_count(statement.get()); ++i) { const auto* text = sqlite3_column_text(statement.get(), i); row.emplace_back(text ? reinterpret_cast<const char*>(text) : ""); } rows.push_back(move(row));
    }
    if (status != SQLITE_DONE) throw runtime_error(sqlite3_errmsg(db.get()));
    return rows;
  };
  string table = graph ? "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" : "CUPTI_ACTIVITY_KIND_KERNEL";
  if (query("SELECT 1 FROM sqlite_master WHERE name='" + table + "' AND type='table'").empty()) return {{"records", 0}};
  auto rows = query(graph ? "SELECT start,end FROM " + table + " WHERE end>start" :
    "SELECT k.start,k.end,s.value FROM " + table + " k JOIN StringIds s ON k.demangledName=s.id WHERE k.end>k.start");
  set<string> names; size_t records = 0; optional<Pattern> filter; if (!kernel.empty()) filter.emplace(kernel);
  for (const auto& row : rows) if (graph || !filter || filter->search(row[2])) { ++records; if (!graph) names.insert(row[2]); }
  return {{"records", records}, {"names", names}};
}
json evidence(const Collect& args, const Plan& p, json& attempts) {
  const auto& tool = args.tool; Env clean = p.env;
  for (const auto& key : injectors) clean.erase(key);
  clean.erase("INJECTION_PARAM"); clean.erase("INJECTION_METRICS");
  if (tool == "nsys" || tool == "ncu") {
    vector<fs::path> candidates = tool == "nsys" ? vector<fs::path>{args.output / "profile.nsys-rep"} : vector<fs::path>{args.output / "profile.ncu-rep", args.output / "profile.ncu-repz"};
    vector<fs::path> reports; for (const auto& path : candidates) if (fs::is_regular_file(path) && fs::file_size(path)) reports.push_back(path);
    if (reports.size() != 1) return {{"records", 0}, {"reason", "missing or ambiguous profiler report"}};
    const auto exported = args.output / (tool == "nsys" ? "export.sqlite" : "export.stdout.log");
    vector<string> argv = tool == "nsys" ? vector<string>{p.command[0], "export", "--type", "sqlite", "--output", exported.string(), reports[0].string()} :
      vector<string>{p.command[0], "--import", reports[0].string(), "--page", "raw", "--csv", "--rename-kernels", "0", "--print-kernel-base", "demangled"};
    attempts.push_back(execute(argv, clean, args.output, "export", min(args.timeout, 60.0)));
    if (attempts.back()["exit"].get<int>() || attempts.back()["timeout"].get<bool>()) return {{"records", 0}, {"reason", "report export failed"}};
    return tool == "nsys" ? nsys_activity(exported, args.workload == "graph", args.kernel) : ncu_activity(exported, args.workload == "graph", args.kernel);
  }
  if (!p.decoder.empty()) {
    size_t count = 0;
    for (const auto& path : files(args.output)) if (path.filename().string().find("pcsampling") != string::npos && path.extension() == ".dat" && fs::file_size(path)) {
      attempts.push_back(execute({p.decoder.string(), "--file-name", path.string(), "--disable-source-correlation", "--verbose"}, clean, args.output, "decoded-" + to_string(count++), min(args.timeout, 60.0)));
    }
    if (!count) return {{"records", 0}, {"reason", "missing PC sample data"}};
  }
  uint64_t records = 0, dropped = 0; bool summary = false, gpu = false, stopped = false, exited = false, named = false, valid = true;
  optional<Pattern> activity; if (has(sanitizers, tool)) activity.emplace(args.activity);
  const regex summaries(R"((?:ERROR SUMMARY:\s*(\d+) errors|RACECHECK SUMMARY:\s*(\d+) hazards))"),
    instructions(R"(kernel instructions\s+[1-9]\d*)"), memory(R"(MEMTRACE:.*grid_launch_id\s+\d+.*warp\s+\d+.* - 0x[0-9a-fA-F]+)"),
    trace(R"((?:CONCURRENT_KERNEL|KERNEL):\s*(\d+)\s+records)"), range(R"(sm__ctas_launched\.sum\s+([+\d.eE-]+)\s*$)"),
    samples(R"(Total Samples:\s*[1-9]\d*)"), losses(R"(Total Dropped Samples:\s*(\d+))"),
    stop(R"(\[Switching focus to CUDA kernel|CUDA thread .* hit|CUDA kernel entry)"), compiler(R"(ptxas info\s*: Used \d+ registers)"), assembly(R"(/\*[0-9a-fA-F]+\*/\s+\w)");
  for (const auto& path : files(args.output)) if (path.extension() == ".log") {
    ifstream in(path); string line; smatch match;
    while (getline(in, line)) {
      if (has(sanitizers, tool)) {
        gpu |= activity->search(line);
        if (regex_search(line, match, summaries)) { summary = true; valid &= stoull(match[1].matched ? match[1].str() : match[2].str()) == 0; }
      } else if (tool == "nvbit-count" || tool == "nvbit-graph") { records += regex_search(line, instructions); valid &= line.find("ran out of kernel_counters") == string::npos; }
      else if (tool == "nvbit-memory") records += regex_search(line, memory);
      else if (tool == "cupti-trace") { if (regex_search(line, match, trace)) records += stoull(match[1]); }
      else if (tool == "cupti-range") { if (regex_search(line, match, range)) { double n = stod(match[1]); records += isfinite(n) && n > 0; } }
      else if (tool == "cupti-pc") { records += regex_search(line, samples); named |= line.find("functionName:") != string::npos; if (regex_search(line, match, losses)) dropped += stoull(match[1]); }
      else if (tool == "cuda-gdb") { stopped |= regex_search(line, stop); exited |= line.find("exited normally") != string::npos; }
      else if (tool == "compiler") records += regex_search(line, compiler);
      else records += regex_search(line, assembly);
    }
  }
  if (has(sanitizers, tool)) return {{"records", int(summary && valid && gpu)}, {"clean_summary", summary && valid}, {"gpu_receipt", gpu}, {"api_reporting", args.api_errors}};
  if (tool == "cuda-gdb") return {{"records", int(stopped && exited)}, {"device_stop", stopped}};
  if (tool == "cupti-pc") return {{"records", named && !dropped ? records : 0}, {"dropped_samples", dropped}};
  return {{"records", valid ? records : 0}};
}
}
int collect_main(vector<string> argv) {
  using namespace collector;
  const auto delimiter = find(argv.begin(), argv.end(), "--");
  if (argv.empty() || find(argv.begin(), delimiter, "--help") != delimiter) { cout << "collect TOOL OUTPUT [--timeout SECONDS] [--workload cdp|leaf|graph] [--limit N] [--kernel REGEX] [--activity REGEX] [--api-errors explicit|extended|all] -- /absolute/target [arguments]\n"; return argv.empty() ? 2 : 0; }
  Collect args;
  try {
    if (argv.size() < 2) throw runtime_error("tool and output are required");
    vector<string> positional;
    for (size_t i = 0; i < argv.size(); ++i) {
      string key = argv[i], value; auto equal = key.find('='); if (equal != string::npos) { value = key.substr(equal + 1); key.resize(equal); }
      if (key == "--") { positional.insert(positional.end(), argv.begin() + i + 1, argv.end()); break; }
      if (key == "--timeout" || key == "--limit" || key == "--workload" || key == "--kernel" || key == "--activity" || key == "--api-errors") {
        if (equal == string::npos) { if (++i == argv.size()) throw runtime_error("missing value for " + key); value = argv[i]; }
        if (key == "--timeout" || key == "--limit") { size_t end; if (key == "--timeout") args.timeout = stod(value, &end); else args.limit = stoi(value, &end); if (end != value.size()) throw runtime_error("invalid number for " + key); }
        else if (key == "--workload") args.workload = value; else if (key == "--kernel") args.kernel = value; else if (key == "--activity") args.activity = value; else args.api_errors = value;
      } else positional.push_back(argv[i]);
    }
    if (positional.size() < 2) throw runtime_error("tool and output are required");
    args.tool = positional[0]; args.output = fs::absolute(positional[1]).lexically_normal();
    args.command.assign(positional.begin() + 2, positional.end());
    if (!has(collectors, args.tool)) throw runtime_error("unknown collector: " + args.tool);
    if (!has({"cdp", "leaf", "graph"}, args.workload) || !has({"explicit", "extended", "all"}, args.api_errors)) throw runtime_error("invalid workload or API error mode");
    if (args.command.empty() || !fs::path(args.command[0]).is_absolute()) throw runtime_error("append -- /absolute/target [arguments]");
    if (!isfinite(args.timeout) || args.timeout <= 0 || args.limit <= 0) throw runtime_error("timeout and limit must be positive");
    if (!args.kernel.empty() && (!(args.tool == "nsys" || args.tool == "ncu") || args.workload == "graph")) throw runtime_error("kernel filters require non-aggregate Nsight collection");
    if (fs::exists(args.output)) throw runtime_error("output directory already exists");
    if (!fs::create_directories(args.output)) throw runtime_error("cannot create output directory");
  } catch (const exception& e) { cerr << "collect: " << e.what() << '\n'; return 2; }
  json receipt{{"schema", "gh.diagnostic.v1"}, {"tool", args.tool}, {"workload", args.workload}, {"ranking_valid", false},
    {"attribution", args.workload == "cdp" ? "host parent call tree" : args.workload}, {"requested_command", args.command}, {"api_errors", args.api_errors}, {"attempts", json::array()}, {"status", "preparing"}};
  const auto path = args.output / "receipt.json"; auto save = [&] { write(path, receipt.dump(2) + "\n"); }; save();
  try {
    auto p = plan(args); receipt["assets"] = p.assets; json changes = json::object(); const auto inherited = environment();
    for (const auto& [key, value] : p.env) if (!inherited.contains(key) || inherited.at(key) != value) changes[key] = value;
    receipt["environment"] = changes; receipt["status"] = "running"; save();
    receipt["attempts"].push_back(execute(p.command, p.env, args.output, "run", args.timeout));
    receipt["activity"] = evidence(args, p, receipt["attempts"]);
    bool passed = receipt["activity"]["records"].get<uint64_t>() > 0;
    for (const auto& attempt : receipt["attempts"]) passed &= attempt["exit"].get<int>() == 0 && !attempt["timeout"].get<bool>();
    receipt["status"] = passed ? "passed" : "failed";
  } catch (const exception& e) { receipt["status"] = "failed"; receipt["error"] = string("runtime_error: ") + e.what(); }
  receipt["artifacts"] = json::array(); for (const auto& artifact : files(args.output)) if (artifact != path) receipt["artifacts"].push_back(identity(artifact));
  save(); cout << receipt["status"].get<string>() << ": " << path.string() << '\n'; return receipt["status"] == "passed" ? 0 : 1;
}
int erasure_main(vector<string> args) {
  if (args.size() != 1 || args[0] == "--help") { cout << "erasure PTX\n"; return args.size() == 1 ? 0 : 2; }
  const fs::path path = args[0]; const string source = read(path); smatch match;
  const regex entry(R"(\.entry\s+\S*disabled_probe\S*\s*\([^)]*\)\s*(?:\.\w+[^\n]*\n\s*)*\{([\s\S]*?)^\})", regex::ECMAScript | regex::multiline);
  const string body = regex_search(source, match, entry) ? match[1].str() : "";
  const regex banned(R"(globaltimer|\b(?:ld\.|call\b|atom\.|red\.|bar\.|bra\b))"); vector<string> forbidden;
  for (sregex_iterator it(body.begin(), body.end(), banned), end; it != end; ++it) forbidden.push_back(it->str());
  const bool passed = !body.empty() && body.find("st.global.u32") != string::npos && forbidden.empty();
  cout << json{{"scope", "disabled marker PTX erasure, no GPU execution"}, {"path", fs::canonical(path).string()}, {"sha256", digest(source)},
    {"passed", passed}, {"forbidden", forbidden}, {"kernel_body", body}}.dump(2) << '\n'; return passed ? 0 : 1;
}
}

/*
Development source inventory contract:
Input: UTF-8 maintained sources and an audited prior JSON receipt. Output: byte,
character, normalized noncomment-line, lexical-token and literal inventories.
This is compiler/artifact infrastructure, executed on the host; it never executes
production or test computation. Entire physical files and every tagged category
byte are counted. Markers are standalone comments at declaration boundaries.
C++ uses the historical unexpanded preprocessing-token lexer and pinned
clang-format 21 style. ICU supplies Unicode word/decimal/space properties. Each
normalization owns a separate temporary file; subprocess completion precedes
reuse. Inputs are rehashed after measurement. No source files are modified.
Python AST normalization is NOT reimplemented: historical measurements are used
only with matching byte hashes and disclosed receipt identity. Baseline file-set
and every file hash must match the frozen receipt. No equivalence or speed claim
follows from source size. Fair comparison records full scope, physical totals,
category totals, formatter version and immutable baseline provenance; compare
complete inventories, not handpicked algorithms or generated-code exclusions.
*/
#include <algorithm>
#include <array>
#include <cstdint>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>
#include <unicode/uchar.h>
#include <unistd.h>

namespace dev {
std::string read(const std::filesystem::path&);
void write(const std::filesystem::path&, const std::string&);
std::string digest(std::string_view);
nlohmann::json identity(const std::filesystem::path&);
std::map<std::string, std::string> environment();
nlohmann::json execute(const std::vector<std::string>&,
                      const std::map<std::string, std::string>&,
                      const std::filesystem::path&, const std::string&, double);
namespace source_size {
using J = nlohmann::json;
namespace F = std::filesystem;
constexpr std::string_view style = "{BasedOnStyle: LLVM, ColumnLimit: 100, IndentWidth: 2, SortIncludes: Never, AllowShortFunctionsOnASingleLine: None, AllowShortIfStatementsOnASingleLine: Never, AllowShortLoopsOnASingleLine: false, AllowShortBlocksOnASingleLine: Never}";
constexpr std::array fields{"files", "source_bytes", "source_characters",
  "formatted_noncomment_loc", "lexical_tokens", "literal_source_bytes",
  "literal_source_lines", "literal_payload_words"};
constexpr std::array categories{"production", "tests", "tooling", "capability-references"};
const std::map<std::string, std::vector<std::string>> roots{
  {"production", {"include", "src", "training/include", "training/src"}},
  {"tests", {"tests", "training/tests"}},
  {"tooling", {"tools", "bench", "support", "cmake", "training/tools", "training/bench", "training/profiling"}}};
const std::set<std::string> old_skips{"__pycache__", ".git", "vendor", "third_party", "external", "node_modules", "build", "_build", "CMakeFiles"};
constexpr std::string_view reference_root = "results/booster-level-batch-20260922/real";
constexpr std::string_view marker = "// GH_SOURCE_CATEGORY:";

struct Point { UChar32 value; std::size_t end; };
Point point(std::string_view s, std::size_t p) {
  if (p >= s.size()) throw std::runtime_error("unexpected UTF-8 end");
  auto first = static_cast<unsigned char>(s[p]);
  unsigned width = first < 0x80 ? 1 : first >= 0xc2 && first <= 0xdf ? 2 :
    first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
  if (!width || p + width > s.size()) throw std::runtime_error("invalid UTF-8 source");
  UChar32 value = first & (width == 1 ? 0x7f : (1u << (7 - width)) - 1);
  for (unsigned k = 1; k < width; ++k) {
    auto c = static_cast<unsigned char>(s[p + k]);
    if ((c & 0xc0) != 0x80) throw std::runtime_error("invalid UTF-8 continuation");
    value = (value << 6) | (c & 0x3f);
  }
  if ((width == 2 && value < 0x80) || (width == 3 && value < 0x800) ||
      (width == 4 && value < 0x10000) || value > 0x10ffff ||
      (value >= 0xd800 && value <= 0xdfff)) throw std::runtime_error("invalid UTF-8 scalar");
  return {value, p + width};
}
bool space(UChar32 c) { return u_isUWhiteSpace(c) || (c >= 0x1c && c <= 0x1f); }
bool decimal(UChar32 c) { return u_charType(c) == U_DECIMAL_DIGIT_NUMBER; }
bool word(UChar32 c) {
  int type = u_charType(c);
  return c == '_' || (type >= U_UPPERCASE_LETTER && type <= U_OTHER_LETTER) ||
    type == U_DECIMAL_DIGIT_NUMBER || type == U_LETTER_NUMBER || type == U_OTHER_NUMBER;
}
std::size_t characters(std::string_view s) {
  std::size_t count = 0;
  for (std::size_t p = 0; p < s.size(); ++count) p = point(s, p).end;
  return count;
}
std::string trim(std::string_view s) {
  std::size_t begin = 0, last = 0;
  while (begin < s.size() && space(point(s, begin).value)) begin = point(s, begin).end;
  for (std::size_t p = begin; p < s.size();) {
    auto c = point(s, p); p = c.end;
    if (!space(c.value)) last = p;
  }
  return std::string(s.substr(begin, last > begin ? last - begin : 0));
}
bool starts(std::string_view s, std::size_t p, std::string_view prefix) {
  return p <= s.size() && s.substr(p).starts_with(prefix);
}
std::size_t identifier_end(std::string_view s, std::size_t p, bool ascii_start = false) {
  if (p == s.size()) return p;
  auto c = point(s, p);
  bool valid = ascii_start ? ((c.value >= 'a' && c.value <= 'z') ||
    (c.value >= 'A' && c.value <= 'Z') || c.value == '_') : word(c.value) && !decimal(c.value);
  if (!valid) return p;
  p = c.end;
  while (p < s.size() && word(point(s, p).value)) p = point(s, p).end;
  return p;
}
enum Kind { comment, literal, number, identifier, punctuation };
struct Token { Kind kind; std::size_t begin, end; };
std::vector<Token> lex(std::string_view s, bool cmake = false) {
  characters(s);
  static const std::vector<std::string> operators = [] {
    std::vector<std::string> v{"<<<", ">>>", "%:%:", "<=>", ">>=", "<<=", "->*", "...", "::", ".*", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "*=", "/=", "%=", "+=", "-=", "&=", "^=", "|=", "##", "<:", ":>", "<%", "%>", "%:"};
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
    return v;
  }();
  std::vector<Token> result;
  for (std::size_t p = 0; p < s.size();) {
    auto begin = p;
    auto c = point(s, p);
    if (space(c.value)) { p = c.end; continue; }
    if (!cmake && starts(s, p, "\\\n")) { p += 2; continue; }
    auto emit = [&](Kind kind, std::size_t end) { result.push_back({kind, begin, end}); p = end; };
    if (cmake) {
      std::size_t b = p + (s[p] == '#'), q = b;
      if (q < s.size() && s[q] == '[') {
        ++q;
        while (q < s.size() && s[q] == '=') ++q;
        if (q < s.size() && s[q] == '[') {
          std::string close = "]" + std::string(s.substr(b + 1, q - b - 1)) + "]";
          auto end = s.find(close, q + 1);
          if (end == s.npos) throw std::runtime_error("unterminated CMake bracket argument");
          emit(s[p] == '#' ? comment : literal, end + close.size()); continue;
        }
      }
      if (s[p] == '#') { auto end = s.find('\n', p); emit(comment, end == s.npos ? s.size() : end); continue; }
    } else if (starts(s, p, "//")) {
      p += 2;
      while (p < s.size()) {
        if (starts(s, p, "\\\r\n")) p += 3;
        else if (starts(s, p, "\\\n")) p += 2;
        else if (s[p] == '\n') break;
        else p = point(s, p).end;
      }
      emit(comment, p); continue;
    } else if (starts(s, p, "/*")) {
      auto end = s.find("*/", p + 2);
      if (end == s.npos) throw std::runtime_error("unterminated C++ comment");
      emit(comment, end + 2); continue;
    }
    bool matched = false;
    if (!cmake) for (auto prefix : {"u8R\"", "uR\"", "UR\"", "LR\"", "R\""}) {
      if (!starts(s, p, prefix)) continue;
      std::size_t d = p + std::string_view(prefix).size(), q = d, length = 0;
      while (q < s.size() && length <= 16 && s[q] != '(') {
        auto k = point(s, q);
        if (k.value == ' ' || k.value == ')' || k.value == '\\' || k.value == '\t' || k.value == '\r' || k.value == '\n') break;
        q = k.end; ++length;
      }
      if (q >= s.size() || length > 16 || s[q] != '(') continue;
      std::string close = ")" + std::string(s.substr(d, q - d)) + "\"";
      auto end = s.find(close, q + 1);
      if (end == s.npos) throw std::runtime_error("unterminated C++ raw string");
      end = identifier_end(s, end + close.size()); emit(literal, end); matched = true; break;
    }
    if (matched) continue;
    for (auto prefix : {"u8", "u", "U", "L", ""}) {
      if (!starts(s, p, prefix)) continue;
      std::size_t q = p + std::string_view(prefix).size();
      if (q >= s.size() || (s[q] != '\'' && s[q] != '"')) continue;
      char quote = s[q++];
      while (q < s.size() && s[q] != quote) {
        if (s[q] == '\\') { ++q; if (q == s.size()) break; }
        q = point(s, q).end;
      }
      if (q < s.size()) { emit(literal, identifier_end(s, q + 1, true)); matched = true; break; }
    }
    if (matched) continue;
    bool numeric = decimal(c.value) || (s[p] == '.' && p + 1 < s.size() && decimal(point(s, p + 1).value));
    if (numeric) {
      p = s[p] == '.' ? point(s, p + 1).end : c.end;
      while (p < s.size()) {
        if ((s[p] == 'e' || s[p] == 'E' || s[p] == 'p' || s[p] == 'P') && p + 1 < s.size() && (s[p + 1] == '+' || s[p + 1] == '-')) p += 2;
        else if (word(point(s, p).value) || s[p] == '.' || s[p] == '\'') p = point(s, p).end;
        else break;
      }
      emit(number, p); continue;
    }
    auto end = identifier_end(s, p);
    if (end > p) { emit(identifier, end); continue; }
    for (const auto& op : operators) if (starts(s, p, op)) {
      emit(punctuation, p + op.size()); matched = true; break;
    }
    if (matched) continue;
    if (s[p] == '\'' || s[p] == '"') throw std::runtime_error("unterminated quoted literal");
    emit(punctuation, c.end);
  }
  return result;
}
std::string without_comments(std::string_view source, bool cmake = false) {
  std::string result;
  std::size_t previous = 0;
  for (const auto& t : lex(source, cmake)) if (t.kind == comment) {
    result += source.substr(previous, t.begin - previous);
    for (auto p = t.begin; p < t.end;) {
      auto c = point(source, p); result += c.value == '\n' ? '\n' : ' '; p = c.end;
    }
    previous = t.end;
  }
  result += source.substr(previous);
  return result;
}
std::vector<std::string> spellings(std::string_view source, bool cmake = false) {
  std::vector<std::string> result;
  for (auto t : lex(source, cmake)) if (t.kind != comment) result.emplace_back(source.substr(t.begin, t.end - t.begin));
  return result;
}
std::string cmake_format(std::string_view source) {
  std::string result, line;
  int depth = 0;
  for (const auto& value : spellings(source, true)) {
    if (characters(line) + characters(value) + 1 > 100 && !line.empty()) { result += line + '\n'; line = "  "; }
    line += (trim(line).empty() ? "" : " ") + value;
    depth += (value == "(") - (value == ")");
    if (depth < 0) throw std::runtime_error("unbalanced CMake command");
    if (value == ")" && depth == 0) { result += line + '\n'; line.clear(); }
  }
  if (depth) throw std::runtime_error("unbalanced CMake command");
  if (!line.empty()) result += line + '\n';
  return result.empty() ? "\n" : result;
}
std::string language(const F::path& path) {
  static const std::set<std::string> cpp{".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx", ".cuh", ".inc", ".def", ".inl", ".ipp", ".tpp"};
  std::string ext = path.extension().string(), lower = ext;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (cpp.contains(lower)) return "cuda-cpp";
  if (ext == ".py") return "python";
  if (path.filename() == "CMakeLists.txt" || ext == ".cmake") return "cmake";
  if (ext == ".json") return "json-registry";
  return {};
}
std::uint64_t payload_words(std::string_view s) {
  std::uint64_t n = 0;
  for (std::size_t p = 0; p < s.size();) {
    auto c = point(s, p); p = c.end;
    if (space(c.value)) continue;
    ++n;
    if (word(c.value)) while (p < s.size() && word(point(s, p).value)) p = point(s, p).end;
  }
  return n;
}
J empty_totals() { J result = J::object(); for (auto f : fields) result[f] = 0; return result; }
void add(J& target, const J& row, bool files = true) {
  for (auto f : fields) if (files || std::string_view(f) != "files")
    target[f] = target.value(f, std::uint64_t(0)) + row.value(f, std::uint64_t(0));
}
struct Context {
  std::string formatter;
  F::path scratch;
  unsigned serial{};
  J prior;
  std::string prior_hash;
  explicit Context(std::string f) : formatter(std::move(f)) {
    std::string pattern = (F::temp_directory_path() / "gh-source-size-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(), pattern.end()); bytes.push_back(0);
    char* p = ::mkdtemp(bytes.data());
    if (!p) throw std::runtime_error("cannot create source-size temporary directory");
    scratch = p;
  }
  ~Context() {
    if (std::uncaught_exceptions()) { std::cerr << "source-size development artifacts preserved: " << scratch << '\n'; return; }
    std::error_code ec; F::remove_all(scratch, ec);
  }
  std::string command(std::vector<std::string> args) {
    std::string label = "format-" + std::to_string(serial++);
    auto receipt = dev::execute(args, dev::environment(), scratch, label, 60);
    if (receipt.value("exit", -1) != 0 || receipt.value("timeout", false))
      throw std::runtime_error("source-size subprocess failed: " + receipt.dump());
    return dev::read(scratch / (label + ".stdout.log"));
  }
  std::string normalize(std::string_view source, const std::string& kind) {
    if (kind == "cuda-cpp") {
      F::path input = scratch / ("input-" + std::to_string(serial) + ".cpp");
      dev::write(input, without_comments(source));
      return command({formatter, "--style=" + std::string(style), "--assume-filename=source.cpp", "--Werror", input.string()});
    }
    if (kind == "cmake") return cmake_format(source);
    if (kind == "json-registry") return nlohmann::ordered_json::parse(source).dump(2, ' ', false) + '\n';
    throw std::runtime_error("normalization unavailable for " + kind);
  }
};
std::uint64_t nonempty_lines(std::string_view source) {
  std::uint64_t n = 0;
  std::size_t begin = 0;
  for (std::size_t p = 0; p < source.size();) {
    auto c = point(source, p);
    bool boundary = (c.value >= 10 && c.value <= 13) || (c.value >= 0x1c && c.value <= 0x1e) ||
      c.value == 0x85 || c.value == 0x2028 || c.value == 0x2029;
    if (boundary) {
      n += !trim(source.substr(begin, p - begin)).empty();
      p = c.end;
      if (c.value == '\r' && p < source.size() && source[p] == '\n') ++p;
      begin = p;
    } else p = c.end;
  }
  if (begin < source.size()) n += !trim(source.substr(begin)).empty();
  return n;
}
J measure_text(Context& context, std::string_view source, const std::string& kind) {
  auto normalized = context.normalize(source, kind);
  auto tokens = lex(normalized, kind == "cmake");
  auto original = lex(source, kind == "cmake");
  J result = empty_totals();
  result["files"] = 1;
  result["source_bytes"] = source.size(); result["source_characters"] = characters(source);
  result["sha256"] = dev::digest(source); result["normalized_sha256"] = dev::digest(normalized);
  result["formatted_noncomment_loc"] = nonempty_lines(normalized);
  result["lexical_tokens"] = std::count_if(tokens.begin(), tokens.end(), [](auto t) { return t.kind != comment; });
  std::uint64_t literal_bytes = 0, literal_lines = 0, literal_words = 0;
  for (auto t : original) if (t.kind == literal) {
    auto value = source.substr(t.begin, t.end - t.begin);
    literal_bytes += value.size(); literal_lines += 1 + std::count(value.begin(), value.end(), '\n'); literal_words += payload_words(value);
  }
  result["literal_source_bytes"] = literal_bytes; result["literal_source_lines"] = literal_lines; result["literal_payload_words"] = literal_words;
  return result;
}
struct Section { std::string category; std::size_t begin, end; };
std::vector<Section> sections(std::string_view source, std::string fallback) {
  std::vector<Section> result;
  std::size_t begin = 0;
  for (auto t : lex(source)) if (t.kind == comment) {
    auto value = source.substr(t.begin, t.end - t.begin);
    if (!value.starts_with(marker)) continue;
    auto line = source.rfind('\n', t.begin); line = line == source.npos ? 0 : line + 1;
    if (!trim(source.substr(line, t.begin - line)).empty()) throw std::runtime_error("source category marker must occupy its own line");
    std::string category = trim(value.substr(marker.size()));
    if (std::find(categories.begin(), categories.end(), category) == categories.end()) throw std::runtime_error("unknown source category: " + category);
    if (t.begin > begin) result.push_back({fallback, begin, t.begin});
    fallback = std::move(category); begin = t.begin;
  }
  if (begin < source.size() || result.empty()) result.push_back({fallback, begin, source.size()});
  return result;
}
struct Inventory { std::map<F::path, std::string> included; std::vector<std::string> excluded; };
Inventory inventory(const F::path& root, bool historical) {
  Inventory result;
  for (const auto& [category, directories] : roots) for (const auto& directory : directories) {
    auto base = root / directory;
    if (!F::exists(base)) continue;
    for (const auto& entry : F::recursive_directory_iterator(base)) {
      if (!entry.is_regular_file()) continue;
      auto relative = entry.path().lexically_relative(root);
      bool skip = false;
      if (historical) for (const auto& part : relative.parent_path()) if (old_skips.contains(part.string())) skip = true;
      if (skip) continue;
      if (!language(entry.path()).empty()) result.included[relative] = category;
      else if (entry.path().extension() != ".md" && entry.path().extension() != ".pyc") result.excluded.push_back(relative.generic_string());
    }
  }
  for (const auto& entry : F::directory_iterator(root)) {
    if (!entry.is_regular_file()) continue;
    std::string kind = language(entry.path());
    if (entry.path().filename() == "CMakeLists.txt" || (!historical && !kind.empty()))
      result.included[entry.path().filename()] = "tooling";
  }
  if (F::is_regular_file(root / "training/CMakeLists.txt")) result.included["training/CMakeLists.txt"] = "tooling";
  auto refs = root / reference_root;
  if (F::exists(refs)) for (const auto& entry : F::directory_iterator(refs)) if (entry.is_regular_file() && entry.path().extension() == ".py")
    result.included[entry.path().lexically_relative(root)] = "capability-references";
  std::sort(result.excluded.begin(), result.excluded.end());
  return result;
}
const J* prior_file(const Context& context, const std::string& relative, const std::string& hash) {
  for (auto name : {"baseline", "fresh"}) if (context.prior.contains(name))
    for (const auto& row : context.prior[name]["files"])
      if (row.value("path", "") == relative && row.value("sha256", "") == hash) return &row;
  return nullptr;
}
J measure_file(Context& context, const F::path& root, const F::path& relative, const std::string& category) {
  auto source = dev::read(root / relative);
  auto kind = language(relative);
  J result;
  if (kind == "python") {
    auto frozen = prior_file(context, relative.generic_string(), dev::digest(source));
    if (!frozen) throw std::runtime_error("Python AST normalization requires an exact-hash audited receipt: " + relative.generic_string());
    result = *frozen;
    result["source_characters"] = characters(source);
    result["normalization_provenance"] = {{"mode", "frozen-python-ast-receipt"}, {"receipt_sha256", context.prior_hash}};
  } else result = measure_text(context, source, kind);
  result["path"] = relative.generic_string(); result["category"] = category; result["language"] = kind;
  J pieces = J::array();
  if (kind == "cuda-cpp") {
    auto parts = sections(source, category);
    for (const auto& part : parts) {
      J row = parts.size() == 1 ? result : measure_text(context, std::string_view(source).substr(part.begin, part.end - part.begin), kind);
      row.erase("sections"); row["category"] = part.category;
      row["begin_byte"] = part.begin; row["end_byte"] = part.end;
      pieces.push_back(std::move(row));
    }
  } else { J row = result; row["begin_byte"] = 0; row["end_byte"] = source.size(); pieces.push_back(std::move(row)); }
  result["sections"] = std::move(pieces);
  if (result["sections"].size() > 1) result["category"] = "partitioned";
  else result["category"] = result["sections"][0]["category"];
  return result;
}
J current_tree(Context& context, const F::path& requested) {
  auto root = F::canonical(requested);
  auto selected = inventory(root, false);
  J files = J::array(), totals = J::object(), languages = J::object(), physical = empty_totals();
  for (auto category : categories) totals[category] = empty_totals();
  std::map<std::string, std::vector<std::string>> duplicates;
  for (const auto& [relative, category] : selected.included) {
    J row = measure_file(context, root, relative, category);
    add(physical, row);
    std::string kind = row["language"];
    if (!languages.contains(kind)) languages[kind] = empty_totals();
    add(languages[kind], row);
    std::set<std::string> seen;
    std::uint64_t counted = 0;
    for (const auto& part : row["sections"]) {
      std::string c = part["category"]; add(totals[c], part, false); seen.insert(c); counted += part["source_bytes"].get<std::uint64_t>();
    }
    if (counted != row["source_bytes"].get<std::uint64_t>()) throw std::runtime_error("source partition lost bytes");
    for (const auto& c : seen) totals[c]["files"] = totals[c]["files"].get<std::uint64_t>() + 1;
    duplicates[row["sha256"].get<std::string>()].push_back(relative.generic_string()); files.push_back(std::move(row));
  }
  for (const auto& row : files) if (dev::digest(dev::read(root / row["path"].get<std::string>())) != row["sha256"].get<std::string>())
    throw std::runtime_error("source changed during audit: " + row["path"].get<std::string>());
  J identical = J::array(); for (const auto& [hash, paths] : duplicates) if (paths.size() > 1) identical.push_back(paths);
  return {{"root", root.string()}, {"totals", totals}, {"physical_totals", physical}, {"languages", languages}, {"files", files},
    {"identical_files_counted_separately", identical}, {"unclassified_files_requiring_review", selected.excluded},
    {"section_normalization", "Each tagged section is independently normalized; physical totals normalize whole files. Put markers at complete declaration boundaries. Category files count each physical file once; category bytes partition all physical bytes."}};
}
J frozen_tree(Context& context, const F::path& requested) {
  auto root = F::canonical(requested);
  const J* original = nullptr;
  for (auto key : {"baseline", "fresh"}) if (context.prior.contains(key)) {
    F::path recorded = context.prior[key].value("root", "");
    if (!recorded.empty() && F::weakly_canonical(recorded) == root) original = &context.prior[key];
  }
  if (!original) throw std::runtime_error("audited receipt has no tree for requested baseline: " + root.string());
  auto selected = inventory(root, true);
  std::set<std::string> expected, actual;
  for (const auto& [path, category] : selected.included) actual.insert(path.generic_string());
  J result = *original;
  for (auto& row : result["files"]) {
    std::string relative = row["path"];
    if (!expected.insert(relative).second) throw std::runtime_error("duplicate path in baseline receipt");
    F::path path(relative);
    if (path.is_absolute() || path.lexically_normal() != path || relative.starts_with("../")) throw std::runtime_error("invalid baseline receipt path");
    auto source = dev::read(root / path);
    if (dev::digest(source) != row["sha256"].get<std::string>() || source.size() != row["source_bytes"].get<std::uint64_t>()) throw std::runtime_error("baseline hash/size mismatch: " + relative);
    if (!selected.included.contains(path) || selected.included.at(path) != row["category"].get<std::string>()) throw std::runtime_error("baseline category mismatch: " + relative);
    row["source_characters"] = characters(source);
  }
  if (actual != expected) throw std::runtime_error("baseline source inventory differs from audited receipt");
  if (J(selected.excluded) != result["unclassified_files_requiring_review"]) throw std::runtime_error("baseline unclassified-file inventory differs from receipt");
  J totals = J::object(), langs = J::object(), physical = empty_totals();
  for (auto category : categories) totals[category] = empty_totals();
  for (const auto& row : result["files"]) {
    add(physical, row); add(totals[row["category"].get<std::string>()], row);
    std::string kind = row["language"]; if (!langs.contains(kind)) langs[kind] = empty_totals(); add(langs[kind], row);
  }
  for (auto category : categories) for (auto f : fields) if (std::string_view(f) != "source_characters" && totals[category][f] != (*original)["totals"][category][f])
    throw std::runtime_error("baseline receipt totals are inconsistent");
  for (const auto& row : result["files"])
    if (dev::digest(dev::read(root / row["path"].get<std::string>())) != row["sha256"].get<std::string>())
      throw std::runtime_error("baseline source changed during audit: " + row["path"].get<std::string>());
  result["totals"] = totals; result["languages"] = langs; result["physical_totals"] = physical;
  result["measurement_provenance"] = {{"mode", "frozen-audited-receipt-all-source-hashes-verified"},
    {"receipt_sha256", context.prior_hash}, {"verified_files", expected.size()},
    {"limitation", "Historical lexical and normalized measurements are retained, not rerun. Python AST unparse is not reimplemented. Historical directory exclusions remain disclosed."},
    {"historical_excluded_directory_names", old_skips}};
  return result;
}
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(std::string("source-size fixture failed: ") + message); }
void self_test(Context& context) {
  std::string source = "auto s=R\"x(a/*not comment*/\nb)x\"; // comment\nint x=1'000; /* block */\n";
  require(spellings(source) == std::vector<std::string>{"auto", "s", "=", "R\"x(a/*not comment*/\nb)x\"", ";", "int", "x", "=", "1'000", ";"}, "raw/comment/digit separator tokenization");
  require(without_comments(source).find("not comment") != std::string::npos && without_comments(source).find("/* block */") == std::string::npos, "literal comment markers preserved");
  require(without_comments("// a\\\ncontinued\nint x;\n").find("continued") == std::string::npos, "continued line comment");
  require(spellings("u8\"a\\\"//b\"_s L'\\n' 0x1.fp+2 <<< >>> >>= <=>") == std::vector<std::string>{"u8\"a\\\"//b\"_s", "L'\\n'", "0x1.fp+2", "<<<", ">>>", ">>=", "<=>"}, "escaped/prefixed literals and longest punctuation");
  require(cmake_format("set(X \"#kept\") # ignored\n#[[gone]]\nset(Y [=[a#b]=])\n") == "set ( X \"#kept\" )\nset ( Y [=[a#b]=] )\n", "CMake bracket/comment normalization");
  require(characters("\xc3\xa9 \xf0\x9f\x98\x80") == 3 && payload_words("\xc3\xa9_1 + x") == 3, "UTF-8 word/byte accounting");
  auto normalized = context.normalize("int f(){return 1;}\n", "cuda-cpp");
  require(std::count(normalized.begin(), normalized.end(), '\n') == 3, "pinned C++ normalization");
  auto parts = sections("// GH_SOURCE_CATEGORY: production\nint x;\n// GH_SOURCE_CATEGORY: tests\nint y;\n", "tooling");
  require(parts.size() == 2 && parts[0].category == "production" && parts[1].category == "tests" && parts[0].end == parts[1].begin, "source section partition");
  require(sections("auto s=R\"x(\n// GH_SOURCE_CATEGORY: tests\n)x\";\n", "production").size() == 1, "marker text inside raw literal ignored");
  for (auto broken : {"/*", "\"unfinished", "R\"x(unfinished"}) {
    bool rejected = false; try { lex(broken); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "unterminated source rejected");
  }
}
} // namespace source_size

int source_size_main(std::vector<std::string> args) {
  namespace S = source_size;
  std::filesystem::path baseline = "/home/b/gpu_histogram-archive-20260923", fresh = std::filesystem::current_path(), output, receipt;
  std::string formatter = "clang-format-21";
  bool self = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "--self-test") { self = true; continue; }
    if (arg == "--help" || arg == "-h") {
      std::cout << "source-size [--baseline PATH] [--fresh PATH] [--baseline-receipt JSON] [--clang-format EXECUTABLE] [--output NEW_JSON] [--self-test]\n"
        "Read-only source audit; baseline uses an audited receipt after file-set and SHA256 verification.\n"
        "Standalone // GH_SOURCE_CATEGORY: production|tests|tooling comments assign subsequent bytes. Untagged root support files count as tooling.\n";
      return 0;
    }
    if (i + 1 == args.size()) throw std::runtime_error("missing value for " + arg);
    if (arg == "--baseline") baseline = args[++i];
    else if (arg == "--fresh") fresh = args[++i];
    else if (arg == "--baseline-receipt") receipt = args[++i];
    else if (arg == "--clang-format") formatter = args[++i];
    else if (arg == "--output") output = args[++i];
    else throw std::runtime_error("unknown source-size option: " + arg);
  }
  if (!output.empty() && std::filesystem::exists(output)) throw std::runtime_error("refusing to overwrite source-size receipt");
  S::Context context(formatter);
  std::string version = S::trim(context.command({formatter, "--version"}));
  if (version.find("version 21.") == std::string::npos) throw std::runtime_error("source-size requires clang-format major 21");
  if (self) {
    S::self_test(context);
    std::cout << S::J{{"schema", "gh.source-size.v2"}, {"development_self_test", "passed"}, {"formatter", version}, {"python_ast_normalizer", "frozen-receipt-only"}}.dump() << '\n';
    return 0;
  }
  if (receipt.empty()) receipt = fresh / "observations/source-size/current-20260923.json";
  std::string frozen = dev::read(receipt); context.prior = S::J::parse(frozen); context.prior_hash = dev::digest(frozen);
  if (context.prior.value("schema", "") != "gh.source-size.v1" || context.prior.value("style", "") != S::style)
    throw std::runtime_error("baseline receipt must use the audited source-size v1 normalization contract");
  UVersionInfo unicode; u_getUnicodeVersion(unicode); char unicode_text[U_MAX_VERSION_STRING_LENGTH]; u_versionToString(unicode, unicode_text);
  auto old = S::frozen_tree(context, baseline), current = S::current_tree(context, fresh);
  if (dev::digest(dev::read(receipt)) != context.prior_hash) throw std::runtime_error("baseline receipt changed during audit");
  S::J report{{"schema", "gh.source-size.v2"}, {"tool", dev::identity("/proc/self/exe")}, {"capability_equivalence", "unproven"}, {"reduction_claim_permitted", false},
    {"formatter", version}, {"style", S::style}, {"unicode_properties", unicode_text}, {"scope", S::roots},
    {"root_source_scope", "Every root-level C/C++/CUDA/Python/CMake file; all legacy maintained-source roots recursively; no nested directory exclusions. Root bytes default to tooling until tagged."},
    {"excluded_directory_names", S::J::array()}, {"artifact_scope", "Build output and observations are external development artifacts outside the maintained-source roots; historical top-level Python capability references are separately counted."},
    {"reference_scope", std::string(S::reference_root) + "/*.py (top level only)"},
    {"baseline_receipt", {{"path", std::filesystem::canonical(receipt).string()}, {"bytes", frozen.size()}, {"sha256", context.prior_hash}}},
    {"tool_binary", dev::identity("/proc/self/exe")}, {"baseline", old}, {"fresh", current}};
  std::string text = report.dump(2) + '\n';
  if (output.empty()) std::cout << text;
  else {
    // Exclusive creation is performed before transport, preserving the previous
    // tool's refusal to replace receipts even if another writer races this one.
    int descriptor = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (descriptor < 0) throw std::runtime_error("cannot exclusively create source-size receipt: " + output.string());
    std::size_t done = 0;
    while (done < text.size()) {
      ssize_t n = ::write(descriptor, text.data() + done, text.size() - done);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) { ::close(descriptor); throw std::runtime_error("failed writing source-size receipt"); }
      done += static_cast<std::size_t>(n);
    }
    if (::close(descriptor)) throw std::runtime_error("failed closing source-size receipt");
    std::cout << S::J{{"receipt", std::filesystem::absolute(output).string()}, {"sha256", dev::digest(text)},
      {"baseline", old["totals"]}, {"fresh", current["totals"]}, {"physical", current["physical_totals"]}, {"reduction_claim_permitted", false}}.dump(2) << '\n';
  }
  return !old["unclassified_files_requiring_review"].empty() || !current["unclassified_files_requiring_review"].empty() ? 2 : 0;
}
} // namespace dev

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: gh_tools {collect|source-size|erasure} [arguments...]\n");
    return 2;
  }
  const std::string command = argv[1];
  const std::vector<std::string> arguments(argv + 2, argv + argc);
  try {
    if (command == "collect") return dev::collect_main(arguments);
    if (command == "source-size" || command == "source_size") return dev::source_size_main(arguments);
    if (command == "erasure" || command == "check-erasure") return dev::erasure_main(arguments);
    std::fprintf(stderr, "unknown development tool: %s\n", argv[1]);
    return 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gh tools: %s\n", error.what());
    return 1;
  }
}

#elif GH_MODE == 12
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
// Byte transport only. Lengths are file extents, not interpreted record shapes.
struct Blob {
  void* device{};
  std::size_t bytes{};
  ~Blob() { if (device) cudaFree(device); }
  bool read(const char* path) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror(path); return false; }
    struct stat info{};
    if (fstat(fd, &info) || info.st_size <= 0) {
      std::fprintf(stderr, "cannot size input: %s\n", path);
      close(fd);
      return false;
    }
    bytes = static_cast<std::size_t>(info.st_size);
    void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { std::perror(path); return false; }
    auto error = cudaMalloc(&device, bytes);
    if (error == cudaSuccess) error = cudaMemcpy(device, mapped, bytes, cudaMemcpyHostToDevice);
    munmap(mapped, bytes);
    if (error != cudaSuccess)
      std::fprintf(stderr, "transport: %s: %s\n", path, cudaGetErrorString(error));
    return error == cudaSuccess;
  }
};
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: quality DATA.ghb MODEL.ghb PREDICTIONS.f64 [QUALITY.json]\n");
    return 2;
  }
  Blob data, model, predictions, quality;
  if (!data.read(argv[1]) || !model.read(argv[2]) || !predictions.read(argv[3])) return 1;
  if (argc == 5 && !quality.read(argv[4])) return 1;
  void* arena{};
  auto error = cudaMalloc(&arena, gh_quality_arena_bytes);
  if (error != cudaSuccess) {
    std::fprintf(stderr, "arena: %s\n", cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  error = gh_quality_launch(data.device, data.bytes, model.device, model.bytes,
      predictions.device, predictions.bytes, quality.device, quality.bytes,
      arena, gh_quality_arena_bytes);
  const auto completed = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = completed;
  if (error != cudaSuccess)
    std::fprintf(stderr, "quality completion: %s\n", cudaGetErrorString(error));
  cudaFree(arena);
  return static_cast<int>(error);
}

#else
static_assert(GH_MODE >= 1 && GH_MODE <= 13);
#if GH_OBSERVE && GH_MODE <= 10
#include <nvtx3/nvToolsExt.h>
#endif

int main() {
  const auto setup = gh_initialize(GH_MODE);
  if (setup != cudaSuccess) return static_cast<int>(setup);
#if GH_OBSERVE && GH_MODE <= 10
  nvtxRangePushA("gh GPU checks: launch through completion");
#endif
  const auto submitted = gh_launch(GH_MODE);
  const auto completed = cudaDeviceSynchronize();
  const auto error = submitted == cudaSuccess ? completed : submitted;
  const auto result = error == cudaSuccess ? gh_after(GH_MODE) : error;
#if GH_OBSERVE && GH_MODE <= 10
  nvtxRangePop();
#endif
  return static_cast<int>(result);
}
#endif
