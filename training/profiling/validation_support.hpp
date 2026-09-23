#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace diagnostic {
inline std::size_t checks{};
inline void require(bool condition, const std::string& what) {
  ++checks;
  if (!condition) throw std::runtime_error(what);
}
inline void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
inline std::string quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
inline void number(std::ostream& out, long double x) {
  if (std::isfinite(x)) out << std::setprecision(21) << x; else out << "null";
}
template<class T> void array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; number(out, values[i]); }
  out << ']';
}
inline bool same(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }
struct Report {
  FILE* file{};
  explicit Report(const std::string& path) {
    file = std::fopen(path.c_str(), "wx");
    if (!file) throw std::runtime_error("cannot create new report: " + path);
  }
  ~Report() { if (file) std::fclose(file); }
  void write(const std::string& text) {
    if (std::fwrite(text.data(), 1, text.size(), file) != text.size() || std::fflush(file))
      throw std::runtime_error("cannot write report");
  }
};
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
struct Event {
  cudaEvent_t value{};
  Event() { check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
  ~Event() { if (value) cudaEventDestroy(value); }
};
struct Graph {
  cudaGraph_t value{}; cudaGraphExec_t executable{};
  ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (value) cudaGraphDestroy(value); }
  template<class F> void capture(cudaStream_t stream, F launch) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try { launch(); }
    catch (...) { cudaGraph_t abandoned{}; cudaStreamEndCapture(stream, &abandoned); if (abandoned) cudaGraphDestroy(abandoned); throw; }
    check(cudaStreamEndCapture(stream, &value));
    check(cudaGraphInstantiate(&executable, value, nullptr, nullptr, 0));
  }
  void launch(cudaStream_t stream) { check(cudaGraphLaunch(executable, stream)); }
};
template<class T> struct Device {
  T* value{}; std::size_t size{};
  explicit Device(std::size_t n) : size(n) { check(cudaMalloc(reinterpret_cast<void**>(&value), std::max(std::size_t(1), n) * sizeof(T))); }
  ~Device() { cudaFree(value); }
  Device(const Device&) = delete;
  void put(const std::vector<T>& host, cudaStream_t stream) {
    require(host.size() == size, "upload size contract");
    check(cudaMemcpyAsync(value, host.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> host(size);
    check(cudaMemcpyAsync(host.data(), value, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream)); return host;
  }
};
inline std::string device_metadata() {
  int device{}, runtime{}, driver{}; cudaDeviceProp prop{};
  check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop, device));
  check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
  std::ostringstream out;
  out << "{\"name\":" << quote(prop.name) << ",\"compute_major\":" << prop.major
      << ",\"compute_minor\":" << prop.minor << ",\"runtime\":" << runtime << ",\"driver\":" << driver << '}';
  return out.str();
}
} // namespace diagnostic
