#pragma once

// Optional benchmark capture boundaries, never production algorithm policy.
// Set GH_PROFILE_CAPTURE=1 only for diagnostic runs. The default path performs
// no CUDA/NVTX calls or synchronization. Enabled boundaries belong outside any
// CUDA stream capture: they bracket complete graph construction/replay or stream
// operations and synchronize only at their beginning/end. Timings from these
// runs include profiling effects and must not rank ordinary benchmark results.
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gh::profiling {

class CaptureRange {
 public:
  explicit CaptureRange(const char* name) {
    const char* setting = std::getenv("GH_PROFILE_CAPTURE");
    if (!setting || setting[0] != '1' || setting[1] != '\0') return;
    check(cudaDeviceSynchronize(), "synchronize before diagnostic capture");
    check(cudaProfilerStart(), "start diagnostic capture");
    domain_ = nvtxDomainCreateA("ghb");
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = nvtxDomainRegisterStringA(domain_, name);
    nvtxDomainRangePushEx(domain_, &attributes);
    active_ = true;
  }

  CaptureRange(const CaptureRange&) = delete;
  CaptureRange& operator=(const CaptureRange&) = delete;
  ~CaptureRange() noexcept {
    if (active_) {
      // Preserve the original exception on an exceptional benchmark exit.
      // Normal exits call finish() explicitly so CUDA errors are reported.
      cudaDeviceSynchronize();
      close();
    }
  }

  void finish() {
    if (!active_) return;
    const auto completion = cudaDeviceSynchronize();
    const auto stopped = close();
    check(completion, "complete diagnostic capture");
    check(stopped, "stop diagnostic capture");
  }

 private:
  static void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
  }

  cudaError_t close() noexcept {
    nvtxDomainRangePop(domain_);
    const auto result = cudaProfilerStop();
    nvtxDomainDestroy(domain_);
    active_ = false;
    return result;
  }

  nvtxDomainHandle_t domain_{};
  bool active_{};
};

}  // namespace gh::profiling
