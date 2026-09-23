#include "check.cuh"
#if GH_OBSERVE
#include <nvtx3/nvToolsExt.h>
#endif
#ifdef GH_COUNT_RUNTIME
#include "gh/count.cuh"
cudaError_t count_graph_checks();
#endif
int main() {
#ifdef GH_COUNT_RUNTIME
  const auto setup = gh::count::initialize_runtime();
  if (setup != cudaSuccess) return static_cast<int>(setup);
#endif
#if GH_OBSERVE
  nvtxRangePushA("gh GPU checks: launch through completion");
#endif
  gh::test::run<<<1, 1>>>();
  const auto result = cudaDeviceSynchronize();
#ifdef GH_COUNT_RUNTIME
  const auto completed = result == cudaSuccess ? count_graph_checks() : result;
#else
  const auto completed = result;
#endif
#if GH_OBSERVE
  nvtxRangePop();
#endif
  return static_cast<int>(completed);
}
