#include "check.cuh"
#include "gh/count.cuh"
#include "../bench/frozen_count.cuh"

int main() {
  const auto setup = gh::count::initialize_runtime();
  if (setup != cudaSuccess) return static_cast<int>(setup);
  const auto reference_setup = gh::bench::frozen_count::initialize_runtime();
  if (reference_setup != cudaSuccess) return static_cast<int>(reference_setup);
  gh::test::run<<<1,1>>>();
  return static_cast<int>(cudaDeviceSynchronize());
}
