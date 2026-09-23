#include "check.cuh"
#include "gh/types.cuh"

namespace gh::test {
namespace {
__device__ __align__(16) std::byte storage[128];
__device__ Status status;
__device__ Status planning_status;
__device__ u32 observed;
__global__ void produce(u32 value) { observed = value; }
__global__ void verify() {
  succeeded(status);
  GH_CHECK(observed == 29);
  printf("PASS core: checked extents, arena bounds, ordered child/tail completion\n");
}
}
__global__ void run() {
  GH_CHECK(add_fits(UINT64_MAX, 0));
  GH_CHECK(!add_fits(UINT64_MAX, 1));
  GH_CHECK(mul_fits(UINT64_MAX, 1));
  GH_CHECK(!mul_fits(UINT64_MAX, 2));
  GH_CHECK(ceil_div(UINT64_MAX, 256) == (UINT64_MAX / 256) + 1);
  GH_CHECK(contains(Array<double>{nullptr, 0}, 0));
  GH_CHECK(!contains(Array<double>{nullptr, 1}, 1));
  auto& local = planning_status;
  local = {};
  Arena layout{{storage, 128}};
  GH_CHECK(layout.take<double>(0) == nullptr && layout.used == 0);
  GH_CHECK(layout.take<u32>(1) == reinterpret_cast<u32*>(storage));
  GH_CHECK(layout.take<double>(2) == reinterpret_cast<double*>(storage + 8));
  GH_CHECK(layout.fits(&local) && local.required_bytes == 24);
  GH_CHECK(!layout.take<double>(UINT64_MAX));
  GH_CHECK(!layout.fits(&local) && (local.errors & extent));
  local = {};
  Arena small{{storage, 7}};
  GH_CHECK(!small.take<double>(1));
  GH_CHECK(!small.fits(&local) && (local.errors & capacity));
  local = {};
  Arena wrapped{{reinterpret_cast<std::byte*>(UINT64_MAX - 15), 32}};
  GH_CHECK(!wrapped.take<double>(4));
  GH_CHECK(!wrapped.fits(&local) && (local.errors & extent));
  status = {};
  produce<<<1, 1>>>(11);
  produce<<<1, 1>>>(29);
  submitted(finish(&status));
  verify<<<1, 1, 0, cudaStreamTailLaunch>>>();
}
}
