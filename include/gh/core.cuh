#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gh {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
template<class T> struct Array { T* data{}; u64 size{}; };
struct Workspace { std::byte* data{}; u64 bytes{}; };
enum Error : u32 { shape = 1, extent = 2, capacity = 4, input = 8,
                   model = 16, numeric = 32, unsupported = 64, runtime = 128 };
struct Status { u32 errors{}, done{}; u64 required_bytes{}; };

__device__ inline void fail(Status* status, Error error) {
  atomicOr(&status->errors, static_cast<u32>(error));
}
__host__ __device__ constexpr bool add_fits(u64 a, u64 b) { return b <= UINT64_MAX - a; }
__host__ __device__ constexpr bool mul_fits(u64 a, u64 b) { return !b || a <= UINT64_MAX / b; }
__host__ __device__ constexpr u64 ceil_div(u64 n, u64 d) { return n / d + (n % d != 0); }
template<class T> __device__ bool contains(Array<T> a, u64 n) {
  const auto address = reinterpret_cast<std::uintptr_t>(a.data);
  return n <= a.size && mul_fits(n, sizeof(T)) &&
    (!n || (a.data && address % alignof(T) == 0 && add_fits(address, n * sizeof(T))));
}

// A local layout cursor: size calculation and partitioning use the same code.
// Null backing storage computes required bytes; dereference requires capacity.
struct Arena {
  Workspace storage;
  u64 used{};
  bool valid{true};
  template<class T> __device__ T* take(u64 n) {
    if (!n) return nullptr;
    constexpr u64 mask = alignof(T) - 1;
    if (!valid || !add_fits(used, mask) || !mul_fits(n, sizeof(T))) {
      valid = false; return nullptr;
    }
    const u64 begin = (used + mask) & ~mask;
    const u64 bytes = n * sizeof(T);
    if (!add_fits(begin, bytes)) { valid = false; return nullptr; }
    used = begin + bytes;
    return storage.data && used <= storage.bytes && add_fits(reinterpret_cast<std::uintptr_t>(storage.data), used)
      ? reinterpret_cast<T*>(storage.data + begin) : nullptr;
  }
  __device__ bool fits(Status* status) const {
    status->required_bytes = used;
    if (!valid) { fail(status, extent); return false; }
    if (!add_fits(reinterpret_cast<std::uintptr_t>(storage.data), used)) { fail(status, extent); return false; }
    if (used > storage.bytes || (used && (!storage.data || reinterpret_cast<std::uintptr_t>(storage.data) % 16))) {
      fail(status, capacity); return false;
    }
    return true;
  }
};

// Called by one GPU coordinator thread. Status is caller-initialized and alive
// through child completion. A tail continuation observes child results safely.
__global__ void complete(Status* status);
__device__ cudaError_t finish(Status* status);
}
