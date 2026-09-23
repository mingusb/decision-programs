# Deeper histogram SM86 instruction inspection

CPU-only inspection of the compiled object, from `/home/b/gpu_histogram`:

```sh
/usr/local/cuda/bin/cuobjdump --dump-sass build/booster-reuse/CMakeFiles/ghb.dir/src/deeper_histogram.cu.o > results/booster-reuse-20260922/deeper-sass.txt
```

The command exited 0. `cuobjdump --version` reports CUDA 13.4, V13.4.49,
build `cuda_13.4.r13.4/compiler.38536908_0`. The dump identifies `sm_86` and
the source `training/src/deeper_histogram.cu`. No GPU workload or profiler was
run to obtain this evidence.

SHA-256 at capture:

| Artifact | SHA-256 |
|---|---|
| cuobjdump executable | `4628edae91e9c7293330a3a5ef2f2916b767a2e00871d461f0221733682e6c91` |
| deeper_histogram.cu | `198382fa4b9a74b00602fc6a464c3f6d43c5fd04a90510b84b9b1fc395714060` |
| deeper_histogram.cu.o | `95990bc2f1b74e4728d7c1069cee283c5108ed7c2340056a3c31ebfa342efcca` |
| deeper-sass.txt | `4851b0a93f1c7bbbed4930e88e478b939200b9a91d4018115bf0129fa3dd78a1` |

The width-4 and width-8 `shared_accumulate` instantiations each contain
`ATOMS.CAST.SPIN.64` at instruction offsets `0xc70`, `0xd20`, and `0xdb0`.
The width-1 instantiation contains the same instruction at `0xae0`, `0xb90`,
and `0xc20`. The first two sites follow `LDS.64` and `DADD`; the third follows
`LDS.64`, `IADD3`, and `IMAD.X` with an increment of one. Together with the
source's three shared `atomicAdd` operations, these identify compare-and-swap
loops for FP64 gradient, FP64 Hessian, and uint64 row count. Each site has a
conditional branch back to its load/update sequence when the update retries.

The shared flush and `global_accumulate` use `RED.E.ADD.F64.RN.STRONG.GPU`
for gradient/Hessian and `RED.E.ADD.64.STRONG.GPU` for counts. These are global
atomic reductions whose old values are not consumed by the source.

This establishes the compiler's instruction choice. It does not establish the
number of retries, contention, achieved occupancy, memory transactions, kernel
duration, or complete-operation ranking. Those require runtime measurements.
In particular, fewer global atomics can coexist with more shared update work;
the disassembly alone cannot decide which policy wins.
