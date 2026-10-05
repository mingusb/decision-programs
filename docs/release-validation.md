# Release validation

Decision Programs 0.1.0 was exercised on Linux under WSL2 with an NVIDIA RTX A5000 Laptop GPU (compute capability 8.6), CUDA 13.4, GCC 15.2 and CMake 4.2. The native training library was CUDA-enabled XGBoost 3.4.1.

## Completed checks

- The complete C++23/CUDA release build succeeded.
- All 16 CTest cases passed: 14 host cases and two GPU cases. The CLI transport case covers 14 scenarios; threshold formatting covers 12 assertions, including exact FP32 round trips and signed zero.
- A fresh installation outside the source checkout ran `doctor`, `demo`, `train`, `convert`, `predict`, `inspect`, `explain`, `export`, `simplify`, `evaluate`, `hpo` and `rl` through the public `decision-programs` entrypoint.
- The small CSV workflow trained three classes, converted and simplified the model, and reported zero runtime/native class mismatches on all 30 example rows. These examples exercise the interface; they are not an accuracy benchmark.
- The documented nonlinear combination workflow completed: two native teacher candidates, checkpoint-based prefix reuse, and out-of-fold composition selected on VALID.
- The runtime demonstration checked 48 decision paths across canonical and compact layouts, and rejected insufficient path capacity and aliased outputs.
- Compute Sanitizer memcheck, racecheck, initcheck and synccheck each completed through `decision-programs profile` with zero reported errors; racecheck also reported zero hazards and warnings. This covers the runtime demonstration, not every possible model or kernel.
- The specialized RL interface completed a 32-episode run, accepted learned policy state and completed its source conversion in the supported qualification environment. This is integration evidence, not a controlled demonstration of an RL compression advantage.
- All 31 maintained unique Lean modules were elaborated and independently replayed with Lean 4.34.1 and `leanchecker`. These are mathematical proofs under their declared premises; they do not establish a formal refinement of the compiled CUDA implementation.

The supported inputs, numerical assumptions and specialized qualification requirements are described in the [project guide](../PROJECT_GUIDE.md) and [CLI reference](CLI.md). A successful small conversion does not imply that the full 448-tree Forest conversion has completed; it has not.
