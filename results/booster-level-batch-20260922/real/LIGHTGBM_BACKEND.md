# LightGBM CUDA backend evidence

Reviewed 2026-09-22 using local source, installed library, retained build logs and
completed validation model files. No import of GPU libraries, CUDA call, training,
prediction, or profiler run was performed for this review. Existing runners and
evidence were not modified.

The combined evidence supports native **LightGBM CUDA tree training** for the
recorded benchmark settings. The wrapper's `backend: cuda` field alone is not
verification: `framework.py:106` records Python `m.params`, and line 168 labels the
requested backend. Native saved model parameters and executable dispatch provide
the additional evidence below. This is source/configuration/binary evidence, not
a per-job kernel trace or a claim that every stage executes on GPU.

## Installed build and source identity

All workspace-relative paths below resolve from `/home/b/gpu_histogram`;
`real/` abbreviates `results/booster-level-batch-20260922/real/`.

* Installed library:
  `build/benchmark-env/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.so`,
  16,956,040 bytes, SHA256
  `bdb2b38a3650cd28336afa9547cb4f5bf4fc7e414d9169002db57582e8b6c6b1`.
  A fresh hash matches `real/environment-final.json` under package LightGBM 4.7.0.
* Cached wheel:
  `/home/b/.cache/uv/sdists-v9/pypi/lightgbm/4.7.0/MuVG_WUWmeTiLjZM/76aff8cd46fbe23a/lightgbm-4.7.0-py3-none-linux_x86_64.whl`,
  SHA256 `15d7f62da161cbf1fcda5e946d0d3d8c5e356364720c57ac3f0ff630807b6324`.
  Its `lightgbm/lib/lib_lightgbm.so` member hashes identically to the installed
  library. Reading and hashing this ZIP member required no package import.
* Cached source root, denoted **S** below:
  `/home/b/.cache/uv/sdists-v9/pypi/lightgbm/4.7.0/MuVG_WUWmeTiLjZM/src`.
  The retained compiler command in
  `real/lightgbm-cuda-build-shared.stderr:577` names this source root and
  `-DUSE_CUDA`. Line 626 records the attempted CMake configuration including
  `-DUSE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=86-real` and shared NCCL.
  **That attempt failed**; it is not evidence of a successful completed build.
  The subsequent `real/lightgbm-cuda-build-cpath.stderr:1–8` records successful
  build and installation of LightGBM 4.7.0 into `build/benchmark-env`, but does not
  contain a full verbose compiler transcript. The wheel/library match and CUDA
  symbols below supply additional successful-artifact evidence.
* Installed `lightgbm/libpath.py:21–35,49` resolves native library candidates and
  loads the first existing candidate with `ctypes.cdll.LoadLibrary`. The package
  native library is the path recorded in the environment manifest. A filesystem
  check found this as the only existing candidate among the three Linux paths.

Read-only `nm -D --defined-only .../lib_lightgbm.so | c++filt` found these exported
definitions in the installed library:

```text
000000000099d120 T LightGBM::CUDAHistogramConstructor::ConstructHistogramForLeaf(...)
00000000009a6100 T LightGBM::CUDASingleGPUTreeLearner::Init(LightGBM::Dataset const*, bool)
00000000009a50e0 T LightGBM::CUDASingleGPUTreeLearner::Train(float const*, float const*, bool)
```

Symbol presence proves implementation availability, not by itself runtime use.
The recorded model configuration and source dispatch address selection.

## Native dispatch and the relevant fallback conditions

* `S/src/io/config.cpp:192–205` recognizes `device_type=cuda` as its own device
  setting. `S/src/treelearner/tree_learner.cpp:47–52` dispatches CUDA plus the
  serial learner directly to `CUDASingleGPUTreeLearner`; unsupported learner types
  cause a fatal error. The CPU learner is selected by the separate CPU branch at
  lines 19–32, not by a CUDA-branch fallback.
* Without compiled CUDA support,
  `S/src/treelearner/cuda/cuda_single_gpu_tree_learner.hpp:165–177` defines a stub
  whose constructor raises `Log::Fatal` with “CUDA Tree Learner was not enabled
  in this build.” It does not successfully train a CPU substitute.
* The compiled implementation at
  `S/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp:32–64` selects the CUDA
  device and initializes CUDA leaf, histogram, data-partition and split-finder
  components. Its training implementation calls the CUDA histogram constructor
  at lines 170–210. Host setup and `SerialTreeLearner::Init` at line 33 remain
  part of this native implementation; a CUDA backend does not mean no CPU work.
* There **is** a warned CPU tree fallback for `linear_tree=true` in
  `S/src/io/config.cpp:425–429`. The inspected native models record
  `linear_tree: 0`, so this condition is inactive here.
* CUDA objective selection is conditional too.
  `S/src/objective/objective_function.cpp:72–77` chooses the CUDA objective factory
  for CUDA, non-GOSS sampling and non-random-forest boosting. The benchmark models
  record `boosting: gbdt` and `data_sample_strategy: bagging`; their regression,
  binary and multiclass objectives map to CUDA objective implementations at
  lines 28–29, 40–41 and 46–47. Other objectives have explicit CPU boosting
  fallbacks at lines 50–67; they are not the benchmark objectives.
* `S/src/boosting/gbdt.cpp:111–118` separately selects GPU boosting from CUDA
  objective support and the sampler's Hessian-change property, then constructs
  the configured tree learner. The bagging sampler returns false from
  `IsHessianChange()` at `S/src/boosting/bagging.hpp:218–219`. This explains why
  the recorded settings do not activate the CPU boosting condition there.

Thus the bounded conclusion is that the inspected settings select the CUDA tree
learner and supported CUDA objectives. It would be incorrect to claim LightGBM
contains no CPU fallback paths for any possible configuration.

## Native saved model evidence

These four completed validation artifacts cover regression, binary, multiclass
and independent multilabel training. All record `boosting: gbdt`,
`tree_learner: serial`, `device_type: cuda`, `data_sample_strategy: bagging`,
`linear_tree: 0`, `gpu_device_id: 0` and `num_gpu: 1`.

| Artifact under `real/validation/` | Native objective | `device_type` line | `linear_tree` line | SHA256 |
|---|---|---:|---:|---|
| `validation-wine-g0-lightgbm/result/model-0.txt` | regression | 507 | 574 | `8680d56342a3d103f6c5d83c3bd62c0b44d96c69748bba1667a41366c993f668` |
| `validation-magic-g0-lightgbm/result/model-0.txt` | binary | 504 | 571 | `48e8ffdbbc9562450b94a4901073b7d2055a70ca4678a0778aac37982d875eaf` |
| `validation-letter-g0-lightgbm/result/model-0.txt` | multiclass | 12387 | 12454 | `62a4736d86d9ad7a647afd38b8060fc9d3523bdc1e6b1478a1e33caa914ca0f9` |
| `validation-delicious-g0-lightgbm/result/model-0.txt` | binary, label 0 | 124 | 191 | `f77c984fd7c328caf9f4b51d197339c020b49e2e9ca297b588e21d54193a48e7` |

These are native serialized training configuration values, not just the wrapper's
requested Python dictionary: `S/src/boosting/gbdt_model_text.cpp:397–399` writes
`config_->ToString()` into the model's `parameters` section. This review sampled
the four files above; it is not a fresh exhaustive audit of every model file or
every subsequent test repetition.

`real/validation/validation-magic-g0-lightgbm/stdout:1` contains the warning
“Using sparse features with CUDA is currently not supported.” Its source is
`S/src/io/dataset.cpp:357–363`: CUDA configuration disables sparse storage and
continues with dense storage. It does not replace the learner with CPU or remove
features. This warning is supporting configuration evidence, **not** a CUDA
trainer startup banner or proof that a particular CUDA kernel executed.

The multilabel wrapper retains every output: `real/framework.py:94–103` iterates
all independent targets, records constant-label predictors separately, and trains
each remaining target with the CUDA configuration at lines 78–83. The complete
loop is timed. This source review does not change the distinct CPU prediction
label at lines 107–122 or the native comparison limits in `real/PROTOCOL.md`.

## Review hashes and limits

SHA256 values below identify the exact reviewed source and evidence. `real/`
means `results/booster-level-batch-20260922/real/` here.

| Path | SHA256 |
|---|---|
| `S/CMakeLists.txt` | `1ed3c79759eeabd3b3ffacb2abf69377a4b5455c47ea2afcbbda1b702685a7cd` |
| `S/src/treelearner/tree_learner.cpp` | `fdb0724763cbc6ec9e87376c251c2fb6f1537be95c002fe8fc78e09f0253a5bc` |
| `S/src/treelearner/cuda/cuda_single_gpu_tree_learner.hpp` | `572e2ad864335c7dee8d94abccda011cbf94e3f0ade1da8eb15f67e3952d0d71` |
| `S/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` | `8271fab425733023f75f1b0d9c38e01e923c6aaa47f0466ef837e0a1634c7d17` |
| `S/src/io/config.cpp` | `2335f222503bbaf9186fe1aa320c8043934beb929e38a48df1e89b50168a2b35` |
| `S/src/io/dataset.cpp` | `6aa4cfb894e62a86bbd7712248a01162b95561335f2fa5ef4c444f055647c6da` |
| `S/src/boosting/gbdt.cpp` | `d2f5d9b5878c3d171dec21b680233e37e7a895007730339f8adb8175448c2b3f` |
| `S/src/boosting/gbdt_model_text.cpp` | `8b951c2b90e6010928333b5379014aeef0f326f10bc3ce1f4a614bf60c4c170c` |
| `S/src/boosting/bagging.hpp` | `8feae67b152c4853b80181841625fb34cf252ce9d020f9026654229cefe57d49` |
| `S/src/objective/objective_function.cpp` | `1b5555652e64a30de3cabe736b9da0da70b9e6d58dc5dd47462f625f40deb5e0` |
| `build/benchmark-env/lib/python3.12/site-packages/lightgbm/libpath.py` | `5a330bae64674386a611349a7bb7e505914d05227f63c500c08b1bfb086857cb` |
| `real/framework.py` | `ac0051654f3162231ba1975687c5e21bf4f3daf0c5e9d02a52b9943d4e136e11` |
| `real/environment-final.json` | `557e8a4ecc73becc8135e3a7e844bef2d8f60d7438324178044f3077f9dcda7b` |
| `real/lightgbm-cuda-build-cpath.stderr` | `ed98a245bf5bde6a1e4306e0f67bdb1f3aa15d552f8e7da30acb67ff2745da31` |
| `real/lightgbm-cuda-build-shared.stderr` | `cfe29928f010408f24d1d1593ec1b6dd744bbb40c2a9f9eafae562de608e202d` |
| `real/validation/validation-magic-g0-lightgbm/stdout` | `2142aebcc1a4df748a2c0a7adc7b3fe24e17a6c55513ac5128f2fd8642b23837` |

No rebuilt binary, dynamic loader trace, per-job kernel trace or reproducible-build
attestation was produced. Cached sources and the wheel have external cache
lifetimes; their paths and content hashes are retained here. The results support
the CUDA backend attribution for the inspected configuration while preserving
CPU preprocessing/setup, CPU public prediction, native algorithm differences,
and the absence of a GPU inference comparison for LightGBM.
