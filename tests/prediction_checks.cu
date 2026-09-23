#include "gh/prediction.cuh"
#include "gh/data.cuh"
#include "gh/model.cuh"
#include "gh/learn.cuh"
#include "check.cuh"
#include <cuda/std/bit>
#include <cmath>

namespace gh::test {
namespace {
constexpr double guard = -987654.25;
constexpr u64 scratch_bytes = 1 << 20;
__device__ const unsigned char legacy_predictions[64] = {
  0,0,0,0,0,0,0,0,       0,0,0,0,0,0,0,0x80,
  0,0,0,0,0,0,0xf0,0x3f, 0,0,0,0,0,0,0,0xc0,
  0,0,0,0,0,0,0x10,0,    1,0,0,0,0,0,0,0,
  0xff,0xff,0xff,0xff,0xff,0xff,0xef,0x7f,
  0xff,0xff,0xff,0xff,0xff,0xff,0xef,0xff};
__device__ const u64 expected_bits[8] = {0,0x8000000000000000ULL,0x3ff0000000000000ULL,
  0xc000000000000000ULL,0x0010000000000000ULL,1,0x7fefffffffffffffULL,0xffefffffffffffffULL};
// Literal GHBDS001 little-endian examples. Unused row padding is excluded from
// each exact file extent; no production encoder generates these input fixtures.
__device__ const unsigned char legacy_datasets[4][80] = {
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 4,0,0,0, 1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0xc0, 0,0,0,0xc0, 0,0,0,0x40, 0,0,0,0x40},
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 4,0,0,0, 1,0,0,0, 2,0,0,0, 0,0,0,0, 0,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0xc0, 0,0,0x80,0x40, 0,0,0,0xc0, 0,0,0x80,0x40,
   0,0,0,0x40, 0,0,0x80,0xc0, 0,0,0,0x40, 0,0,0x80,0xc0},
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 4,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 2,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0, 0,0,0,0, 0,0,0x80,0x3f, 0,0,0x80,0x3f},
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 6,0,0,0, 1,0,0,0, 1,0,0,0, 2,0,0,0, 3,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0xbf,
   0,0,0x80,0x3f, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0, 0,0,0x80,0x3f, 0,0,0,0x40,
   0,0,0,0, 0,0,0x80,0x3f, 0,0,0,0x40}};
__device__ std::byte wire[82];
__device__ double values[10];
__device__ Status status;
__device__ u64 invalid_bits(u32 k) {
  return k == 10 ? 0x7ff0000000000000ULL : k == 11 ? 0xfff0000000000000ULL :
    k == 12 ? 0x7ff8000000000123ULL : 0x7ff0000000000001ULL;
}
__device__ u32 error_for(u32 k) {
  return k < 2 ? 0 : k == 2 ? shape : k == 3 ? extent : k < 6 ? input : k < 10 ? capacity : numeric;
}
__global__ void codec(u32 id);
__global__ void pipeline(u32 id, u32 phase);
__global__ void codec_check(u32 id) {
  const bool decode = id < 14;
  const u32 k = id % 14;
  const u32 expected_error = !decode && (k == 4 || k == 5) ? capacity : error_for(k);
  GH_CHECK(status.done && status.errors == expected_error);
  if (k != 2 && k != 3) GH_CHECK(status.required_bytes == (k == 1 ? 0 : 64));
  if (decode) {
    for (u32 i = 0; i < 10; ++i) {
      const bool written = i > 0 && i <= 8 && (k == 0 || (k >= 10 && i > 1));
      GH_CHECK(cuda::std::bit_cast<u64>(values[i]) == (written ? expected_bits[i-1] : cuda::std::bit_cast<u64>(guard)));
    }
    for (u32 i = 0; i < 82; ++i) {
      const auto expected = !i || i >= 65 ? std::byte{0xa5} : k >= 10 && i < 9
        ? std::byte(invalid_bits(k) >> (8*(i-1))) : std::byte(legacy_predictions[i-1]);
      GH_CHECK(wire[i] == expected);
    }
  } else {
    for (u32 i = 0; i < 82; ++i) {
      const bool written = i >= 1 && i < 65 && (k == 0 || (k >= 10 && i >= 9));
      GH_CHECK(wire[i] == (written ? std::byte(legacy_predictions[i-1]) : std::byte{0xa5}));
    }
    for (u32 i = 0; i < 8; ++i)
      GH_CHECK(cuda::std::bit_cast<u64>(values[i+1]) == (k >= 10 && !i ? invalid_bits(k) : expected_bits[i]));
    GH_CHECK(values[0] == guard && values[9] == guard);
  }
  if (id + 1 < 28) codec<<<1,1,0,cudaStreamTailLaunch>>>(id+1);
  else pipeline<<<1,1,0,cudaStreamTailLaunch>>>(0,0);
  submitted(cudaGetLastError());
}
__global__ void codec(u32 id) {
  const bool decode = id < 14;
  const u32 k = id % 14;
  for (auto& x : wire) x = std::byte{0xa5};
  for (auto& x : values) x = guard;
  for (u32 i = 0; i < 8; ++i) {
    if (!decode) values[i+1] = cuda::std::bit_cast<double>(expected_bits[i]);
    else for (u32 b = 0; b < 8; ++b) wire[1+8*i+b] = std::byte(legacy_predictions[8*i+b]);
  }
  if (k >= 10) {
    if (decode) for (u32 b = 0; b < 8; ++b) wire[1+b] = std::byte(invalid_bits(k) >> (8*b));
    else values[1] = cuda::std::bit_cast<double>(invalid_bits(k));
  }
  u32 rows = k == 1 ? 0 : k == 3 ? UINT32_MAX : 2;
  u32 outputs = k == 2 ? 0 : k == 3 ? UINT32_MAX : 4;
  Array<std::byte> bytes{wire+1,k == 1 ? 0u : k == 4 ? 63u : k == 5 && decode ? 65u : 64u};
  Array<double> data{values+1,k == 6 ? 7u : 8u};
  if (!decode && k == 5) data.size = 7;
  if (k == 7) data.data = reinterpret_cast<double*>(reinterpret_cast<std::byte*>(values)+1);
  if (k == 8) bytes.data = nullptr;
  if (k == 9) bytes.data = reinterpret_cast<std::byte*>(UINT64_MAX-3);
  status = {};
  if (decode) submitted(decode_predictions({bytes.data,bytes.size},rows,outputs,data,&status));
  else submitted(encode_predictions({data.data,data.size},rows,outputs,bytes,&status));
  codec_check<<<1,1,0,cudaStreamTailLaunch>>>(id);
  submitted(cudaGetLastError());
}

struct ModelStorage {
  Feature features[3]; float metadata[10]; u32 offsets[4];
  Node nodes[10]; Tree trees[5]; double base[5]; u64 output_offsets[6];
};
__device__ ModelStorage original_storage{}, restored_storage{};
__device__ Model original, restored;
__device__ Schema fitted;
__device__ DatasetRecord record;
__device__ Training training;
__device__ float x[8], y[10];
__device__ std::uint16_t bins[8];
__device__ double margins[20], losses[4], before[20], after[20], raw[20], decoded[20];
__device__ std::byte dataset_bytes[82], model_bytes[1026], prediction_bytes[146];
__device__ u64 model_size;
__device__ __align__(16) std::byte scratch[scratch_bytes+32];
__device__ Workspace workspace() { return {scratch+16,scratch_bytes}; }
__device__ Model initialize_model(ModelStorage& storage) {
  for (auto& f : storage.features) f = {777,777,FeatureType::categorical};
  for (auto& v : storage.metadata) v = float(guard);
  for (auto& v : storage.offsets) v = 777;
  for (auto& n : storage.nodes) n = {-1,-1,-1,0,0,guard};
  for (auto& t : storage.trees) t = {777,777,777};
  for (auto& v : storage.base) v = guard;
  for (auto& v : storage.output_offsets) v = 777;
  return {{{storage.features+1,1},{storage.metadata+1,8},{storage.offsets+1,2}},
    {storage.nodes+1,8},{storage.trees+1,3},{storage.base+1,3},{storage.output_offsets+1,4}};
}
__device__ void model_guards(const ModelStorage& s) {
  GH_CHECK(s.features[0].begin == 777 && s.features[2].begin == 777);
  GH_CHECK(s.metadata[0] == float(guard) && s.metadata[9] == float(guard));
  GH_CHECK(s.offsets[0] == 777 && s.offsets[3] == 777);
  GH_CHECK(s.nodes[0].value == guard && s.nodes[9].value == guard);
  GH_CHECK(s.trees[0].begin == 777 && s.trees[4].begin == 777);
  GH_CHECK(s.base[0] == guard && s.base[4] == guard);
  GH_CHECK(s.output_offsets[0] == 777 && s.output_offsets[5] == 777);
}
__device__ double analytical(u32 id, u32 row, u32 output, bool transformed) {
  if (id == 3) return transformed ? 1./3. : -1.0986122886681098;
  const double sign = row < 2 ? -1. : 1.;
  if (id == 2 && transformed) return row < 2 ? .11920292202211755 : .8807970779778823;
  return sign * (id == 1 && output ? -4. : 2.);
}
__device__ void check_prediction(u32 id, const double* prediction, bool transformed) {
  for (u32 r = 0; r < record.data.rows; ++r) for (u32 o = 0; o < original.outputs; ++o) {
    const double actual = prediction[u64(r)*original.outputs+o], expected = analytical(id,r,o,transformed);
    GH_CHECK(isfinite(actual) && fabs(actual-expected) <= 4e-15);
  }
}
__global__ void pipeline(u32 id, u32 phase) {
  if (phase) succeeded(status);
  const u64 prior_required = status.required_bytes;
  status = {};
  const u64 cells = phase ? u64(record.data.rows)*original.outputs : 0;
  switch (phase) {
    case 0:
      original = initialize_model(original_storage); restored = initialize_model(restored_storage);
      fitted = original.schema;
      for (auto& v : x) v = float(guard);
      for (auto& v : y) v = float(guard);
      for (auto& v : bins) v = 65535;
      for (auto& v : margins) v = guard;
      for (auto& v : losses) v = guard;
      for (auto& v : before) v = guard;
      for (auto& v : after) v = guard;
      for (auto& v : raw) v = guard;
      for (auto& v : decoded) v = guard;
      for (auto& v : dataset_bytes) v = std::byte{0xa5};
      for (auto& v : model_bytes) v = std::byte{0xa5};
      for (auto& v : prediction_bytes) v = std::byte{0xa5};
      for (u32 i = 0; i < 16; ++i) scratch[i] = scratch[scratch_bytes+16+i] = std::byte{0xa5};
      for (u32 i = 0; i < (id == 1 || id == 3 ? 80 : 64); ++i) dataset_bytes[i+1] = std::byte(legacy_datasets[id][i]);
      submitted(decode_dataset({dataset_bytes+1,id == 1 || id == 3 ? 80u : 64u},
        {x+1,6},{y+1,8},&record,&status));
      break;
    case 1:
      GH_CHECK(record.data.rows == (id == 3 ? 6 : 4) && record.data.columns == 1);
      GH_CHECK(record.data.outputs == (id == 1 ? 2 : 1));
      submitted(fit_schema(record.data,{},4,&fitted,{bins+1,6},workspace(),&status));
      break;
    case 2: {
      GH_CHECK(fitted.metadata_count == 1 && fitted.metadata.data[0] == -1.f);
      GH_CHECK(fitted.total_bins == 3 && fitted.features.data[0].type == FeatureType::numeric);
      for (u32 r = 0; r < record.data.rows; ++r) GH_CHECK(bins[r+1] == (r < record.data.rows/2 ? 1 : 2));
      TrainConfig config;
      config.objective = record.objective; config.classes = record.classes;
      config.rounds = id == 3 ? 0 : 1; config.max_depth = 1; config.min_leaf_rows = 1;
      config.l2 = 0; config.learning_rate = 1; config.histogram = Histogram::global;
      config.tree_build = TreeBuild::output_batch; config.output_tile = 2;
      training = {}; training.model = &original; training.margins = {margins+1,18}; training.loss = {losses+1,2};
      submitted(train(record.data,&fitted,{bins+1,6},config,&training,workspace(),&status));
      break;
    }
    case 3:
      GH_CHECK(original.outputs == (id == 3 ? 3 : id == 1 ? 2 : 1));
      check_prediction(id,margins+1,false);
      submitted(validate_model(&original,workspace(),&status));
      break;
    case 4: submitted(predict(&original,{bins+1,6},record.data.rows,{before+1,18},false,&status)); break;
    case 5:
      check_prediction(id,before+1,true);
      submitted(encode_model(&original,{model_bytes+1,1024},{&model_size,1},workspace(),&status));
      break;
    case 6:
      GH_CHECK(model_size == (id == 1 ? 248 : id == 3 ? 72 : 148));
      submitted(decode_model({model_bytes+1,model_size},&restored,workspace(),&status)); break;
    case 7: submitted(predict(&restored,{bins+1,6},record.data.rows,{raw+1,18},true,&status)); break;
    case 8:
      check_prediction(id,raw+1,false);
      for (u64 i = 0; i < cells; ++i) GH_CHECK(cuda::std::bit_cast<u64>(raw[i+1]) == cuda::std::bit_cast<u64>(margins[i+1]));
      submitted(predict(&restored,{bins+1,6},record.data.rows,{after+1,18},false,&status));
      break;
    case 9:
      check_prediction(id,after+1,true);
      for (u64 i = 0; i < cells; ++i) GH_CHECK(cuda::std::bit_cast<u64>(after[i+1]) == cuda::std::bit_cast<u64>(before[i+1]));
      submitted(encode_predictions({after+1,18},record.data.rows,restored.outputs,{prediction_bytes+1,144},&status));
      break;
    case 10:
      GH_CHECK(prior_required == cells*8);
      submitted(decode_predictions({prediction_bytes+1,cells*8},record.data.rows,restored.outputs,{decoded+1,18},&status));
      break;
    default:
      GH_CHECK(prior_required == cells*8);
      for (u64 i = 0; i < cells; ++i) GH_CHECK(cuda::std::bit_cast<u64>(decoded[i+1]) == cuda::std::bit_cast<u64>(before[i+1]));
      for (u32 i = 0; i < 146; ++i) if (!i || i > cells*8) GH_CHECK(prediction_bytes[i] == std::byte{0xa5});
      for (u32 i = 0; i < 1026; ++i) if (!i || i > model_size) GH_CHECK(model_bytes[i] == std::byte{0xa5});
      for (u32 i = 0; i < 82; ++i) GH_CHECK(dataset_bytes[i] ==
        (i && i <= (id == 1 || id == 3 ? 80 : 64) ? std::byte(legacy_datasets[id][i-1]) : std::byte{0xa5}));
      GH_CHECK(x[0] == float(guard) && x[7] == float(guard) && y[0] == float(guard) && y[9] == float(guard));
      GH_CHECK(bins[0] == 65535 && bins[7] == 65535 && losses[0] == guard && losses[3] == guard);
      for (u32 i = record.data.rows+1; i < 8; ++i) GH_CHECK(x[i] == float(guard) && bins[i] == 65535);
      for (u32 i = record.data.rows*record.data.outputs+1; i < 10; ++i) GH_CHECK(y[i] == float(guard));
      for (u32 r = 0; r < record.data.rows; ++r) {
        GH_CHECK(x[r+1] == (r < record.data.rows/2 ? -1.f : 1.f));
        for (u32 o = 0; o < record.data.outputs; ++o) {
          const float target = id == 3 ? float(r%3) : id == 2 ? float(r>=2) : float(analytical(id,r,o,false));
          GH_CHECK(y[1+r*record.data.outputs+o] == target);
        }
      }
      for (u32 i = 0; i < 20; ++i) if (!i || i > cells) {
        GH_CHECK(margins[i] == guard && before[i] == guard && after[i] == guard);
        GH_CHECK(raw[i] == guard && decoded[i] == guard);
      }
      for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0xa5} && scratch[scratch_bytes+16+i] == std::byte{0xa5});
      model_guards(original_storage); model_guards(restored_storage);
      if (id < 3) pipeline<<<1,1,0,cudaStreamTailLaunch>>>(id+1,0);
      else printf("PASS prediction: literal binary64 codec, malformed extents, finite bits and four resident dataset-to-prediction pipelines\n");
      submitted(cudaGetLastError()); return;
  }
  pipeline<<<1,1,0,cudaStreamTailLaunch>>>(id,phase+1);
  submitted(cudaGetLastError());
}
}
__global__ void run() { codec<<<1,1>>>(0); submitted(cudaGetLastError()); }
}
