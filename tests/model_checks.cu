#include "gh/model.cuh"
#include "check.cuh"
#include <cuda/std/bit>

namespace gh::test {
namespace {
constexpr u32 rows = 65, model_cases = 34, wire_cases = 16;
constexpr double guard = -987654.25;
__device__ Model resident;
__device__ Feature features[4];
__device__ float metadata[8];
__device__ u32 offsets[5];
__device__ Node nodes[18];
__device__ Tree trees[10];
__device__ double base[35], output[rows * 33 + 2];
__device__ u64 output_offsets[36], written;
__device__ std::uint16_t bins[rows * 2 + 2];
__device__ __align__(16) std::byte scratch[16 + 4096 + 16];
__device__ std::byte input_bytes[1026], encoded[1026];
__device__ Status status;
__device__ const unsigned char literal[]{
  0x47,0x48,0x42,0x4d,0x4f,0x44,0x45,0x4c, 1,0,0,0, 0,0,0,0,
  1,0,0,0, 1,0,0,0, 1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0x80, 0,0,0,0, 0,0,0,0, 0,0,0,0,
  0,0,0,0, 1,0,0,0,
  255,255,255,255, 255,255,255,255, 255,255,255,255,
  0,0,0,0, 1,0,0,0, 0,0,0,0,0,0,0xf8,0x3f};
static_assert(sizeof(literal) == 88);
__device__ Workspace workspace() { return {scratch + 16, 4096}; }
__device__ u64 bits(double value) { return cuda::std::bit_cast<u64>(value); }
__device__ double number(u64 value) { return cuda::std::bit_cast<double>(value); }

__device__ void initialize() {
  status = {};
  for (auto& x : scratch) x = std::byte{0x5a};
  for (auto& x : output) x = guard;
  for (auto& x : base) x = guard;
  for (auto& x : nodes) x = {-1,-1,-1,0,0,guard};
  for (auto& x : trees) x = {123,456,789};
  for (auto& x : metadata) x = -12345;
  for (auto& x : bins) x = 65535;
  features[1] = {0,3,FeatureType::numeric}; features[2] = {3,3,FeatureType::categorical};
  metadata[1] = -3; metadata[2] = 0; metadata[3] = 2;
  metadata[4] = 2; metadata[5] = 5; metadata[6] = 8;
  offsets[1] = 0; offsets[2] = 5; offsets[3] = 9;
  base[1] = .25; base[2] = -.5; base[3] = -0.; base[4] = .25;
  // Physical node order differs from grouped descriptor order.
  nodes[1] = {0,1,2,2,1,0}; nodes[2].value = 1; nodes[3].value = -2;
  nodes[4].value = 0x1p53;
  nodes[5] = {1,1,2,2,0,0}; nodes[6].value = 2; nodes[7].value = -1;
  nodes[8].value = 1; nodes[9].value = .125; nodes[10].value = -0x1p53;
  trees[1] = {4,3,0}; trees[2] = {0,3,1}; trees[3] = {8,1,1};
  trees[4] = {3,1,3}; trees[5] = {7,1,3}; trees[6] = {9,1,3};
  output_offsets[1] = 0; output_offsets[2] = 1; output_offsets[3] = 3;
  output_offsets[4] = 3; output_offsets[5] = 6;
  for (u32 r = 0; r < rows; ++r) { bins[1 + r] = r % 5; bins[1 + rows + r] = r % 4; }
  resident = {{{features + 1,2},{metadata + 1,6},{offsets + 1,3},2,9,5,6},
              {nodes + 1,16},{trees + 1,8},{base + 1,33},{output_offsets + 1,34},10,6,4,Objective::squared_error};
}
__device__ void one_output(u32 count) {
  resident.outputs = 1; resident.node_count = count; resident.tree_count = count;
  output_offsets[1] = 0; output_offsets[2] = count;
  for (u32 i = 0; i < count; ++i) { trees[i + 1] = {i,1,0}; nodes[i + 1] = {-1,-1,-1,0,0,0}; }
}
__device__ void guards() {
  GH_CHECK(base[0] == guard && base[34] == guard && nodes[0].value == guard && nodes[17].value == guard);
  GH_CHECK(trees[0].begin == 123 && trees[9].begin == 123);
  GH_CHECK(metadata[0] == -12345 && metadata[7] == -12345);
  GH_CHECK(bins[0] == 65535 && bins[rows * 2 + 1] == 65535);
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[4112 + i] == std::byte{0x5a});
}
__device__ bool valid_case(u32 c) { return c == 0 || c == 21 || c == 22 || c == 24 || (c >= 28 && c <= 32); }
__global__ void model_case(u32 c);
__global__ void wire_case(u32 c);
__global__ void roundtrip_start();
__device__ void next_model(u32 c) {
  if (c + 1 < model_cases) model_case<<<1,1,0,cudaStreamTailLaunch>>>(c + 1);
  else wire_case<<<1,1,0,cudaStreamTailLaunch>>>(0);
  submitted(cudaGetLastError());
}
__global__ void prediction_checked(u32 c) {
  GH_CHECK(status.done == 1);
  guards();
  if (c >= 30) {
    if (c == 31) succeeded(status);
    else GH_CHECK(status.errors & (c == 30 ? input : capacity));
    for (auto x : output) GH_CHECK(x == guard);
  } else {
    succeeded(status);
    for (u32 r = 0; r < rows; ++r) for (u32 o = 0; o < resident.outputs; ++o) {
      const double expected = c == 28 ? (o == 0 ? .5 : o == 1 ? 1. : 0.) :
          c == 29 ? __ddiv_rn(1.,33.) : c != 0 ? number(0x7fefffffffffffffULL) :
          o == 0 ? (r % 4 == 2 ? 2.25 : -.75) : o == 1 ? (r % 5 <= 2 ? .625 : -2.375) : o == 2 ? -0. : 0.;
      GH_CHECK(bits(output[1 + u64(r) * resident.outputs + o]) == bits(expected));
    }
    GH_CHECK(output[0] == guard && output[1 + u64(rows) * resident.outputs] == guard);
  }
  next_model(c);
}
__global__ void validation_checked(u32 c) {
  GH_CHECK(status.done == 1);
  guards();
  if (!valid_case(c)) {
    GH_CHECK(status.errors & (c == 20 || c == 23 || c == 25 ? numeric : c == 26 || c == 27 || c == 33 ? capacity : model));
    if (c == 27) GH_CHECK(status.required_bytes == 136);
    next_model(c); return;
  }
  succeeded(status); status = {};
  if (c == 30) bins[4] = 65535;
  submitted(predict(&resident,{bins + 1,rows * 2},c == 31 ? 0 : rows,
                    {output + 1,c == 32 ? 0u : u64(rows) * resident.outputs},c < 28,&status));
  prediction_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void model_case(u32 c) {
  initialize();
  switch (c) {
    case 1: trees[2].begin = 4; break;
    case 2: resident.node_count = 11; break;
    case 3: nodes[1].right = 1; break;
    case 4: nodes[1].left = 0; break;
    case 5:
      one_output(1); resident.node_count = 3; trees[1].count = 3;
      nodes[2] = {0,1,2,1,0,0}; nodes[3] = {-1,-1,-1,0,0,0}; break;
    case 6: one_output(1); resident.node_count = 2; trees[1].count = 2; break;
    case 7: nodes[2].threshold = 1; break;
    case 8: nodes[2].missing_left = 2; break;
    case 9: nodes[1].feature = 2; break;
    case 10: nodes[1].threshold = 5; break;
    case 11: metadata[2] = -3; break;
    case 12: features[1].type = FeatureType(99); break;
    case 13: base[1] = number(0x7ff0000000000000ULL); break;
    case 14: nodes[2].value = number(0x7ff0000000000001ULL); break;
    case 15: trees[1].output = 3; break;
    case 16: output_offsets[5] = 5; break;
    case 17: offsets[3] = 8; break;
    case 18: features[2].begin = 4; break;
    case 19: resident.objective = Objective(99); break;
    case 20: base[1] = nodes[6].value = number(0x7fefffffffffffffULL); break;
    case 21: case 22: case 23: case 24: case 25:
      one_output(c == 24 ? 2 : 1); base[1] = number(0x7fefffffffffffffULL);
      nodes[1].value = c == 21 ? 1. : c == 23 ? 0x1.0000000000001p959 : c == 25 ? -base[1] : 0x1p959;
      if (c == 24) nodes[2].value = 0x1p959;
      break;
    case 26: resident.nodes.size = 9; break;
    case 28: case 29:
      resident.outputs = c == 28 ? 3 : 33; resident.tree_count = resident.node_count = 0;
      resident.objective = c == 28 ? Objective::binary_logistic : Objective::multiclass_softmax;
      for (u32 o = 0; o <= resident.outputs; ++o) output_offsets[o + 1] = 0;
      for (u32 o = 0; o < resident.outputs; ++o) base[o + 1] = c == 28 ? (o == 0 ? -0. : o == 1 ? 1000. : -1000.) : 0.;
      break;
  }
  submitted(validate_model(&resident,c == 27 ? Workspace{scratch + 16,135} :
                                     c == 33 ? Workspace{scratch + 17,4095} : workspace(),&status));
  validation_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}

__device__ unsigned char wire_byte(u32 c, u32 i) {
  if (c == 1 && i == 8) return 2;
  if (c == 2 && i == 12) return 99;
  if (c == 3 && i == 40) return 2;
  if (c == 4 && i == 48) return 1;
  if (c == 5 && i == 56) return 0;
  if (c == 6 && i == 72) return 1;
  if (c == 7 && i == 76) return 2;
  if (c == 8 && i == 86) return 0xf0;
  if (c == 8 && i == 87) return 0x7f;
  if (c == 13 && i >= 60 && i < 72) return 0;
  if (c == 14 && i == 38) return 0xf0;
  if (c == 14 && i == 39) return 0x7f;
  if (c == 15 && i == 16) return 0;
  return i < 88 ? literal[i] : 0;
}
__device__ void next_wire(u32 c) {
  if (c + 1 < wire_cases) wire_case<<<1,1,0,cudaStreamTailLaunch>>>(c + 1);
  else roundtrip_start<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
__global__ void encode_rejection(u32 c);
__global__ void encode_rejection_checked(u32 c) {
  GH_CHECK(status.done == 1 && (status.errors & capacity));
  GH_CHECK(written == 0xface);
  for (auto b : encoded) GH_CHECK(b == std::byte{0xa5});
  guards();
  if (c < 2) { encode_rejection<<<1,1,0,cudaStreamTailLaunch>>>(c + 1); submitted(cudaGetLastError()); }
  else next_wire(0);
}
__global__ void encode_rejection(u32 c) {
  status = {}; written = 0xface;
  for (auto& b : encoded) b = std::byte{0xa5};
  submitted(encode_model(&resident,{encoded + 1,c == 0 ? 87u : 1024u},{&written,c == 1 ? 0u : 1u},
                         c == 2 ? Workspace{scratch + 16,0} : workspace(),&status));
  encode_rejection_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void literal_encoded() {
  succeeded(status); GH_CHECK(written == 88);
  for (u32 i = 0; i < 88; ++i) GH_CHECK(encoded[i + 1] == std::byte(literal[i]));
  GH_CHECK(encoded[0] == std::byte{0xa5} && encoded[89] == std::byte{0xa5});
  encode_rejection<<<1,1,0,cudaStreamTailLaunch>>>(0); submitted(cudaGetLastError());
}
__global__ void wire_checked(u32 c) {
  GH_CHECK(status.done == 1); guards();
  for (u32 i = 0; i < 89; ++i) GH_CHECK(input_bytes[i + 1] == std::byte(wire_byte(c,i)));
  GH_CHECK(input_bytes[0] == std::byte{0xa5} && input_bytes[90] == std::byte{0xa5});
  if (c) { GH_CHECK(status.errors & (c == 11 || c == 12 ? capacity : model)); next_wire(c); return; }
  succeeded(status);
  GH_CHECK(resident.outputs == 1 && resident.tree_count == 1 && resident.node_count == 1);
  GH_CHECK(bits(base[1]) == 0x8000000000000000ULL && nodes[1].feature == -1 && nodes[1].missing_left == 1 && nodes[1].value == 1.5);
  GH_CHECK(resident.schema.total_bins == 2 && features[1].count == 0 && output_offsets[2] == 1);
  status = {};
  for (auto& b : encoded) b = std::byte{0xa5};
  submitted(encode_model(&resident,{encoded + 1,1024},{&written,1},workspace(),&status));
  literal_encoded<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void wire_case(u32 c) {
  initialize();
  for (auto& b : input_bytes) b = std::byte{0xa5};
  for (u32 i = 0; i < 89; ++i) input_bytes[i + 1] = std::byte(wire_byte(c,i));
  if (c == 11) resident.nodes.size = 0;
  submitted(decode_model({input_bytes + 1,c == 9 ? 87u : c == 10 ? 89u : 88u},&resident,
                          c == 12 ? Workspace{scratch + 16,0} : workspace(),&status));
  wire_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void roundtrip_prediction() {
  succeeded(status); guards();
  for (u32 r = 0; r < rows; ++r) for (u32 o = 0; o < 4; ++o) {
    const double expected = o == 0 ? (r % 4 == 2 ? 2.25 : -.75) : o == 1 ? (r % 5 <= 2 ? .625 : -2.375) : o == 2 ? -0. : 0.;
    GH_CHECK(bits(output[1 + r * 4 + o]) == bits(expected));
  }
  GH_CHECK(output[0] == guard && output[rows * 4 + 1] == guard);
  printf("model checks passed: %u model/prediction cases, %u wire cases, literal and permuted-forest roundtrip\n",model_cases,wire_cases);
}
__global__ void roundtrip_decoded() {
  succeeded(status); GH_CHECK(resident.node_count == 10 && resident.tree_count == 6);
  GH_CHECK(trees[1].begin == 0 && trees[1].output == 0 && trees[2].begin == 3 && trees[2].output == 1);
  GH_CHECK(nodes[1].feature == 1 && nodes[4].feature == 0 && bits(base[3]) == 0x8000000000000000ULL);
  status = {};
  submitted(predict(&resident,{bins + 1,rows * 2},rows,{output + 1,rows * 4},true,&status));
  roundtrip_prediction<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void roundtrip_encoded() {
  succeeded(status); GH_CHECK(written == 440);
  GH_CHECK(encoded[0] == std::byte{0xa5} && encoded[441] == std::byte{0xa5});
  initialize();
  submitted(decode_model({encoded + 1,written},&resident,workspace(),&status));
  roundtrip_decoded<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void roundtrip_validated() {
  succeeded(status); status = {};
  for (auto& b : encoded) b = std::byte{0xa5};
  submitted(encode_model(&resident,{encoded + 1,1024},{&written,1},workspace(),&status));
  roundtrip_encoded<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void roundtrip_start() {
  initialize(); submitted(validate_model(&resident,workspace(),&status));
  roundtrip_validated<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
}
__global__ void run() { model_case<<<1,1,0,cudaStreamTailLaunch>>>(0); submitted(cudaGetLastError()); }
}
