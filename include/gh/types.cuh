#pragma once
#include "gh/core.cuh"

namespace gh {
enum class Objective : u32 { squared_error, binary_logistic, multiclass_softmax };
enum class FeatureType : u32 { numeric, categorical };
enum class RadixPolicy : u32 { radix8 = 8, radix4 = 4 };
struct Feature { u64 begin{}; u32 count{}; FeatureType type{}; };
struct Schema {
  Array<Feature> features;
  Array<float> metadata;
  Array<u32> offsets;
  u32 columns{}, total_bins{}, max_feature_bins{};
  u64 metadata_count{};
};
struct Dataset {
  Array<const float> values, targets, weights;
  u32 rows{}, columns{}, outputs{1};
};
struct Node {
  std::int32_t feature{-1}, left{-1}, right{-1};
  u32 threshold{}, missing_left{};
  double value{};
};
struct Tree { u64 begin{}; u32 count{}, output{}; };
struct Model {
  Schema schema;
  Array<Node> nodes;
  Array<Tree> trees;
  Array<double> base;
  Array<u64> output_offsets;
  u64 node_count{}, tree_count{};
  u32 outputs{1};
  Objective objective{};
};
static_assert(sizeof(Node) == 32 && offsetof(Node, value) == 24);
static_assert(sizeof(Feature) == 16 && sizeof(Tree) == 16);
}
