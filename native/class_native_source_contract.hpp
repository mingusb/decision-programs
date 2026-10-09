#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <stdexcept>

// Preconditions shared by structural readers of the reviewed scalar numeric
// XGBoost 3.4.1 predictor. These checks establish imported-model restrictions;
// they do not alone authorize native-class pruning.
namespace dp_native_source_contract {
inline void require_unit_tree_weights(const nlohmann::json& model,
                                      std::size_t trees) {
    if (!model.contains("weight_drop")) return;
    const auto& weights = model.at("weight_drop");
    if (!weights.is_array())
        throw std::runtime_error("source weight_drop must be an array");
    // Native OptionalWeights supplies 1.0f when storage is absent or empty.
    if (weights.empty()) return;
    if (weights.size() != trees)
        throw std::runtime_error("source tree weight count differs");
    for (const auto& weight : weights)
        if (!weight.is_number() || weight.get<double>() != 1.0)
            throw std::runtime_error("non-unit source tree weights are unsupported");
}
inline void require_numeric_successors(std::int32_t node, std::int32_t left,
                                       std::int32_t right) {
    // Native GetNextNode uses LeftChild + !decision for a numeric split.
    // Native traversal also requires forward indices. A graph-valid
    // permutation need not have the same native routing/traversal behavior.
    if (left <= node || std::int64_t(right) != std::int64_t(left) + 1)
        throw std::runtime_error("numeric source successors must advance and be consecutive in native order");
}
} // namespace dp_native_source_contract
