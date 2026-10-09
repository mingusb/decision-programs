// CPU-only JSON/parser and serialization checks. No numerical inference or fit.
#include "class_io.hpp"
#include "class_conversion/class_io.hpp"

#include <functional>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: class_io_checks SOURCE.json NEW_DIRECTORY");
        const dpnative::fs::path directory = argv[2];
        if (!dpnative::fs::create_directories(directory)) throw std::runtime_error("check directory already exists");
        const auto original = dpnative::read_source(argv[1]);
        auto fixture = dpnative::json::parse(original.bytes);
        auto& model = fixture["learner"]["gradient_booster"]["model"];
        model["trees"] = dpnative::json::array({model["trees"].at(0)});
        model["tree_info"] = dpnative::json::array({model["tree_info"].at(0)});
        model["iteration_indptr"] = dpnative::json::array({0, 1});
        model["gbtree_model_param"]["num_trees"] = "1";
        dpnative::atomic_json(directory / "valid-prefix.json", fixture);
        const auto valid = dpnative::read_source(directory / "valid-prefix.json");
        if (valid.roots.size() != 1 || valid.features != original.features || valid.outputs != original.outputs)
            throw std::runtime_error("valid prefix metadata changed");
        using J = dpnative::json;
        auto tree = [](J& value) -> J& { return value["learner"]["gradient_booster"]["model"]["trees"][0]; };
        auto ensemble = [](J& value) -> J& { return value["learner"]["gradient_booster"]["model"]; };
        auto accept_both = [&](const J& candidate) {
            const auto path = directory / "valid-weight-case.json";
            dpnative::atomic_json(path, candidate);
            const auto host = dpnative::read_source(path);
            const auto adaptive = class_conversion_native::read_source_bytes(candidate.dump());
            if (host.value != valid.value || adaptive.value != valid.value ||
                host.left != valid.left || adaptive.left != valid.left ||
                host.right != valid.right || adaptive.right != valid.right)
                throw std::runtime_error("unit-weight import changed model semantics");
        };
        accept_both(fixture);
        auto empty_weights = fixture; ensemble(empty_weights)["weight_drop"] = J::array();
        accept_both(empty_weights);
        auto unit_weights = fixture; ensemble(unit_weights)["weight_drop"] = J::array({1.0});
        accept_both(unit_weights);
        const std::vector<std::pair<std::string, std::function<void(J&)>>> cases{
            {"non-unit-weight", [&](J& j) { ensemble(j)["weight_drop"] = J::array({0.5}); }},
            {"weight-count", [&](J& j) { ensemble(j)["weight_drop"] = J::array({1.0, 1.0}); }},
            {"weight-object", [&](J& j) { ensemble(j)["weight_drop"] = J::object(); }},
            {"weight-string", [&](J& j) { ensemble(j)["weight_drop"] = J::array({"1"}); }},
            {"weight-null", [&](J& j) { ensemble(j)["weight_drop"] = J::array({nullptr}); }},
            {"permuted-successors", [&](J& j) {
                auto& t = tree(j); const auto left = t["left_children"][0];
                t["left_children"][0] = t["right_children"][0]; t["right_children"][0] = left;
            }},
            {"backward-successors", [&](J& j) {
                auto& t = tree(j); t["tree_param"]["num_nodes"] = "5";
                // Reachable and acyclic, with adjacent successors at every
                // split, but split 3 points backward to 1 and 2.
                t["left_children"] = J::array({3,-1,-1,1,-1});
                t["right_children"] = J::array({4,-1,-1,2,-1});
                t["split_indices"] = J::array({0,0,0,0,0});
                t["split_conditions"] = J::array({0.5,-1.0,1.0,0.25,2.0});
                t["split_type"] = J::array({0,0,0,0,0});
                t["default_left"] = J::array({true,false,false,true,false});
            }},
            {"wrong-version", [](J& j) { j["version"] = J::array({3, 4, 0}); }},
            {"fractional-version", [](J& j) { j["version"][0] = 3.0; }},
            {"decimal-suffix", [](J& j) { j["learner"]["learner_model_param"]["num_feature"] = "54junk"; }},
            {"fractional-channel", [&](J& j) { ensemble(j)["tree_info"][0] = 0.5; }},
            {"wrong-tree-id", [&](J& j) { tree(j)["id"] = 2; }},
            {"wrong-node-count", [&](J& j) { tree(j)["tree_param"]["num_nodes"] = "1"; }},
            {"wrong-tree-count", [&](J& j) { ensemble(j)["gbtree_model_param"]["num_trees"] = "2"; }},
            {"wrong-iteration-start", [&](J& j) { ensemble(j)["iteration_indptr"] = J::array({1, 1}); }},
            {"wrong-iteration-end", [&](J& j) { ensemble(j)["iteration_indptr"] = J::array({0, 0}); }},
            {"unordered-iterations", [&](J& j) { ensemble(j)["iteration_indptr"] = J::array({0, 2, 1}); }},
            {"invalid-default", [&](J& j) { tree(j)["default_left"][0] = 2; }},
            {"fractional-default", [&](J& j) { tree(j)["default_left"][0] = 1.0; }},
            {"fractional-feature", [&](J& j) { tree(j)["split_indices"][0] = 0.5; }},
            {"wrong-split-array", [&](J& j) { tree(j)["split_type"] = J::array(); }},
            {"shared-successor", [&](J& j) { tree(j)["right_children"][0] = tree(j)["left_children"][0]; }},
            {"cycle", [&](J& j) { tree(j)["left_children"][0] = 0; }}
        };
        std::vector<std::string> rejected;
        for (const auto& [name, modify] : cases) {
            auto candidate = fixture;
            modify(candidate);
            const auto path = directory / (name + ".json");
            dpnative::atomic_json(path, candidate);
            bool failed = false;
            try { (void)dpnative::read_source(path); }
            catch (const std::exception&) { failed = true; }
            if (!failed) throw std::runtime_error("reader accepted malformed source: " + name);
            failed = false;
            try { (void)class_conversion_native::read_source_bytes(candidate.dump()); }
            catch (const std::exception&) { failed = true; }
            if (!failed) throw std::runtime_error("adaptive reader accepted malformed source: " + name);
            rejected.push_back(name);
        }
        J result{{"complete", true}, {"cpu_scope", "JSON parsing and serialization only"},
                 {"python_runtime", false}, {"source_sha256", original.identity},
                 {"source_trees", original.roots.size()}, {"valid_prefix_trees", valid.roots.size()},
                 {"malformed_cases_rejected", rejected}, {"both_structural_readers_checked", true}, {"valid_weight_variants", 3}, {"count", rejected.size() * 2 + 6}};
        dpnative::atomic_json(directory / "result.json", result);
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native reader checks failed: " << error.what() << '\n';
        return 1;
    }
}
