#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h" // graph-view metadata, matching this source tree
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * gdn = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<ggml_tensor *> inputs;
    int split = 0;
    size_t tail_offset = 0;
    size_t tail_elements = 0;
    ~fixture() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

// Equivalent to ggml_graph_view: retain the FULL graph hash set/use counts.
// The internal helper is not exported by the Windows DLL.
static ggml_cgraph graph_slice(ggml_cgraph * full, int begin, int end) {
    ggml_cgraph view = *full;
    view.size = 0;
    view.nodes = full->nodes + begin;
    view.n_nodes = end - begin;
    view.n_leafs = 0;
    view.leafs = nullptr;
    view.grads = nullptr;
    view.grad_accs = nullptr;
    view.uid = 0;
    return view;
}

static std::vector<float> run_case(ggml_backend_t backend, const std::string & name,
                                 int tokens, int slots, json & details, int repeats) {
    constexpr int H = 4, S = 32, sequences = 1;
    const int D = H * S * S;
    const int written = std::min(tokens, slots);
    fixture f;
    f.ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!f.ctx) throw std::runtime_error("ggml_init failed");
    auto input = [&](const char * label, int64_t a, int64_t b, int64_t c, int64_t d) {
        auto * t = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F32, a, b, c, d);
        ggml_set_name(t, label);
        f.inputs.push_back(t);
        return t;
    };
    auto * q = input("q", S, H, tokens, sequences);
    auto * k = input("k", S, H, tokens, sequences);
    auto * v = input("v", S, H, tokens, sequences);
    auto * g = input("g", 1, H, tokens, sequences);
    auto * beta = input("beta", 1, H, tokens, sequences);
    auto * state = input("state", S, S, H, sequences);
    q = ggml_l2_norm(f.ctx, q, 1e-6f);
    k = ggml_l2_norm(f.ctx, k, 1e-6f);
    f.gdn = ggml_gated_delta_net(f.ctx, q, k, v, g, beta, state, slots);
    ggml_set_name(f.gdn, "gdn_output_unmarked");
    f.tail_offset = size_t(S * H * tokens * sequences) * sizeof(float);
    f.tail_elements = size_t(D * sequences * written);
    auto * tail = ggml_view_3d(f.ctx, f.gdn, D, sequences, written,
                              D * sizeof(float), D * sequences * sizeof(float), f.tail_offset);
    ggml_set_name(tail, "original_snapshot_tail");
    auto * cache = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, D, sequences, written);
    auto * destination = ggml_view_3d(f.ctx, cache, D, sequences, written,
                                     D * sizeof(float), D * sequences * sizeof(float), 0);
    auto * copied = ggml_cpy(f.ctx, tail, destination);
    ggml_set_name(copied, "persistent_cache_copy_unmarked");
    f.graph = ggml_new_graph_custom(f.ctx, 128, false);
    ggml_build_forward_expand(f.graph, copied);
    f.split = f.graph->n_nodes;
    auto * flat_copy = ggml_reshape_1d(f.ctx, copied, int64_t(f.tail_elements));
    if (name == "copied_cache") {
        f.out = ggml_scale(f.ctx, flat_copy, 1.0f);
    } else {
        ggml_tensor * live = tail;
        if (name == "nested_tail" || name == "cross_split_tail") {
            live = ggml_reshape_1d(f.ctx, live, int64_t(f.tail_elements));
            live = ggml_view_1d(f.ctx, live, int64_t(f.tail_elements), 0);
            live = ggml_reshape_3d(f.ctx, live, D, sequences, written);
            ggml_set_name(live, "nested_live_snapshot_alias");
        }
        auto * flat_live = ggml_reshape_1d(f.ctx, live, int64_t(f.tail_elements));
        f.out = ggml_concat(f.ctx, flat_copy, flat_live, 0);
    }
    ggml_set_name(f.out, "checked_output");
    ggml_set_output(f.out);
    ggml_build_forward_expand(f.graph, f.out);
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, backend);
    if (!f.buffer) throw std::runtime_error("backend allocation failed");
    ggml_backend_buffer_clear(f.buffer, 0);
    for (size_t which = 0; which < f.inputs.size(); ++which) {
        auto * tensor = f.inputs[which];
        std::vector<float> values(size_t(ggml_nelements(tensor)));
        for (size_t j = 0; j < values.size(); ++j) {
            const float wave = float(int((j * 17 + which * 11) % 29) - 14) / 29.0f;
            values[j] = which == 3 ? -0.4f + 0.1f * wave :
                        which == 4 ? 0.45f + 0.1f * wave :
                        which == 5 ? 0.02f + 0.02f * wave : 0.1f + 0.2f * wave;
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    // Make tail-elision failure deterministic. The correct path overwrites every
    // written snapshot element. The positive control never reads this tail.
    std::vector<float> poison(f.tail_elements, std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(f.gdn, poison.data(), f.tail_offset, poison.size() * sizeof(float));
    auto compute = [&](ggml_cgraph * graph) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph compute failed");
        ggml_backend_synchronize(backend);
    };
    for (int repeat = 0; repeat < repeats; ++repeat) {
    ggml_backend_tensor_set(f.gdn, poison.data(), f.tail_offset, poison.size() * sizeof(float));
    if (name == "cross_split_tail") {
        auto first = graph_slice(f.graph, 0, f.split);
        auto second = graph_slice(f.graph, f.split, f.graph->n_nodes);
        const bool shared = first.visited_hash_set.keys == f.graph->visited_hash_set.keys &&
                            first.use_counts == f.graph->use_counts;
        if (!shared) throw std::runtime_error("graph-view metadata is not shared");
        details["shared_full_graph_metadata"] = shared;
        details["first_view_nodes"] = first.n_nodes;
        details["later_view_nodes"] = second.n_nodes;
        compute(&first);
        compute(&second);
    } else {
        compute(f.graph);
    }
    }
    details["repeats"] = repeats;
    details["graph_nodes"] = f.graph->n_nodes;
    details["tail_elements"] = f.tail_elements;
    std::vector<float> result(size_t(ggml_nelements(f.out)));
    ggml_backend_tensor_get(f.out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

int main(int argc, char ** argv) {
    ggml_backend_t cpu = nullptr, target = nullptr;
    try {
        std::string backend_name = "CUDA0", fusion = "on", graphs = "off";
        std::string directory;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("each option requires a value");
            const std::string value = argv[++i];
            if (arg == "--backend") backend_name = value;
            else if (arg == "--backend-dir") directory = value;
            else if (arg == "--fusion") fusion = value;
            else if (arg == "--graphs") graphs = value;
            else throw std::runtime_error("unknown option: " + arg);
        }
        if ((fusion != "on" && fusion != "off") || (graphs != "on" && graphs != "off"))
            throw std::runtime_error("fusion/graphs must be on or off");
        // Only these two variables change. PATH is never assigned or overridden.
        _putenv_s("GGML_CUDA_DISABLE_FUSION", fusion == "off" ? "1" : "0");
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS", graphs == "off" ? "1" : "");
        // This distribution registers its linked CPU/CUDA backends statically.
        // Explicit dynamic loading is optional, for other build configurations.
        if (!directory.empty()) {
            ggml_backend_load_all_from_path(directory.c_str());
        }
        cpu = ggml_backend_init_by_name("CPU", nullptr);
        target = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
        if (!cpu || !target) throw std::runtime_error("requested backend unavailable");
        bool all_pass = true;
        for (const auto & name : {"copied_cache", "live_tail", "nested_tail", "cross_split_tail"}) {
            for (const auto & dims : {std::pair<int,int>{1,1}, std::pair<int,int>{2,2}}) {
                json details, cpu_details;
                const int repeats = graphs == "on" ? 5 : 1;
                const auto expected = run_case(cpu, name, dims.first, dims.second, cpu_details, repeats);
                const auto actual = run_case(target, name, dims.first, dims.second, details, repeats);
                if (actual.size() != expected.size()) throw std::runtime_error("output size mismatch");
                double max_abs = 0, sum_error = 0, sum_expected = 0, max_pair = 0;
                size_t nonfinite = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(expected[i])) throw std::runtime_error("nonfinite CPU oracle");
                    if (!std::isfinite(actual[i])) { ++nonfinite; continue; }
                    const double difference = double(actual[i]) - expected[i];
                    max_abs = std::max(max_abs, std::abs(difference));
                    sum_error += difference * difference;
                    sum_expected += double(expected[i]) * expected[i];
                }
                if (std::string(name) != "copied_cache" && nonfinite == 0) {
                    for (size_t i = 0; i < actual.size() / 2; ++i)
                        max_pair = std::max(max_pair, std::abs(double(actual[i]) - actual[i + actual.size()/2]));
                }
                const double nmse = sum_error / std::max(sum_expected, 1e-30);
                const bool pass = nonfinite == 0 && max_abs <= 5e-4 && nmse <= 1e-6 && max_pair == 0;
                all_pass = all_pass && pass;
                json row = {{"case",name},{"backend",backend_name},{"fusion",fusion},{"graphs",graphs},
                            {"tokens",dims.first},{"slots",dims.second},{"pass",pass},
                            {"nonfinite",nonfinite},{"max_abs_error",max_abs},{"nmse",nmse},
                            {"cache_tail_max_abs",max_pair},{"details",details}};
                std::cout << row.dump() << std::endl;
            }
        }
        ggml_backend_free(target); target = nullptr;
        ggml_backend_free(cpu); cpu = nullptr;
        std::cout << json({{"summary",true},{"pass",all_pass},{"cases",8}}).dump() << std::endl;
        return all_pass ? 0 : 1;
    } catch (const std::exception & e) {
        if (target) ggml_backend_free(target);
        if (cpu) ggml_backend_free(cpu);
        std::cout << json({{"error",e.what()}}).dump() << std::endl;
        return 2;
    }
}
