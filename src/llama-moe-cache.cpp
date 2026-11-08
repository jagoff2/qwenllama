#include "llama-moe-cache.h"
#include "llama-moe-stream.h"
#include "llama-moe-stream-plan.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t cache_alignment = 512;
constexpr size_t cache_scratch_bytes = size_t(32) << 20;
constexpr size_t cache_quant_bytes = size_t(1) << 20;

size_t cache_align(size_t bytes) {
    return (bytes + cache_alignment - 1) & ~(cache_alignment - 1);
}

bool cache_quant_supported(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0;
}

bool cache_cpu_source_supported(ggml_backend_dev_t cpu, ggml_tensor * source) {
    if (!cpu) {
        return false;
    }
    ggml_init_params params{};
    params.mem_size = 4 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (!context) {
        return false;
    }
    auto * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, source->ne[0], 1, 1);
    auto * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, 10, 1);
    auto * map = ggml_new_tensor_2d(context, GGML_TYPE_I32, 1, source->ne[2]);
    auto * op = ggml_mul_mat_id_cpu_masked(context, source, input, ids, map);
    const bool supported = ggml_backend_dev_supports_op(cpu, op);
    ggml_free(context);
    return supported;
}

bool cache_gpu_source_supported(ggml_backend_dev_t device, ggml_tensor * source, int64_t input_channels) {
    if (!device) {
        return false;
    }
    // Metadata only: keep the authoritative host weight descriptor and probe
    // the actual arena device's forced MMVQ path at the largest cache tile.
    ggml_init_params params{};
    params.mem_size = 3 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (!context) {
        return false;
    }
    auto * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, source->ne[0], input_channels, 8);
    auto * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, 10, 8);
    auto * op = ggml_mul_mat_id(context, source, input, ids);
    ggml_set_name(op, "fn_cache_gpu_capability_probe");
    const bool supported = ggml_backend_dev_supports_op(device, op);
    ggml_free(context);
    return supported;
}

struct cache_entry {
    int expert = -1;
    uint64_t pending = 0;
    uint64_t last_use = 0;
};

struct cache_layer_state {
    llama_moe_cache_layer_bind bind;
    ggml_tensor * source[3] = { nullptr, nullptr, nullptr };
    std::vector<int32_t> map;
    std::vector<uint16_t> score;
    std::vector<cache_entry> entries;
    std::deque<std::vector<int32_t>> prompt;
    std::vector<bool> seen;
    size_t triplet_bytes = 0;
};

struct cache_scratch_span {
    ggml_backend_buffer_t buffer = nullptr;
    char * base = nullptr;
    size_t bytes = 0;
    int device = -1;
};
}

struct llama_moe_cache {
    llama_moe_stream * stream = nullptr;
    llama_moe_cache_params params;
    ggml_context * context = nullptr;
    std::vector<cache_layer_state> layers;
    std::vector<cache_scratch_span> scratch;
    std::map<int, void *> quant_scratch;
    bool cache_mode = false;
    bool active = false;
    bool failed = false;
    bool collecting_prompt = false;
    uint64_t generation = 0;
    uint64_t next_tag = 1;
    uint64_t tick = 0;
    uint64_t decay_rows = 0;
    uint64_t hit_rows = 0;
    uint64_t miss_rows = 0;
    uint64_t baseline_rows = 0;
    uint64_t admission_bytes = 0;
};

namespace {
[[noreturn]] void cache_failure(llama_moe_cache & cache, const char * message) {
    cache.failed = true;
    cache.active = false;
    ++cache.generation;
    throw std::runtime_error(message);
}

void cache_upload_maps(llama_moe_cache & cache) {
    for (auto & layer : cache.layers) {
        if (layer.bind.slot_map) {
            ggml_backend_tensor_set(layer.bind.slot_map, layer.map.data(), 0, layer.map.size() * sizeof(int32_t));
        }
    }
}

void cache_poll(llama_moe_cache & cache, bool wait) {
    for (auto & layer : cache.layers) {
        for (size_t slot = 1; slot < layer.entries.size(); ++slot) {
            auto & entry = layer.entries[slot];
            if (!entry.pending) {
                continue;
            }
            const int state = llama_moe_stream_cache_query(cache.stream, layer.bind.device, entry.pending, wait);
            if (state < 0) {
                cache_failure(cache, "MoE cache expert copy failed");
            }
            if (state > 0) {
                entry.pending = 0;
                layer.map[entry.expert] = int32_t(slot);
            }
        }
    }
}

size_t cache_pending(const llama_moe_cache & cache) {
    size_t count = 0;
    for (const auto & layer : cache.layers) {
        for (const auto & entry : layer.entries) {
            count += entry.pending != 0;
        }
    }
    return count;
}

void cache_invalidate(llama_moe_cache & cache) {
    for (auto & layer : cache.layers) {
        std::fill(layer.map.begin(), layer.map.end(), 0);
        std::fill(layer.entries.begin(), layer.entries.end(), cache_entry{});
        std::fill(layer.score.begin(), layer.score.end(), 0);
        std::fill(layer.seen.begin(), layer.seen.end(), false);
    }
    cache.decay_rows = 0;
}

void cache_zero_dummy(llama_moe_cache & cache) {
    size_t largest = 0;
    for (const auto & layer : cache.layers) {
        for (const auto * tensor : layer.source) {
            if (tensor) {
                largest = std::max(largest, tensor->nb[2]);
            }
        }
    }
    std::vector<uint8_t> zero(largest, 0);
    for (auto & layer : cache.layers) {
        for (int f = 0; f < 3; ++f) {
            auto * bank = layer.bind.bank[f];
            if (!bank) {
                continue;
            }
            ggml_backend_tensor_set(bank, zero.data(), 0, bank->nb[2]);
            const auto buft = ggml_backend_buffer_get_type(bank->buffer);
            const size_t pad = ggml_backend_buft_get_alloc_size(buft, bank) - ggml_nbytes(bank);
            if (pad) {
                // The backend owns the allocation tail; tensor_set cannot address it.
                ggml_cuda_moe_cache_copy clear{};
                clear.tag = cache.next_tag++;
                clear.dst[0] = static_cast<char *>(bank->data) + ggml_nbytes(bank);
                clear.nbytes[0] = pad;
                if (!llama_moe_stream_cache_submit(cache.stream, layer.bind.device, clear) ||
                    llama_moe_stream_cache_query(cache.stream, layer.bind.device, clear.tag, true) != 1) {
                    cache_failure(cache, "MoE cache padding clear failed");
                }
            }
        }
    }
}

bool cache_submit(llama_moe_cache & cache, cache_layer_state & layer, int slot, int expert) {
    if (cache_pending(cache) >= 63) {
        return false;
    }
    ggml_cuda_moe_cache_copy copy{};
    copy.tag = cache.next_tag++;
    for (int f = 0; f < 3; ++f) {
        copy.host[f] = static_cast<const char *>(layer.source[f]->data) + size_t(expert) * layer.source[f]->nb[2];
        copy.dst[f] = static_cast<char *>(layer.bind.bank[f]->data) + size_t(slot) * layer.bind.bank[f]->nb[2];
        copy.nbytes[f] = layer.source[f]->nb[2];
    }
    if (!llama_moe_stream_cache_submit(cache.stream, layer.bind.device, copy)) {
        cache_failure(cache, "MoE cache admission submission failed");
    }
    auto & entry = layer.entries[slot];
    if (entry.expert >= 0) {
        layer.map[entry.expert] = 0;
    }
    entry.expert = expert;
    entry.pending = copy.tag;
    entry.last_use = cache.tick;
    cache.admission_bytes += layer.triplet_bytes;
    return true;
}

void cache_seed(llama_moe_cache & cache) {
    const uint64_t copied_before = cache.admission_bytes;
    for (auto & layer : cache.layers) {
        if (!layer.bind.slot_map) {
            continue;
        }
        for (const auto & row : layer.prompt) {
            for (int expert : row) {
                if (layer.score[expert] < std::numeric_limits<uint16_t>::max()) {
                    ++layer.score[expert];
                }
            }
        }
        std::vector<int> ranked(layer.map.size());
        std::iota(ranked.begin(), ranked.end(), 0);
        std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) { return layer.score[a] > layer.score[b]; });
        for (int slot = 1; slot <= layer.bind.capacity && layer.score[ranked[slot - 1]]; ++slot) {
            if (cache_pending(cache) >= 63) {
                cache_poll(cache, true);
            }
            if (!cache_submit(cache, layer, slot, ranked[slot - 1])) {
                cache_failure(cache, "MoE cache seed queue full");
            }
        }
    }
    cache_poll(cache, true);
    LLAMA_LOG_INFO("moe_cache: prompt seed ready, copied %.1f MiB\n", (cache.admission_bytes - copied_before) / 1048576.0);
}
}

llama_moe_cache * llama_moe_cache_create(const llama_model & model, llama_moe_stream * stream,
                                        const llama_moe_cache_params & params) {
    const char * multi_env = std::getenv("LLAMA_MOE_CACHE_MULTI_GPU");
    const bool multi = params.multi_device || (multi_env && std::strcmp(multi_env, "1") == 0);
    if (!params.enable || model.arch != LLM_ARCH_QWEN4EXP || !stream || !stream->enabled || stream->devices.empty() ||
        (!multi && stream->devices.size() != 1) ||
        stream->extents.empty() || params.slots_per_layer < 1 || params.slots_per_layer > 511 ||
        params.warm_rows < 1 || params.warm_rows > 256 || params.max_rows < 1 || params.max_rows > 49) {
        return nullptr;
    }
    auto * cache = new llama_moe_cache;
    cache->stream = stream;
    cache->params = params;
    cache->layers.resize(model.layers.size());
    std::map<int, std::vector<llama_moe_plan::cache_layer>> sources;
    std::map<int, std::vector<uint64_t>> extents;
    std::map<int, std::vector<size_t>> extent_indices;
    std::map<ggml_backend_buffer_t, int> buffer_devices;
    for (size_t i = 0; i < stream->extents.size(); ++i) {
        const auto & extent = stream->extents[i];
        if (!extent.base || !extent.buffer || extent.usable_bytes <= cache_scratch_bytes ||
            std::find(stream->devices.begin(), stream->devices.end(), extent.device) == stream->devices.end()) {
            delete cache;
            return nullptr;
        }
        extents[extent.device].push_back(extent.usable_bytes);
        extent_indices[extent.device].push_back(i);
        buffer_devices[extent.buffer] = extent.device;
    }
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    for (size_t il = 0; il < model.layers.size(); ++il) {
        llama_moe_stream_layer_bind streamed;
        if (!llama_moe_stream_bind(stream, int(il), streamed)) {
            continue;
        }
        const auto owner = buffer_devices.find(streamed.arena[0]->buffer);
        if (owner == buffer_devices.end()) {
            delete cache;
            return nullptr;
        }
        const auto buft = ggml_backend_buffer_get_type(streamed.arena[0]->buffer);
        const auto device = ggml_backend_buft_get_device(buft);
        auto & layer = cache->layers[il];
        layer.source[0] = model.layers[il].ffn_up_exps;
        layer.source[1] = model.layers[il].ffn_gate_exps;
        layer.source[2] = model.layers[il].ffn_down_exps;
        llama_moe_plan::cache_layer source;
        source.il = int(il);
        int n_expert = 0;
        bool supported = model.hparams.n_expert_used(uint32_t(il)) == 10;
        for (int f = 0; f < 3; ++f) {
            const auto * tensor = layer.source[f];
            supported = supported && llama_moe_stream_host_source(tensor) && ggml_is_contiguous(tensor) &&
                        tensor->ne[3] == 1 && tensor->ne[2] == 512 && cache_quant_supported(tensor->type) &&
                        tensor->ne[0] == (f == 2 ? 640 : 2560) && tensor->ne[1] == (f == 2 ? 2560 : 640);
            supported = supported && cache_cpu_source_supported(cpu, layer.source[f]);
            supported = supported && cache_gpu_source_supported(device, layer.source[f], f == 2 ? 10 : 1);
            if (!supported) {
                break;
            }
            n_expert = int(tensor->ne[2]);
            source.expert_bytes[f] = tensor->nb[2];
            source.tail_bytes[f] = ggml_backend_buft_get_alloc_size(buft, tensor) - ggml_nbytes(tensor);
            layer.triplet_bytes += tensor->nb[2];
        }
        if (!supported || n_expert <= params.slots_per_layer) {
            LLAMA_LOG_WARN("moe_cache: unsupported expert layout or backend capability at layer %zu; using baseline\n", il);
            delete cache;
            return nullptr;
        }
        layer.map.assign(n_expert, 0);
        layer.score.assign(n_expert, 0);
        layer.seen.assign(n_expert, false);
        layer.entries.resize(params.slots_per_layer + 1);
        layer.bind.host_slots = layer.map.data();
        layer.bind.n_expert = n_expert;
        layer.bind.capacity = params.slots_per_layer;
        layer.bind.device = owner->second;
        sources[owner->second].push_back(source);
    }
    if (sources.empty()) {
        delete cache;
        return nullptr;
    }
    llama_moe_plan::cache_plan plan;
    for (const auto & item : sources) {
        auto local = llama_moe_plan::plan_cache(item.second, extents.at(item.first), params.slots_per_layer, cache_scratch_bytes);
        if (!local.enabled) {
            LLAMA_LOG_WARN("moe_cache: device %d: %s; using baseline\n", item.first, local.reason.c_str());
            delete cache;
            return nullptr;
        }
        for (auto bank : local.banks) {
            bank.extent = int(extent_indices.at(item.first).at(bank.extent));
            plan.banks.push_back(bank);
        }
        plan.bank_bytes += local.bank_bytes;
        LLAMA_LOG_INFO("moe_cache: device %d, %zu layers, %d slots/layer, %.2f MiB banks\n",
                       item.first, local.banks.size(), params.slots_per_layer, local.bank_bytes / 1048576.0);
    }
    ggml_init_params init{};
    init.mem_size = (16u << 10) + plan.banks.size() * 4 * ggml_tensor_overhead();
    init.no_alloc = true;
    cache->context = ggml_init(init);
    if (!cache->context) {
        delete cache;
        return nullptr;
    }
    for (size_t i = 0; i < stream->extents.size(); ++i) {
        const auto & extent = stream->extents[i];
        const size_t offset = (extent.usable_bytes - cache_scratch_bytes) & ~(cache_alignment - 1);
        cache->scratch.push_back({extent.buffer, static_cast<char *>(extent.base) + offset, cache_scratch_bytes, extent.device});
        if (!cache->quant_scratch.count(extent.device)) {
            auto & span = cache->scratch.back();
            cache->quant_scratch[extent.device] = span.base;
            span.base += cache_quant_bytes;
            span.bytes -= cache_quant_bytes;
        }
    }
    bool ok = true;
    for (const auto & bank_plan : plan.banks) {
        auto & layer = cache->layers[bank_plan.il];
        const auto & extent = stream->extents[bank_plan.extent];
        GGML_ASSERT(extent.device == layer.bind.device);
        const auto buft = ggml_backend_buffer_get_type(extent.buffer);
        for (int f = 0; f < 3; ++f) {
            const auto * source = layer.source[f];
            auto * bank = ggml_new_tensor_3d(cache->context, source->type, source->ne[0], source->ne[1], params.slots_per_layer + 1);
            ggml_format_name(bank, "fn_cache_bank_%d_%d", bank_plan.il, f);
            void * address = static_cast<char *>(extent.base) + bank_plan.offset + bank_plan.family_offset[f];
            ok = ok && ggml_backend_tensor_alloc(extent.buffer, bank, address) == GGML_STATUS_SUCCESS;
            layer.bind.bank[f] = bank;
        }
        auto * map = ggml_new_tensor_2d(cache->context, GGML_TYPE_I32, 1, layer.bind.n_expert);
        ggml_format_name(map, "fn_cache_map_%d", bank_plan.il);
        ggml_set_input(map);
        auto & scratch = cache->scratch[bank_plan.extent];
        const size_t bytes = cache_align(ggml_backend_buft_get_alloc_size(buft, map));
        if (bytes > scratch.bytes) {
            ok = false;
            break;
        }
        ok = ok && ggml_backend_tensor_alloc(scratch.buffer, map, scratch.base) == GGML_STATUS_SUCCESS;
        scratch.base += bytes;
        scratch.bytes -= bytes;
        layer.bind.slot_map = map;
    }
    if (!ok) {
        ggml_free(cache->context);
        delete cache;
        return nullptr;
    }
    LLAMA_LOG_INFO("moe_cache: %zu devices, %zu layers, %d slots/layer, %zu experts, %.2f MiB banks, %zu MiB scratch\n",
                   sources.size(), plan.banks.size(), params.slots_per_layer,
                   plan.banks.size() * params.slots_per_layer, plan.bank_bytes / 1048576.0,
                   stream->extents.size() * (cache_scratch_bytes >> 20));
    return cache;
}

void llama_moe_cache_destroy(llama_moe_cache * cache) {
    if (!cache) {
        return;
    }
    if (cache->cache_mode) {
        llama_moe_stream_cache_drain(cache->stream);
        llama_moe_stream_leave_cache(cache->stream);
    }
    if (cache->context) {
        ggml_free(cache->context);
    }
    delete cache;
}

llama_moe_cache_prompt_state llama_moe_cache_save_prompt(const llama_moe_cache * cache) {
    llama_moe_cache_prompt_state state;
    if (cache) {
        state.resize(cache->layers.size());
        for (size_t i = 0; i < state.size(); ++i) {
            state[i].assign(cache->layers[i].prompt.begin(), cache->layers[i].prompt.end());
        }
    }
    return state;
}

bool llama_moe_cache_restore_prompt(llama_moe_cache * cache, const llama_moe_cache_prompt_state & state) {
    if (!cache || cache->cache_mode || state.size() != cache->layers.size()) {
        return false;
    }
    for (size_t i = 0; i < state.size(); ++i) {
        if (state[i].size() > size_t(cache->params.warm_rows)) return false;
        for (const auto & row : state[i]) {
            for (int32_t expert : row) {
                if (expert < 0 || expert >= cache->layers[i].bind.n_expert) return false;
            }
        }
    }
    for (size_t i = 0; i < state.size(); ++i) {
        cache->layers[i].prompt.assign(state[i].begin(), state[i].end());
    }
    return true;
}

bool llama_moe_cache_prepare(llama_moe_cache * cache, int64_t n_tokens, bool prefill) {
    if (!cache || n_tokens < 1) {
        return false;
    }
    if (cache->failed) {
        throw std::runtime_error("MoE cache context is disabled after a previous device failure");
    }
    cache->active = false;
    if (prefill) {
        if (!cache->collecting_prompt) {
            for (auto & layer : cache->layers) {
                layer.prompt.clear();
            }
            cache->collecting_prompt = true;
        }
        if (n_tokens >= cache->stream->min_tokens) {
            if (cache->cache_mode) {
                if (!llama_moe_stream_leave_cache(cache->stream)) {
                    cache_failure(*cache, "MoE cache to prefill transition failed");
                }
                cache->cache_mode = false;
                ++cache->generation;
            }
            cache_invalidate(*cache);
        } else if (cache->cache_mode) {
            cache_poll(*cache, false);
        }
        return false;
    }
    cache->collecting_prompt = false;
    if (!cache->cache_mode) {
        if (!llama_moe_stream_enter_cache(cache->stream)) {
            cache_failure(*cache, "MoE prefill to cache transition failed");
        }
        cache->cache_mode = true;
        ++cache->generation;
        cache_invalidate(*cache);
        for (const auto & scratch : cache->quant_scratch) {
            if (!llama_moe_stream_cache_set_scratch(cache->stream, scratch.first, scratch.second, cache_quant_bytes)) {
                cache_failure(*cache, "MoE cache scratch binding failed");
            }
        }
        cache_zero_dummy(*cache);
        cache_seed(*cache);
    } else {
        cache_poll(*cache, false);
    }
    // Graph inputs upload both mapping snapshots after prepare returns.
    cache->active = n_tokens <= cache->params.max_rows;
    return cache->active;
}

bool llama_moe_cache_bind(const llama_moe_cache * cache, int il, llama_moe_cache_layer_bind & out) {
    if (!cache || cache->failed || il < 0 || size_t(il) >= cache->layers.size() ||
        !cache->layers[il].bind.slot_map) {
        return false;
    }
    out = cache->layers[il].bind;
    return true;
}

void llama_moe_cache_observe(llama_moe_cache * cache, int il, const int32_t * ids,
                           int64_t n_rows, int n_selected, size_t row_stride, bool prefill) {
    if (!cache || cache->failed || !ids || il < 0 || size_t(il) >= cache->layers.size() ||
        n_rows < 1 || n_selected < 1 || n_selected > 512 || row_stride < size_t(n_selected)) {
        return;
    }
    auto & layer = cache->layers[il];
    if (!layer.bind.slot_map) {
        return;
    }
    const int64_t first = prefill ? std::max(int64_t(0), n_rows - cache->params.warm_rows) : 0;
    for (int64_t row = first; row < n_rows; ++row) {
        std::vector<int32_t> selected;
        for (int k = 0; k < n_selected; ++k) {
            const int expert = ids[size_t(row) * row_stride + k];
            if (expert < 0 || expert >= layer.bind.n_expert) {
                cache_failure(*cache, "MoE cache observed an invalid router expert ID");
            }
            if (prefill) {
                selected.push_back(expert);
                continue;
            }
            layer.seen[expert] = true;
            if (layer.score[expert] < std::numeric_limits<uint16_t>::max()) {
                ++layer.score[expert];
            }
            const int slot = layer.map[expert];
            if (!cache->active) {
                ++cache->baseline_rows;
            } else if (slot) {
                layer.entries[slot].last_use = cache->tick;
                ++cache->hit_rows;
            } else {
                ++cache->miss_rows;
            }
        }
        if (prefill) {
            layer.prompt.push_back(std::move(selected));
            while (layer.prompt.size() > size_t(cache->params.warm_rows)) {
                layer.prompt.pop_front();
            }
        }
    }
}

void llama_moe_cache_finish(llama_moe_cache * cache, int64_t n_tokens, bool prefill) {
    if (!cache || cache->failed || prefill || !cache->cache_mode || n_tokens < 1) {
        return;
    }
    if (!cache->active) {
        for (auto & layer : cache->layers) {
            std::fill(layer.seen.begin(), layer.seen.end(), false);
        }
        return;
    }
    ++cache->tick;
    cache->decay_rows += uint64_t(n_tokens);
    if (cache->decay_rows >= 32) {
        const uint64_t shifts = std::min(uint64_t(16), cache->decay_rows / 32);
        for (auto & layer : cache->layers) {
            for (auto & score : layer.score) {
                score >>= shifts;
            }
        }
        cache->decay_rows %= 32;
    }
    cache_poll(*cache, false);
    struct candidate { int layer; int expert; uint16_t score; uint16_t gain; size_t bytes; };
    std::vector<candidate> candidates;
    for (size_t il = 0; il < cache->layers.size(); ++il) {
        auto & layer = cache->layers[il];
        uint16_t victim_score = std::numeric_limits<uint16_t>::max();
        for (size_t slot = 1; slot < layer.entries.size(); ++slot) {
            const auto & entry = layer.entries[slot];
            if (!entry.pending) {
                victim_score = std::min(victim_score, entry.expert < 0 ? uint16_t(0) : layer.score[entry.expert]);
            }
        }
        for (size_t expert = 0; expert < layer.map.size(); ++expert) {
            if (layer.seen[expert] && !layer.map[expert] && layer.score[expert] >= 2 && layer.score[expert] > victim_score) {
                bool pending = false;
                for (const auto & entry : layer.entries) {
                    pending = pending || (entry.pending && entry.expert == int(expert));
                }
                if (!pending) {
                    candidates.push_back({int(il), int(expert), layer.score[expert],
                                          uint16_t(layer.score[expert] - victim_score), layer.triplet_bytes});
                }
            }
        }
        std::fill(layer.seen.begin(), layer.seen.end(), false);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        const uint64_t left = uint64_t(a.gain) * b.bytes;
        const uint64_t right = uint64_t(b.gain) * a.bytes;
        return left != right ? left > right : a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
    });
    uint64_t budget = cache->params.admission_bytes;
    for (const auto & candidate : candidates) {
        if (candidate.bytes > budget || cache_pending(*cache) >= 63) {
            continue;
        }
        auto & layer = cache->layers[candidate.layer];
        int victim = -1;
        for (int slot = 1; slot <= layer.bind.capacity; ++slot) {
            const auto & entry = layer.entries[slot];
            if (entry.pending) {
                continue;
            }
            if (entry.expert < 0) {
                victim = slot;
                break;
            }
            if (victim < 0 || layer.score[entry.expert] < layer.score[layer.entries[victim].expert] ||
                (layer.score[entry.expert] == layer.score[layer.entries[victim].expert] &&
                 entry.last_use < layer.entries[victim].last_use)) {
                victim = slot;
            }
        }
        if (victim < 0) {
            continue;
        }
        const int old = layer.entries[victim].expert;
        if (llama_moe_plan::cache_admit(candidate.score, old < 0 ? 0 : layer.score[old], old < 0) &&
            cache_submit(*cache, layer, victim, candidate.expert)) {
            budget -= candidate.bytes;
        }
    }
    if ((cache->tick & 63) == 0) {
        LLAMA_LOG_INFO("moe_cache: batches=%" PRIu64 " hit_routes=%" PRIu64 " miss_routes=%" PRIu64
                       " baseline_routes=%" PRIu64 " copied_mib=%.1f\n",
                       cache->tick, cache->hit_rows, cache->miss_rows, cache->baseline_rows,
                       cache->admission_bytes / 1048576.0);
    }
}

bool llama_moe_cache_sync(llama_moe_cache * cache) {
    if (!cache || cache->failed) {
        return false;
    }
    if (cache->cache_mode) {
        cache_poll(*cache, true);
        cache_upload_maps(*cache);
    }
    return true;
}

uint64_t llama_moe_cache_generation(const llama_moe_cache * cache) {
    return cache ? cache->generation : 0;
}

bool llama_moe_cache_failed(const llama_moe_cache * cache) {
    return cache && cache->failed;
}

static const cache_scratch_span * cache_scratch_on_device(const llama_moe_cache * cache, int device, size_t extent) {
    if (cache) {
        for (const auto & span : cache->scratch) {
            if (span.device == device && extent-- == 0) {
                return &span;
            }
        }
    }
    return nullptr;
}

size_t llama_moe_cache_scratch_capacity(const llama_moe_cache * cache, int device, size_t extent) {
    const auto * span = cache_scratch_on_device(cache, device, extent);
    return span ? span->bytes : 0;
}

bool llama_moe_cache_scratch_bind(llama_moe_cache * cache, ggml_tensor * tensor, int device, size_t extent, size_t offset) {
    const auto * found = cache_scratch_on_device(cache, device, extent);
    if (!found || !tensor || offset % cache_alignment) {
        return false;
    }
    const auto & span = *found;
    const size_t bytes = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(span.buffer), tensor);
    if (offset > span.bytes || bytes > span.bytes - offset) {
        return false;
    }
    return ggml_backend_tensor_alloc(span.buffer, tensor, span.base + offset) == GGML_STATUS_SUCCESS;
}
