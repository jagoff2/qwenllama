#ifndef LLAMA_MOE_CACHE_H
#define LLAMA_MOE_CACHE_H

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_model;
struct llama_moe_stream;
struct llama_moe_cache;

struct llama_moe_cache_params {
    bool enable = false;
    int slots_per_layer = 27;
    int warm_rows = 256;
    int max_rows = 49;
    uint64_t admission_bytes = uint64_t(64) << 20;
    bool multi_device = false;
};

struct llama_moe_cache_layer_bind {
    ggml_tensor * bank[3] = { nullptr, nullptr, nullptr };
    ggml_tensor * slot_map = nullptr;
    const int32_t * host_slots = nullptr;
    int n_expert = 0;
    int capacity = 0;
    int device = -1;
};

llama_moe_cache * llama_moe_cache_create(const llama_model & model, llama_moe_stream * stream,
                                        const llama_moe_cache_params & params);
void llama_moe_cache_destroy(llama_moe_cache * cache);

// Host-only routing history survives a synchronized arena-layout transition.
using llama_moe_cache_prompt_state = std::vector<std::vector<std::vector<int32_t>>>;
llama_moe_cache_prompt_state llama_moe_cache_save_prompt(const llama_moe_cache * cache);
bool llama_moe_cache_restore_prompt(llama_moe_cache * cache, const llama_moe_cache_prompt_state & state);

// Caller drains scheduler work before prepare, observe, finish, or destruction.
bool llama_moe_cache_prepare(llama_moe_cache * cache, int64_t n_tokens, bool prefill);
bool llama_moe_cache_bind(const llama_moe_cache * cache, int il, llama_moe_cache_layer_bind & out);
// row_stride is measured in int32_t elements, not bytes.
void llama_moe_cache_observe(llama_moe_cache * cache, int il, const int32_t * ids,
                           int64_t n_rows, int n_selected, size_t row_stride, bool prefill);
void llama_moe_cache_finish(llama_moe_cache * cache, int64_t n_tokens, bool prefill);
bool llama_moe_cache_sync(llama_moe_cache * cache);
uint64_t llama_moe_cache_generation(const llama_moe_cache * cache);
bool llama_moe_cache_failed(const llama_moe_cache * cache);

// Extent is an ordinal among spans on this device. Offsets are after cache
// metadata; binding never allocates device memory or crosses a device boundary.
size_t llama_moe_cache_scratch_capacity(const llama_moe_cache * cache, int device, size_t extent);
bool llama_moe_cache_scratch_bind(llama_moe_cache * cache, ggml_tensor * tensor, int device, size_t extent, size_t offset);

#endif
