// Streamed MoE expert prefill - llama side.
//
// Owns the device arena slots that the CUDA engine (ggml/src/ggml-cuda/moe-stream.cu) rotates expert
// layer images through, and answers the two questions the graph builder asks:
//
//   * is this micro-batch streamed?                     (llama_moe_stream_select)
//   * which tensors / fence parameters does layer il use? (llama_moe_stream_layer_bind)
//
// When disabled, or when the micro-batch is below the token threshold, select() returns nullptr and
// build_moe_ffn emits the stock graph, i.e. exactly the pre-existing behaviour.
#ifndef LLAMA_MOE_STREAM_H
#define LLAMA_MOE_STREAM_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-moe-stream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

#define LLAMA_MOE_STREAM_MAX_DEVICES 16

struct llama_moe_stream_params {
    bool     enable      = false;
    int      n_slots     = 3;      // arena slots per device
    int64_t  min_tokens  = 1024;   // only stream micro-batches at least this large
    int      pin_host    = 1;      // stage through pinned host memory before the H2D copy
    uint64_t budget_bytes = 0;    // per-device VRAM ceiling for slots (0 = auto)
    int      hard_max_slots = 8;

    void set_budget_mib(uint32_t mib) { budget_bytes = uint64_t(mib) << 20; }

    // where the arena slot of a layer lives. independent of which device computes the rest of the
    // layer: see llama_moe_plan::place_arena and --fn-stream-gpu-mode.
    int      place_mode   = 0;     // 0 = owning device, 1 = single device, 2 = weighted split
    int      place_device = -1;    // mode 1 target device; -1 = device owning most streamed layers
    std::vector<float> place_w;    // mode 2 per-device byte weights; empty = equal
};

inline bool llama_moe_stream_host_source(const ggml_tensor * t) {
    return t && t->buffer && t->data && ggml_backend_buffer_is_host(t->buffer);
}

inline size_t llama_moe_stream_descriptor_bytes(size_t n_layer) {
    return (16u << 10) + n_layer * 3 * ggml_tensor_overhead();
}

// What build_moe_ffn needs in order to emit one streamed layer.
struct llama_moe_stream_layer_bind {
    ggml_tensor * arena[3] = { nullptr, nullptr, nullptr };  // up, gate, down views of the slot
    int32_t layer      = -1;
    int32_t slot       = -1;
    int32_t kick_slot  = -1;
    int32_t kick_layer = -1;
    int64_t nbytes     = 0;
};

struct llama_moe_stream {
    bool     enabled  = false;
    bool     armed    = false;   // a graph currently refers to the ring for this micro-batch
    int64_t  min_tokens = 0;
    int      n_slots  = 0;
    size_t   slot_bytes[LLAMA_MOE_STREAM_MAX_DEVICES] = { 0 };

    std::vector<llama_moe_stream_layer_bind> bind;   // indexed by layer, slot < 0 when not streamed

    // per device bootstrap ring priming lists
    std::vector<int32_t> boot_slot[LLAMA_MOE_STREAM_MAX_DEVICES];
    std::vector<int32_t> boot_layer[LLAMA_MOE_STREAM_MAX_DEVICES];

    // ownership: the descriptor context and the arena buffers must outlive every graph that uses them
    ggml_context *          ctx_arena = nullptr;
    std::vector<ggml_backend_buffer_t> slot_buf;

    // One engine, and therefore one ring, per CUDA device that owns arena slots: with -sm layer, with
    // --fn-stream-gpu-mode primary, or with a weighted split, the streamed layers span more than one
    // GPU. Every control path (arm, report, teardown) must iterate `devices`; `device` is only
    // devices.front(), kept so logs can name a primary.
    std::vector<int> devices;
    int  device     = -1;
    bool registered = false;                            // the engine owns a live slot list
    bool destroyed  = false;

    // engine entry points (nullptr when the CUDA backend did not provide them)
    int  (* proc_setup)(const struct ggml_cuda_moe_stream_params *, const struct ggml_cuda_moe_stream_layer *, int,
                        const struct ggml_cuda_moe_stream_slot *, int)   = nullptr;
    void (* proc_teardown)(int)      = nullptr;
    void (* proc_stats)(int, struct ggml_cuda_moe_stream_stats *) = nullptr;
    void (* proc_reset)(int)         = nullptr;
    void (* proc_begin)(int, const int32_t *, const int32_t *, int) = nullptr;

    uint64_t n_ubatch_streamed  = 0;
    uint64_t n_tokens_streamed  = 0;
    uint64_t n_bytes_per_ubatch = 0;   // planned PCIe bytes per streamed micro-batch
    int      n_remote_layers    = 0;   // streamed layers whose arena is on another device
    uint64_t n_bytes_remote     = 0;   // their per-micro-batch bytes (they cross a device hop)

    std::string note;
};

// Build the engine. `dev_of_layer` maps layer index -> the device that owns the layer
// (llama_model::dev_layer); nullptr entries mean "not a GPU, leave on the baseline path".
// Returns nullptr when streaming cannot apply; never fatal.
llama_moe_stream * llama_moe_stream_create(const llama_model & model,
                                           const std::vector<ggml_backend_dev_t> & dev_of_layer,
                                           const llama_moe_stream_params & params);
void llama_moe_stream_destroy(llama_moe_stream * ms);

bool llama_moe_stream_enabled(const llama_moe_stream * ms);

// Choose the plan for a micro-batch of n_tokens; nullptr => build the baseline graph.
const llama_moe_stream * llama_moe_stream_select(const llama_moe_stream * ms, int64_t n_tokens);

// Fill `out` for layer il; false when the layer is not streamed (baseline tensors apply).
bool llama_moe_stream_bind(const llama_moe_stream * ms, int il, llama_moe_stream_layer_bind & out);

// Prime the ring before the graph runs. Does nothing unless llama_moe_stream_select(ms, n_tokens)
// would succeed, so callers can invoke it unconditionally before ggml graph compute.
void llama_moe_stream_arm(llama_moe_stream * ms, int64_t n_tokens);

// Print aggregated telemetry for the micro-batch that just finished.
void llama_moe_stream_report(llama_moe_stream * ms, int64_t n_tokens, double t_ms);

#endif // LLAMA_MOE_STREAM_H
