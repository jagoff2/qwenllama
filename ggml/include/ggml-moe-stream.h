// Public ABI for the streamed MoE expert prefill engine (CUDA backend implementation).
//
// Goal: during a large cold prefill, the expert weights of MoE layers that do not fit in VRAM are
// held in host RAM. Instead of evaluating those layers on the CPU, the engine keeps a small ring of
// device-resident "expert arena slots" and streams whole layer images through them:
//
//   host RAM (authoritative quant weights)  --[async H2D]-->  arena slot  --> MUL_MAT_ID on device
//
// Because a prefill micro-batch of >= ~512 tokens touches essentially every routed expert of a
// layer, the whole layer image has to cross PCIe once per micro-batch either way. Moving the GEMMs
// to the device removes the CPU as the bottleneck; the ring plus CUDA events keep the transfer and
// the compute overlapped.
//
// The engine lives inside the CUDA backend (it needs streams, events, pinned memory and the device
// context). llama.cpp configures it through these entry points, obtained with
// ggml_backend_reg_get_proc_address(), so nothing outside the CUDA backend links against it.
#ifndef GGML_MOE_STREAM_H
#define GGML_MOE_STREAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_MOE_STREAM_MAX_FAMILIES 3   // up, gate, down

// One host-resident MoE layer: the three expert tensors of that layer, in host memory.
struct ggml_cuda_moe_stream_layer {
    int32_t  layer;                                  // model layer index
    // per family: authoritative host pointer and byte size (0 == family not used)
    const void * host[GGML_MOE_STREAM_MAX_FAMILIES];
    size_t       nbytes[GGML_MOE_STREAM_MAX_FAMILIES];
    // byte offset of each family inside an arena slot. the caller owns the layout (it is the code
    // that bound the ggml descriptors), the engine only copies bytes to it
    size_t       dst_off[GGML_MOE_STREAM_MAX_FAMILIES];
    // CUDA kernels may read past the final row; clear these bytes on every fill.
    size_t       pad_bytes[GGML_MOE_STREAM_MAX_FAMILIES];
};

// One arena slot: a contiguous device allocation that mirrors the byte layout of a layer image.
struct ggml_cuda_moe_stream_slot {
    int32_t  slot;                                   // slot index used by the fence op
    void *   dev;                                    // device base pointer of the slot
    size_t   cap;                                    // usable bytes of the arena (and of the pinned mirror)
    size_t   nbytes[GGML_MOE_STREAM_MAX_FAMILIES];   // largest family size on this device (diagnostics)
};

struct ggml_cuda_moe_stream_params {
    int      n_slots;        // slots registered for this device
    int      pin_host;       // 1 = stage through a pinned host ring, 0 = cudaMemcpyAsync from pageable
    size_t   pin_bytes;      // pinned staging bytes per device (0 -> engine default when pin_host)
    int      device;         // CUDA device ordinal these params apply to
};

// Aggregated telemetry. Everything is a running total since the last reset; the caller prints deltas.
struct ggml_cuda_moe_stream_stats {
    uint64_t n_fence_pre;          // pre-fences executed
    uint64_t n_fence_post;         // post-fences executed
    uint64_t n_fill;               // H2D fills issued
    uint64_t n_fill_bytes;         // bytes transferred host -> device
    uint64_t n_stage_bytes;        // bytes copied host -> pinned staging
    uint64_t n_fill_sync;          // fills that had to be done synchronously at the pre-fence
    uint64_t n_slot_wait_ns;       // device-side stall time waiting for a fill (host-measured)
    uint64_t n_stage_ns;           // host memcpy time into the pinned ring
    uint64_t n_copy_submit_ns;     // time spent submitting async copies
    uint64_t n_worker_wake;        // worker activations
    uint64_t n_worker_skip;        // worker activations that found nothing to do
    uint64_t n_error;              // engine errors seen (degradations, not fatal)
    double   n_stage_bytes_dummy;  // reserved
};

// Proc-address names exported by the CUDA backend device registry.
#define GGML_MOE_STREAM_PROC_SETUP   "ggml_cuda_moe_stream_setup_v2"
#define GGML_MOE_STREAM_PROC_TEARDOWN "ggml_cuda_moe_stream_teardown"
#define GGML_MOE_STREAM_PROC_STATS   "ggml_cuda_moe_stream_stats"
#define GGML_MOE_STREAM_PROC_RESET   "ggml_cuda_moe_stream_reset"
#define GGML_MOE_STREAM_PROC_ACTIVE  "ggml_cuda_moe_stream_active"
#define GGML_MOE_STREAM_PROC_BEGIN   "ggml_cuda_moe_stream_begin"

typedef int  (*ggml_cuda_moe_stream_setup_t)(
        const struct ggml_cuda_moe_stream_params   * params,
        const struct ggml_cuda_moe_stream_layer    * layers,
        int                                          n_layers,
        const struct ggml_cuda_moe_stream_slot     * slots,
        int                                          n_slots);
typedef void (*ggml_cuda_moe_stream_teardown_t)(int device);
typedef void (*ggml_cuda_moe_stream_stats_t)(int device, struct ggml_cuda_moe_stream_stats * out);
typedef void (*ggml_cuda_moe_stream_reset_t)(int device);
typedef int  (*ggml_cuda_moe_stream_active_t)(int device);

// Arm the ring for one micro-batch: queue the fills for the bootstrap slots (the first n_slots
// streamed layers, which have no predecessor release point). Must be called on the thread that is
// about to compute the graph, once per streamed micro-batch, before ggml graph compute.
typedef void (*ggml_cuda_moe_stream_begin_t)(int device, const int32_t * slots,
                                             const int32_t * layers, int n);

#ifdef __cplusplus
}
#endif

#endif // GGML_MOE_STREAM_H
