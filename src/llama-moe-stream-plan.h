// Streamed MoE expert prefill: slot planner.
//
// Pure, dependency-free logic that decides, for one prefill micro-batch, which arena slot holds
// which MoE layer image, which fills must be issued before the graph is launched, and which fill a
// post-layer fence kicks. Kept free of ggml / CUDA / llama types so it can be unit tested anywhere.
//
// Model: on one device the streamed layers are visited in a fixed order 0..n-1. There are S arena
// slots. The fill of ordinal m lands in slot m % S and must not start before the compute that read
// slot m % S for ordinal m - S has finished. That release point is the post-fence of ordinal m - S,
// so the post-fence of ordinal k kicks the fill of ordinal k + S. Ordinals 0..min(S,n)-1 have no
// predecessor, so they are filled before the graph runs (bootstrap).
#ifndef LLAMA_MOE_STREAM_PLAN_H
#define LLAMA_MOE_STREAM_PLAN_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama_moe_plan {

// A MoE layer whose expert image lives in host memory and therefore has to cross PCIe before the
// layer can be evaluated on the device that owns it.
struct layer_src {
    int      il     = -1;   // model layer index
    int      device = -1;   // device ordinal that owns the layer (dense weights + KV live there)
    uint64_t nbytes = 0;    // bytes of the streamed expert image (up + gate + down)
};

struct config {
    int      n_slots      = 2;    // requested arena slots per device (>= 1; 1 == no prefetch)
    int      hard_max_slots = 8;  // ceiling for a requested slot count
    uint64_t slot_bytes     = 0;  // uniform slot size, must be >= every layer image
    uint64_t budget_bytes   = 0;  // device bytes available for slots; 0 == unlimited
    uint64_t slot_overhead_bytes = 0; // allocation padding per slot
};

struct source_traits {
    bool gate_input = false;
    bool fused_gate_up = false;
    bool have_all = false;
    bool scales = false;
    bool biases = false;
    bool contiguous = false;
    int device = -1;
    unsigned host_mask = 0;
    unsigned device_mask = 0;
    unsigned data_mask = 0;
};

// Empty string means eligible; otherwise returns the first failed predicate.
const char * source_rejection(const source_traits & src, int max_devices);

// Auto leaves reserve bytes; explicit budgets are capped by observed free memory.
uint64_t arena_budget(uint64_t requested_bytes, uint64_t free_bytes, uint64_t reserve_bytes);

struct result {
    bool     enabled        = false;
    int      n_slots        = 0;   // slots actually used after clamping
    uint64_t slot_bytes     = 0;
    uint64_t arena_bytes    = 0;   // n_slots * slot_bytes, per device
    int      n_streamed     = 0;   // layers that need streaming
    int      n_resident     = 0;   // layers already resident on device (nbytes == 0)
    int      n_devices      = 0;   // devices that carry at least one streamed layer
    uint64_t stream_bytes   = 0;   // bytes crossing PCIe per micro-batch (all devices)
    std::vector<int> slot;         // per input layer position: arena slot, or -1 when resident
    std::vector<int> ordinal;      // per input layer position: per-device visit ordinal, or -1
    std::vector<int> bootstrap;    // input positions that must be filled before graph launch
    std::vector<int> post_kick;    // per input position: input position kicked by its post-fence, or -1
    std::string reason;            // why the plan is disabled
};

// Largest slot size that fits the budget for `want` slots, clamped to [0, hard_max].
int clamp_slots(uint64_t slot_bytes, uint64_t budget_bytes, int want, int hard_max);

// Byte layout of one layer image inside an arena slot: nbytes includes backend padding, family f starts at dst_off[f], every used
// family start is padded to `align` so the device kernels see the alignment they assume for a tensor
// base. Returns the number of bytes the image occupies (<= sum(nbytes) + 2*align).
uint64_t layout_families(const uint64_t nbytes[3], uint64_t align, uint64_t dst_off[3]);

// Single-device ring. See the file comment for the invariant.
void build_ring(int n, int n_slots, std::vector<int> & slot, std::vector<int> & bootstrap,
                std::vector<int> & post_kick);

// Whole-model plan. `layers` must be in model layer order (ascending il). Layers with nbytes == 0
// are treated as already resident on their device and are not streamed.
result plan(const std::vector<layer_src> & layers, const config & cfg);

// Where the arena slot holding a layer image should live. The arena does not have to sit on the device
// that owns the layer: the expert GEMM then runs on the arena device and the activations cross the
// link instead of the expert image. These are the candidate strategies from the design notes, exposed
// at runtime so an offline sweep can pick the winner.
enum place_mode {
    PLACE_LAYER   = 0,   // Candidate C: arena on the device that owns the layer
    PLACE_PRIMARY = 1,   // Candidate A: every arena on one device (GPU0-only expert mode)
    PLACE_SPLIT   = 2,   // Candidate B: arenas spread over devices by byte weight (dual-GPU mode)
};

// Arena device per layer, in layer order. Entries with layer_dev < 0 are not streamed and come back
// as -1 unchanged. `weights` shorter than n_dev is padded with 1.0; a zero weight excludes the device.
// Deterministic: the same input always produces the same assignment.
std::vector<int> place_arena(const std::vector<int> & layer_dev,
                             const std::vector<uint64_t> & nbytes,
                             int n_dev, int mode, int primary_dev,
                             const std::vector<float> & weights,
                             std::string * note);

// Expert-reuse estimate for a micro-batch: expected number of distinct experts touched when each of
// `n_tokens` tokens picks `k` of `n_expert` experts uniformly at random. Used for reporting only;
// the streamed image is the whole layer, so this is diagnostic, not a traffic model.
double expected_unique_experts(int64_t n_expert, int64_t k, int64_t n_tokens);

} // namespace llama_moe_plan

#endif // LLAMA_MOE_STREAM_PLAN_H
