// Unit tests for the streamed MoE expert slot planner (llama-moe-plan).
// Planner logic and host-buffer contracts; no CUDA runtime or model required.

#include "llama-moe-stream-plan.h"
#include "llama-moe-stream.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do {                                                       \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_fail;                                                               \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__);                         \
            std::printf(__VA_ARGS__);                                               \
            std::printf("\n");                                                      \
        }                                                                           \
    } while (0)

using namespace llama_moe_plan;

static std::vector<layer_src> make_layers(int n_per_dev, int n_dev, uint64_t bytes, bool resident_first = false) {
    std::vector<layer_src> v;
    for (int il = 0; il < n_per_dev * n_dev; ++il) {
        layer_src l;
        l.il     = il;
        l.device = il / n_per_dev;
        l.nbytes = (resident_first && il == 0) ? 0 : bytes;
        v.push_back(l);
    }
    return v;
}

// Verify the ring invariant for a single device: slot rotation, bootstrap coverage, and that the
// post-fence of ordinal k is exactly the release of the slot that ordinal k + S will be written into.
static void test_ring(int n, int slots) {
    std::vector<int> slot, boot, kick;
    build_ring(n, slots, slot, boot, kick);

    CHECK((int) slot.size() == n, "ring(%d,%d) slot size %zu", n, slots, slot.size());
    CHECK((int) kick.size() == n, "ring(%d,%d) kick size %zu", n, slots, kick.size());

    for (int t = 0; t < n; ++t) {
        CHECK(slot[t] == t % slots, "ring(%d,%d) slot[%d]=%d", n, slots, t, slot[t]);
        CHECK(slot[t] >= 0 && slot[t] < slots, "ring(%d,%d) slot out of range at %d", n, slots, t);

        const int want = (t + slots < n) ? t + slots : -1;
        CHECK(kick[t] == want, "ring(%d,%d) post_kick[%d]=%d want %d", n, slots, t, kick[t], want);
    }

    // every ordinal must be filled exactly once: bootstrap covers [0, min(S,n)), kicks cover the rest
    std::set<int> filled;
    for (int t : boot) {
        CHECK(t >= 0 && t < n, "ring(%d,%d) bootstrap out of range %d", n, slots, t);
        CHECK(filled.insert(t).second, "ring(%d,%d) bootstrap fills %d twice", n, slots, t);
    }
    CHECK((int) boot.size() == (n < slots ? n : slots), "ring(%d,%d) bootstrap size %zu", n, slots, boot.size());

    for (int t = 0; t < n; ++t) {
        if (kick[t] >= 0) {
            CHECK(filled.insert(kick[t]).second, "ring(%d,%d) kick fills %d twice", n, slots, kick[t]);
        }
    }
    for (int t = 0; t < n; ++t) {
        CHECK(filled.count(t) == 1, "ring(%d,%d) ordinal %d never filled", n, slots, t);
    }

    // write-after-read hazard: a slot write for ordinal t must be released by ordinal t-S finishing
    for (int t = slots; t < n; ++t) {
        CHECK(slot[t] == slot[t - slots], "ring(%d,%d) slot reuse mismatch at %d", n, slots, t);
        CHECK(kick[t - slots] == t, "ring(%d,%d) slot %d reused at %d without a release", n, slots, slot[t], t);
    }
}

// Simulate the pipeline and prove there is no deadlock: compute is sequential per device, a fill may
// only start once the previous reader of its slot finished, and compute of t waits for fill(t).
static void test_no_deadlock(int n, int slots) {
    std::vector<int> slot, boot, kick;
    build_ring(n, slots, slot, boot, kick);
    if (n == 0) {
        return;
    }

    std::set<int>   filled;
    std::vector<int> released(n, 0);   // 0 == compute of this ordinal not finished
    for (int t : boot) {
        filled.insert(t);
    }

    int steps = 0;
    const int limit = 10 * n + 100;
    bool progress = true;
    while ((int) filled.size() < n && progress) {
        progress = false;
        ++steps;
        if (steps > limit) {
            CHECK(false, "pipeline(%d,%d) did not converge", n, slots);
            return;
        }
        // compute ordinals in order; the head of the queue is the only candidate
        for (int t = 0; t < n; ++t) {
            if (released[t]) {
                continue;
            }
            if (!filled.count(t)) {
                break;   // head blocked on its fill
            }
            released[t] = 1;
            progress = true;
            if (kick[t] >= 0 && slot[kick[t]] == slot[t]) {
                CHECK(filled.insert(kick[t]).second, "pipeline(%d,%d) double fill %d", n, slots, kick[t]);
                progress = true;
            }
            break;       // sequential compute
        }
    }
    CHECK((int) filled.size() == n, "pipeline(%d,%d) stalled with %zu/%d filled", n, slots, filled.size(), n);
}

static void test_rings(void) {
    const int ns[]     = { 0, 1, 2, 3, 7, 8, 9, 41, 48, 100 };
    const int ss[]     = { 1, 2, 3, 4, 8 };
    for (int s : ss) {
        for (int n : ns) {
            test_ring(n, s);
            test_no_deadlock(n, s);
        }
    }
}

static void test_plan_single_device(void) {
    const uint64_t B = 1024;
    auto layers = make_layers(48, 1, B);

    config cfg;
    cfg.n_slots       = 2;
    cfg.hard_max_slots = 8;
    cfg.slot_bytes    = B;
    cfg.budget_bytes  = 0;

    result r = plan(layers, cfg);
    CHECK(r.enabled, "single device plan disabled: %s", r.reason.c_str());
    CHECK(r.n_slots == 2, "single device n_slots=%d", r.n_slots);
    CHECK(r.n_streamed == 48, "single device n_streamed=%d", r.n_streamed);
    CHECK(r.n_devices == 1, "single device n_devices=%d", r.n_devices);
    CHECK(r.stream_bytes == 48 * B, "single device stream_bytes=%llu", (unsigned long long) r.stream_bytes);
    CHECK(r.arena_bytes == 2 * B, "single device arena_bytes=%llu", (unsigned long long) r.arena_bytes);
    CHECK(r.bootstrap.size() == 2, "bootstrap %zu", r.bootstrap.size());
    CHECK(r.slot[0] == 0 && r.slot[1] == 1 && r.slot[2] == 0, "slot rotation wrong");
    CHECK(r.post_kick[0] == 2 && r.post_kick[46] == -1 && r.post_kick[47] == -1, "post kick ends wrong");
    CHECK((int) r.post_kick.size() == 48, "post_kick size %zu", r.post_kick.size());
}

static void test_plan_two_devices(void) {
    const uint64_t B = 1024;
    // GPU0 owns layers 0..40, GPU1 owns layers 41..47 (bandwidth-proportional split)
    std::vector<layer_src> layers;
    for (int il = 0; il < 48; ++il) {
        layer_src l;
        l.il     = il;
        l.device = il <= 40 ? 0 : 1;
        l.nbytes = B;
        layers.push_back(l);
    }

    config cfg;
    cfg.n_slots      = 3;
    cfg.hard_max_slots = 8;
    cfg.slot_bytes   = B;

    result r = plan(layers, cfg);
    CHECK(r.enabled, "two device plan disabled: %s", r.reason.c_str());
    CHECK(r.n_devices == 2, "n_devices=%d", r.n_devices);
    CHECK(r.n_slots == 3, "n_slots=%d", r.n_slots);

    // each device restarts its own ring at slot 0
    CHECK(r.slot[0] == 0 && r.slot[1] == 1 && r.slot[2] == 2 && r.slot[3] == 0, "dev0 ring wrong");
    CHECK(r.slot[41] == 0, "dev1 ring must start at slot 0, got %d", r.slot[41]);
    CHECK(r.slot[42] == 1 && r.slot[43] == 2 && r.slot[44] == 0, "dev1 ring wrong");

    // device 0: ordinal 38 (il 38) kicks ordinal 41 on device 1? no - kicks must stay on the device
    for (size_t i = 0; i < layers.size(); ++i) {
        const int k = r.post_kick[i];
        if (k >= 0) {
            CHECK(layers[k].device == layers[i].device,
                  "post kick crosses devices: %d -> %d", (int) i, k);
            CHECK(r.ordinal[k] == r.ordinal[i] + r.n_slots,
                  "post kick ordinal skip: %d(%d) -> %d(%d)", (int) i, r.ordinal[i], k, r.ordinal[k]);
        }
    }

    // bootstrap = first n_slots streamed layers of each device
    std::set<int> boot(r.bootstrap.begin(), r.bootstrap.end());
    CHECK(boot.count(0) && boot.count(1) && boot.count(2), "dev0 bootstrap missing");
    CHECK(boot.count(41) && boot.count(42) && boot.count(43), "dev1 bootstrap missing");
    CHECK(boot.size() == 6, "bootstrap size %zu", boot.size());
    CHECK(r.bootstrap.size() == 6, "bootstrap vector %zu", r.bootstrap.size());
}

static void test_plan_mixed_residency(void) {
    const uint64_t B = 1024;
    std::vector<layer_src> layers = make_layers(48, 1, B);
    // layers 0,1 already resident on device (fit kept them there)
    layers[0].nbytes = 0;
    layers[1].nbytes = 0;

    config cfg;
    cfg.n_slots    = 2;
    cfg.slot_bytes = B;

    result r = plan(layers, cfg);
    CHECK(r.enabled, "mixed plan disabled: %s", r.reason.c_str());
    CHECK(r.n_resident == 2, "n_resident=%d", r.n_resident);
    CHECK(r.n_streamed == 46, "n_streamed=%d", r.n_streamed);
    CHECK(r.slot[0] == -1 && r.slot[1] == -1, "resident layers must not get a slot");
    CHECK(r.slot[2] == 0 && r.slot[3] == 1 && r.slot[4] == 0, "ring must skip resident layers");
    CHECK(r.stream_bytes == 46 * B, "stream_bytes=%llu", (unsigned long long) r.stream_bytes);
    CHECK(r.post_kick[2] == 4, "post kick over a resident gap wrong: %d", r.post_kick[2]);
    CHECK(r.post_kick[0] == -1 && r.post_kick[1] == -1, "resident layers must not kick");
}

static void test_clamp(void) {
    CHECK(clamp_slots(0, 0, 4, 8) == 0, "zero slot bytes must disable");
    CHECK(clamp_slots(1024, 0, 4, 8) == 4, "unlimited budget must honour the request");
    CHECK(clamp_slots(1024, 0, 0, 8) == 1, "request below 1 must clamp to 1");
    CHECK(clamp_slots(1024, 0, 99, 8) == 8, "request above hard max must clamp");
    CHECK(clamp_slots(1024, 3500, 4, 8) == 3, "budget must cap at 3, got");
    CHECK(clamp_slots(1024, 1023, 4, 8) == 0, "budget below one slot must disable");
    CHECK(clamp_slots(1, uint64_t(1) << 32, 4, 8) == 4, "large budget must not narrow to zero");

    config cfg;
    cfg.n_slots       = 4;
    cfg.hard_max_slots = 8;
    cfg.slot_bytes    = 1024;
    cfg.budget_bytes  = 3 * 1024;
    result r = plan(make_layers(10, 1, 1024), cfg);
    CHECK(r.enabled && r.n_slots == 3, "plan budget clamp n_slots=%d", r.n_slots);
    CHECK(r.arena_bytes == 3 * 1024, "arena_bytes=%llu", (unsigned long long) r.arena_bytes);

    cfg.budget_bytes = 100;
    result r2 = plan(make_layers(10, 1, 1024), cfg);
    CHECK(!r2.enabled, "plan must disable when a slot does not fit");

    // a layer image bigger than the slot must be rejected rather than truncated
    cfg.budget_bytes = 0;
    cfg.slot_bytes   = 512;
    result r3 = plan(make_layers(10, 1, 1024), cfg);
    CHECK(!r3.enabled, "plan must reject slot_bytes < layer image");

    // nothing to stream
    std::vector<layer_src> all_res = make_layers(8, 1, 0);
    cfg.slot_bytes = 1024;
    result r4 = plan(all_res, cfg);
    CHECK(!r4.enabled, "plan must disable when no layer needs streaming");

    // empty model
    result r5 = plan({}, cfg);
    CHECK(!r5.enabled, "plan must disable for an empty layer list");
}

static void test_source_eligibility(void) {
    ggml_backend_buffer_t cpu = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 64);
    CHECK(cpu != nullptr, "CPU buffer allocation");
    if (!cpu) { return; }
    ggml_tensor t = {};
    t.buffer = cpu;
    t.data = ggml_backend_buffer_get_base(cpu);
    CHECK(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(cpu)) == nullptr,
          "ordinary CPU buffer currently has no device, reproducing the original predicate failure");
    CHECK(llama_moe_stream_host_source(&t), "ordinary CPU source must be accepted without a device");

    // CUDA_Host inherits CPU is_host but advertises a GPU device. Exercise that exact contract.
    ggml_backend_device gpu = {};
    gpu.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    ggml_backend_buffer_type host_type = *ggml_backend_cpu_buffer_type();
    host_type.device = &gpu;
    ggml_backend_buffer mock = {};
    mock.buft = &host_type;
    t.buffer = &mock;
    CHECK(ggml_backend_dev_type(ggml_backend_buft_get_device(mock.buft)) == GGML_BACKEND_DEVICE_TYPE_GPU,
          "pinned host contract must advertise a GPU");
    CHECK(llama_moe_stream_host_source(&t), "GPU-associated host buffer must be accepted");
    host_type.iface.is_host = nullptr;
    CHECK(!llama_moe_stream_host_source(&t), "device buffer without is_host must be rejected");
    host_type.iface.is_host = ggml_backend_cpu_buffer_type()->iface.is_host;
    t.data = nullptr;
    CHECK(!llama_moe_stream_host_source(&t), "unallocated tensor must be rejected");
    t.buffer = nullptr;
    CHECK(!llama_moe_stream_host_source(&t), "missing buffer must be rejected");
    CHECK(!llama_moe_stream_host_source(nullptr), "missing tensor must be rejected");
    ggml_backend_buffer_free(cpu);

    source_traits good;
    good.gate_input = good.have_all = good.contiguous = true;
    good.device = 0;
    good.host_mask = good.data_mask = 7;
    CHECK(!source_rejection(good, 16)[0], "GPU owner with all-host experts (-cmoe) must be eligible");
    auto check = [&](source_traits src, const char * reason) {
        CHECK(std::strcmp(source_rejection(src, 16), reason) == 0, "expected %s, got %s",
              reason, source_rejection(src, 16));
    };
    source_traits s = good; s.gate_input = false; check(s, "missing_gate_input");
    s = good; s.fused_gate_up = true; check(s, "unsupported_fused_gate_up");
    s = good; s.have_all = false; check(s, "missing_expert_family");
    s = good; s.scales = true; check(s, "unsupported_expert_scales");
    s = good; s.biases = true; check(s, "unsupported_expert_biases");
    s = good; s.device = -1; check(s, "no_cuda_layer_owner");
    s = good; s.device = 16; check(s, "device_ordinal_out_of_range");
    s = good; s.data_mask = 3; check(s, "missing_expert_data");
    s = good; s.contiguous = false; check(s, "noncontiguous_expert_layout");
    s = good; s.host_mask = 0; s.device_mask = 7; check(s, "device_resident");
    s = good; s.host_mask = 3; s.device_mask = 4; check(s, "mixed_host_nonhost_experts");
    s = good; s.host_mask = 0; check(s, "nonhost_expert_source");
}

static void test_budget_units(void) {
    const uint64_t mib = uint64_t(1) << 20;
    llama_moe_stream_params sp;
    CHECK(sp.budget_bytes == 0, "default budget is auto");
    sp.set_budget_mib(4096);
    CHECK(sp.budget_bytes == 4096 * mib, "4096 MiB must convert exactly once");
    CHECK(arena_budget(sp.budget_bytes, 8192 * mib, 2048 * mib) == 4096 * mib, "explicit budget");
    CHECK(arena_budget(sp.budget_bytes, 3072 * mib, 2048 * mib) == 3072 * mib, "explicit budget capped by free VRAM");
    sp.set_budget_mib(0);
    CHECK(arena_budget(sp.budget_bytes, 8192 * mib, 2048 * mib) == 6144 * mib, "auto budget reserves headroom");
    CHECK(arena_budget(sp.budget_bytes, 2048 * mib, 2048 * mib) == 0, "exhausted auto budget is zero");
    CHECK(arena_budget(sp.budget_bytes, 1024 * mib, 2048 * mib) == 0, "auto reserve must not underflow");
    sp.set_budget_mib(std::numeric_limits<uint32_t>::max());
    CHECK(sp.budget_bytes == uint64_t(std::numeric_limits<uint32_t>::max()) * mib, "MiB conversion uses 64-bit arithmetic");

    config cfg;
    cfg.n_slots = 3;
    cfg.slot_bytes = 1024 * mib;
    cfg.slot_overhead_bytes = 16 * mib;
    cfg.budget_bytes = 3072 * mib;
    const result r = plan(make_layers(48, 1, cfg.slot_bytes), cfg);
    CHECK(r.enabled && r.n_slots == 2, "budget must include per-slot allocation slack");
    CHECK(r.arena_bytes == 2080 * mib, "reported arena bytes include allocation slack");
    cfg.budget_bytes = 1024 * mib;
    CHECK(!plan(make_layers(48, 1, cfg.slot_bytes), cfg).enabled, "image-only capacity cannot hold an allocation");
}

static void test_descriptor_capacity(void) {
    ggml_init_params ip = {};
    ip.mem_size = llama_moe_stream_descriptor_bytes(48);
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr, "arena descriptor context");
    if (!ctx) { return; }
    for (int i = 0; i < 48 * 3; ++i) {
        CHECK(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4) != nullptr, "descriptor %d fits", i);
    }
    ggml_free(ctx);
}

static void test_unique_experts(void) {
    // one token touches exactly k experts
    CHECK(std::fabs(expected_unique_experts(512, 10, 1) - 10.0) < 1e-9, "one token case");
    // the prefill case from the target model: 512 experts, k=10, a 512 token ubatch touches nearly all
    const double u512 = expected_unique_experts(512, 10, 512);
    CHECK(u512 > 511.0 && u512 <= 512.0, "512/10/512 unique=%f", u512);
    // small ubatches touch far fewer experts - this is why the streamed path needs large ubatches
    const double u64 = expected_unique_experts(512, 10, 64);
    CHECK(u64 > 300 && u64 < 420, "512/10/64 unique=%f", u64);
    // monotonic in tokens, and bounded by the expert count
    double prev = 0;
    for (int64_t n = 1; n <= 4096; n *= 2) {
        const double u = expected_unique_experts(512, 10, n);
        CHECK(u >= prev && u <= 512.0, "monotonic/bound failed at n=%lld u=%f", (long long) n, u);
        prev = u;
    }
    CHECK(expected_unique_experts(512, 10, 0) == 0.0, "zero tokens");
    CHECK(expected_unique_experts(8, 8, 1) == 8.0, "k >= n_expert case");
}

// The arena layout is shared with the CUDA engine through explicit offsets, so the guarantees the
// engine relies on are checked here: aligned family starts, no overlap, bounded image size.
static void test_layout(void) {
    uint64_t off[3];

    // exact multiples of the alignment: nothing is padded
    const uint64_t exact[3] = { 512, 1024, 2048 };
    uint64_t total = layout_families(exact, 512, off);
    CHECK(off[0] == 0 && off[1] == 512 && off[2] == 1536, "exact layout %llu %llu %llu",
          (unsigned long long) off[0], (unsigned long long) off[1], (unsigned long long) off[2]);
    CHECK(total == 3584, "exact total %llu", (unsigned long long) total);

    // quantized block sizes that are not multiples of the alignment (Q4_K super-block 144 B,
    // Q5_1 288 B, Q8_0 34 B per 32 elements) must still produce aligned, disjoint ranges
    const uint64_t odd[3] = { 144 * 7, 288 * 13 + 288, 34 * 4097 };
    total = layout_families(odd, 512, off);
    for (int f = 0; f < 3; ++f) {
        CHECK(off[f] % 512 == 0, "family %d start %llu not aligned", f, (unsigned long long) off[f]);
        CHECK(off[f] + odd[f] <= total, "family %d runs past the image", f);
        if (f > 0) {
            CHECK(off[f] >= off[f - 1] + odd[f - 1], "family %d overlaps the previous one", f);
        }
    }
    const uint64_t sum = odd[0] + odd[1] + odd[2];
    CHECK(total >= sum && total <= sum + 2 * 512, "padded total %llu vs sum %llu",
          (unsigned long long) total, (unsigned long long) sum);

    // a family may be absent; the remaining ones still start aligned and packed
    const uint64_t partial[3] = { 0, 100, 4096 + 100 };
    total = layout_families(partial, 512, off);
    CHECK(off[0] == 0, "absent family must report offset 0");
    CHECK(off[1] == 0, "first used family should start at 0, got %llu", (unsigned long long) off[1]);
    CHECK(off[2] == 512, "second used family should start at 512, got %llu", (unsigned long long) off[2]);
    CHECK(total == 512 + 4096 + 100, "partial total %llu", (unsigned long long) total);

    // align == 1 keeps the layout packed
    total = layout_families(odd, 1, off);
    CHECK(total == sum, "align 1 should not pad, got %llu want %llu",
          (unsigned long long) total, (unsigned long long) sum);

    // empty image
    const uint64_t none[3] = { 0, 0, 0 };
    CHECK(layout_families(none, 512, off) == 0, "empty image must be zero bytes");

    // the per-layer image sizes the planner is given must fit the uniform slot the engine is told
    config cfg;
    cfg.slot_bytes = sum + 2 * 512;
    std::vector<layer_src> ls(1);
    ls[0].il     = 0;
    ls[0].device = 0;
    ls[0].nbytes = total;
    result r = plan(ls, cfg);
    CHECK(r.enabled, "planner must accept an image that fits the slot: %s", r.reason.c_str());
}

static void test_place_arena(void);


static void test_place_arena(void) {
    using llama_moe_plan::PLACE_LAYER;
    using llama_moe_plan::PLACE_PRIMARY;
    using llama_moe_plan::PLACE_SPLIT;

    const uint64_t gib = 1024ull * 1024 * 1024;

    // the measured geometry of this box: 41 layers on the gen5 x8 device, 7 on the gen4 x4 one
    std::vector<int>      owner;
    std::vector<uint64_t> bytes;
    for (int i = 0; i < 41; ++i) { owner.push_back(0); bytes.push_back(gib); }
    for (int i = 0; i < 7;  ++i) { owner.push_back(1); bytes.push_back(gib); }
    std::vector<float> no_w;

    std::string note;

    // the default must be the identity, or the flag would change the baseline
    {
        const std::vector<int> a = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_LAYER, -1, no_w, &note);
        CHECK(a == owner, "PLACE_LAYER must be the identity");
        CHECK(note.empty(), "PLACE_LAYER must not report movement");
    }

    // primary is a choice, not an opinion: either device can host every arena
    {
        const std::vector<int> a0 = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_PRIMARY, 0, no_w, &note);
        int n = 0;
        for (int d : a0) { n += d == 0; }
        CHECK(n == 48, "PLACE_PRIMARY(0) put %d of 48 layers on device 0", n);

        const std::vector<int> a1 = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_PRIMARY, 1, no_w, &note);
        n = 0;
        for (int d : a1) { n += d == 1; }
        CHECK(n == 48, "PLACE_PRIMARY(1) put %d of 48 layers on device 1", n);
    }

    // auto = the device carrying the most streamed bytes, and a bad device number falls back to it
    {
        const std::vector<int> auto_a = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_PRIMARY, -1, no_w, &note);
        CHECK(auto_a[0] == 0 && auto_a[47] == 0, "auto primary should pick the device with 41 GiB");
        CHECK(note.find("device 0") != std::string::npos, "note must name the device: %s", note.c_str());

        const std::vector<int> bad = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_PRIMARY, 9, no_w, &note);
        CHECK(bad == auto_a, "an out-of-range primary device must fall back to auto");

        std::vector<int> one_dev(6, 1);
        std::vector<uint64_t> one_bytes(6, gib);
        const std::vector<int> a = llama_moe_plan::place_arena(one_dev, one_bytes, 2, PLACE_PRIMARY, -1, no_w, &note);
        CHECK(a[0] == 1, "auto primary with all layers on device 1 must stay there, got %d", a[0]);
    }

    // split: equal weights balance the bytes exactly here
    {
        const std::vector<int> a = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_SPLIT, -1, no_w, &note);
        int n0 = 0, n1 = 0;
        for (int d : a) {
            CHECK(d == 0 || d == 1, "split assigned a device outside the set: %d", d);
            n0 += d == 0;
            n1 += d == 1;
        }
        CHECK(n0 == 24 && n1 == 24, "equal split of 48 equal layers gave %d/%d", n0, n1);
    }

    // weighted split hits the requested ratio, and a weight of 0 excludes a device entirely
    {
        const std::vector<float> w41 = { 4.0f, 1.0f };
        const std::vector<int> a = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_SPLIT, -1, w41, &note);
        int n0 = 0;
        for (int d : a) { n0 += d == 0; }
        CHECK(n0 == 38, "4:1 split of 48 layers should put 38 on device 0, got %d", n0);

        const std::vector<float> w_only0 = { 1.0f, 0.0f };
        const std::vector<int> b = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_SPLIT, -1, w_only0, &note);
        int n1 = 0;
        for (int d : b) { n1 += d == 1; }
        CHECK(n1 == 0, "a zero weight must exclude the device, %d layers landed there", n1);

        // a weight list shorter than the device list is padded with 1, so 2:1 over 2 devices
        const std::vector<float> w_two = { 2.0f };
        const std::vector<int> c = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_SPLIT, -1, w_two, &note);
        int m0 = 0;
        for (int d : c) { m0 += d == 0; }
        CHECK(m0 == 32, "missing weights must default to 1, want 32 on device 0, got %d", m0);
    }

    // the guarantee behind the greedy rule: no device is further from its byte target than one image
    {
        const std::vector<float> w = { 3.0f, 1.0f };
        const std::vector<int> a = llama_moe_plan::place_arena(owner, bytes, 2, PLACE_SPLIT, -1, w, &note);
        uint64_t total = 0, got[2] = { 0, 0 };
        for (size_t i = 0; i < owner.size(); ++i) {
            total += bytes[i];
            got[a[i]] += bytes[i];
        }
        for (int d = 0; d < 2; ++d) {
            const double target = (double) total * (w[d] / 4.0);
            const double err    = std::fabs(target - (double) got[d]);
            CHECK(err <= (double) gib + 1e-6, "device %d got %.0f MiB, target %.0f MiB",
                  d, got[d] / 1048576.0, target / 1048576.0);
        }
    }

    // layers that are not streamed stay out, and the mapping is deterministic
    {
        std::vector<int> mixed = owner;
        mixed[3] = -1;
        mixed[44] = -1;
        const std::vector<int> a = llama_moe_plan::place_arena(mixed, bytes, 2, PLACE_PRIMARY, 1, no_w, &note);
        CHECK(a[3] == -1 && a[44] == -1, "non-streamed layers must stay -1");
        const std::vector<int> b = llama_moe_plan::place_arena(mixed, bytes, 2, PLACE_PRIMARY, 1, no_w, &note);
        CHECK(a == b, "placement must be deterministic");

        std::vector<int> none(4, -1);
        std::vector<uint64_t> none_b(4, 0);
        const std::vector<int> c = llama_moe_plan::place_arena(none, none_b, 2, PLACE_SPLIT, -1, no_w, &note);
        CHECK(c == none, "nothing streamed must place nothing");
    }
}

int main(void) {
    test_rings();
    test_plan_single_device();
    test_plan_two_devices();
    test_plan_mixed_residency();
    test_clamp();
    test_source_eligibility();
    test_budget_units();
    test_descriptor_capacity();
    test_unique_experts();
    test_layout();
    test_place_arena();

    std::printf("test-moe-stream-plan: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
