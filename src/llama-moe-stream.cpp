#include "llama-moe-stream.h"
#include "llama-moe-stream-plan.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-moe-stream.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

// slack so that ggml's per-tensor allocation alignment never runs past the end of an arena buffer
#define FN_SLOT_SLACK (16u << 20)

// expert arena descriptor alignment: at least GGML_MEM_ALIGN and at least the 512 B granularity
// that the CUDA buffer allocator uses, so a family start is never a misaligned tensor base
#define FN_SLOT_ALIGN 512u

namespace {

struct cuda_dev_info {
    ggml_backend_reg_t reg  = nullptr;
    ggml_backend_dev_t dev  = nullptr;
    int                ord  = -1;
};

// enumerate the CUDA devices once; the ordinal must match ggml-cuda's own device numbering
std::vector<cuda_dev_info> cuda_devices() {
    std::vector<cuda_dev_info> out;
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        return out;
    }
    const size_t n = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_reg_dev_get(reg, i);
        if (!d) {
            continue;
        }
        cuda_dev_info c;
        c.reg = reg;
        c.dev = d;
        c.ord = (int) i;
        out.push_back(c);
    }
    return out;
}

void * proc(ggml_backend_reg_t reg, const char * name) {
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

// the engine identifies devices by the CUDA registry index, which is the ordinal ggml-cuda uses
int dev_ordinal(const std::vector<cuda_dev_info> & devs, ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return -1;
    }
    for (const auto & c : devs) {
        if (c.dev == dev) {
            return c.ord;
        }
    }
    return -1;
}

// the three expert tensors of one MoE layer, in engine family order
ggml_tensor * family_tensor(const llama_layer & l, int f) {
    switch (f) {
        case 0: return l.ffn_up_exps;
        case 1: return l.ffn_gate_exps;
        case 2: return l.ffn_down_exps;
        default: return nullptr;
    }
}

ggml_backend_dev_t buf_dev(const ggml_backend_buffer_t buf) {
    return buf ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf)) : nullptr;
}

bool is_device_resident(const ggml_tensor * t) {
    if (!t || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    ggml_backend_dev_t dev = buf_dev(t->buffer);
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
}

void log_tensor(int il, const char * family, const ggml_tensor * t) {
    const ggml_backend_buffer_t buf = t ? t->buffer : nullptr;
    const ggml_backend_buffer_type_t buft = buf ? ggml_backend_buffer_get_type(buf) : nullptr;
    const ggml_backend_dev_t dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
    LLAMA_LOG_INFO("moe_stream_diag: layer=%d family=%s tensor=%s data=%p buffer=%p buft=%s "
                   "buffer_host=%d buft_host=%d buft_device=%s device_type=%d nbytes=%zu host_source=%d\n",
                   il, family, t ? t->name : "(missing)", t ? t->data : nullptr, (void *) buf,
                   buft ? ggml_backend_buft_name(buft) : "(none)",
                   buf ? ggml_backend_buffer_is_host(buf) : 0, buft ? ggml_backend_buft_is_host(buft) : 0,
                   dev ? ggml_backend_dev_name(dev) : "(null)", dev ? (int) ggml_backend_dev_type(dev) : -1,
                   t ? ggml_nbytes(t) : 0, llama_moe_stream_host_source(t));
}

} // namespace

llama_moe_stream * llama_moe_stream_create(const llama_model & model,
                                           const std::vector<ggml_backend_dev_t> & dev_of_layer,
                                           const llama_moe_stream_params & params) {
    if (!params.enable) {
        return nullptr;
    }
    const char * diag_env = std::getenv("LLAMA_MOE_STREAM_DIAGNOSTICS");
    const bool diagnostics = diag_env && diag_env[0] && std::strcmp(diag_env, "0") != 0;

    const auto devs = cuda_devices();
    if (devs.empty()) {
        LLAMA_LOG_INFO("%s: no CUDA backend, streamed experts unavailable\n", __func__);
        return nullptr;
    }

    void * v_setup = proc(devs[0].reg, GGML_MOE_STREAM_PROC_SETUP);
    if (!v_setup) {
        LLAMA_LOG_INFO("%s: CUDA backend missing entry point %s\n", __func__, GGML_MOE_STREAM_PROC_SETUP);
        return nullptr;
    }

    const int n_layer = (int) model.hparams.n_layer();

    // resolve the entry points before anything is allocated: any failure path has to be able to
    // unregister the engine before the arenas it points at go away
    void * v_teardown = proc(devs[0].reg, GGML_MOE_STREAM_PROC_TEARDOWN);
    void * v_stats    = proc(devs[0].reg, GGML_MOE_STREAM_PROC_STATS);
    void * v_reset    = proc(devs[0].reg, GGML_MOE_STREAM_PROC_RESET);
    void * v_begin    = proc(devs[0].reg, GGML_MOE_STREAM_PROC_BEGIN);
    if (!v_teardown || !v_begin || !v_stats || !v_reset) {
        LLAMA_LOG_WARN("%s: CUDA entry points present: teardown=%d begin=%d stats=%d reset=%d; streaming disabled\n",
                       __func__, v_teardown != nullptr, v_begin != nullptr, v_stats != nullptr, v_reset != nullptr);
        return nullptr;
    }

    // pass 1: classify every MoE layer as host-resident (streamable) or device-resident
    std::vector<llama_moe_plan::layer_src> plan_in(n_layer);
    std::vector<std::array<ggml_tensor *, 3>> orig(n_layer);
    std::vector<std::array<uint64_t, 3>>     lay_off(n_layer);   // arena offset of each family
    std::vector<std::array<uint64_t, 3>>     lay_pad(n_layer);
    int n_moe = 0;
    int n_eligible = 0;
    std::map<std::string, int> rejected;

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer & l = model.layers[il];

        llama_moe_plan::source_traits traits;
        traits.gate_input   = l.ffn_gate_inp != nullptr;
        traits.fused_gate_up = l.ffn_gate_up_exps != nullptr;
        traits.have_all     = true;
        traits.contiguous   = true;
        traits.scales       = l.ffn_up_exps_s || l.ffn_gate_exps_s || l.ffn_down_exps_s;
        traits.biases       = l.ffn_up_exps_b || l.ffn_gate_exps_b || l.ffn_down_exps_b;
        traits.device       = dev_ordinal(devs, il < (int) dev_of_layer.size() ? dev_of_layer[il] : nullptr);
        uint64_t fam[3] = { 0, 0, 0 };
        for (int f = 0; f < 3; ++f) {
            ggml_tensor * t = orig[il][f] = family_tensor(l, f);
            if (!t) {
                traits.have_all = false;
                continue;
            }
            if (llama_moe_stream_host_source(t)) { traits.host_mask |= 1u << f; }
            if (is_device_resident(t))          { traits.device_mask |= 1u << f; }
            if (t->buffer && t->data)           { traits.data_mask |= 1u << f; }
            traits.contiguous = traits.contiguous && ggml_is_contiguous(t);
            fam[f] = ggml_nbytes(t);
            // Placement can move the arena to any CUDA device. Reserve the largest backend span.
            for (const auto & c : devs) {
                fam[f] = std::max(fam[f], (uint64_t) ggml_backend_buft_get_alloc_size(ggml_backend_dev_buffer_type(c.dev), t));
            }
            lay_pad[il][f] = fam[f] - ggml_nbytes(t);
        }
        const bool is_moe = traits.gate_input || traits.fused_gate_up || orig[il][0] || orig[il][1] || orig[il][2];
        n_moe += is_moe;
        const char * rejection = llama_moe_plan::source_rejection(traits, LLAMA_MOE_STREAM_MAX_DEVICES);
        plan_in[il].il     = il;
        plan_in[il].device = traits.device;
        if (!rejection[0]) {
            plan_in[il].nbytes = llama_moe_plan::layout_families(fam, FN_SLOT_ALIGN, &lay_off[il][0]);
            ++n_eligible;
        } else if (is_moe) {
            ++rejected[rejection];
        }
        if (diagnostics && is_moe) {
            const ggml_backend_dev_t owner = model.dev_layer(il);
            LLAMA_LOG_INFO("moe_stream_diag: layer=%d owner=%s owner_type=%d cuda_owner=%d gate_input=%d "
                           "fused=%d families=%d scales=%d biases=%d contiguous=%d host_mask=%u device_mask=%u "
                           "data_mask=%u eligible=%d reason=%s plan_nbytes=%" PRIu64 "\n",
                           il, owner ? ggml_backend_dev_name(owner) : "(null)",
                           owner ? (int) ggml_backend_dev_type(owner) : -1, traits.device, traits.gate_input,
                           traits.fused_gate_up, traits.have_all, traits.scales, traits.biases, traits.contiguous,
                           traits.host_mask, traits.device_mask, traits.data_mask, !rejection[0],
                           rejection[0] ? rejection : "eligible", plan_in[il].nbytes);
            for (int f = 0; f < 3; ++f) {
                log_tensor(il, f == 0 ? "up" : f == 1 ? "gate" : "down", orig[il][f]);
            }
            if (l.ffn_gate_up_exps) { log_tensor(il, "gate_up", l.ffn_gate_up_exps); }
        }
    }

    LLAMA_LOG_INFO("%s: MoE source classification: total=%d eligible=%d rejected=%d\n",
                   __func__, n_moe, n_eligible, n_moe - n_eligible);
    for (const auto & entry : rejected) {
        LLAMA_LOG_INFO("%s: rejected %d layer(s): %s\n", __func__, entry.second, entry.first.c_str());
    }

    if (n_moe == 0) {
        LLAMA_LOG_INFO("%s: model has no MoE expert tensors\n", __func__);
        return nullptr;
    }

    // arena placement: the arena does not have to live on the device that owns the layer. concentrating
    // every arena on one device, or splitting the streamed byte total across devices by weight, changes
    // which PCIe link carries the image. selected at runtime (--fn-stream-gpu-mode) and ablated
    // independently; PLACE_LAYER (the default) reproduces the pre-existing behaviour exactly.
    int      n_remote     = 0;
    uint64_t bytes_remote = 0;
    {
        std::vector<int>      layer_dev(n_layer, -1);
        std::vector<uint64_t> lay_bytes(n_layer, 0);
        for (int il = 0; il < n_layer; ++il) {
            if (plan_in[il].nbytes) {
                layer_dev[il] = plan_in[il].device;
                lay_bytes[il] = plan_in[il].nbytes;
            }
        }

        std::string note;
        const std::vector<int> arena = llama_moe_plan::place_arena(
                layer_dev, lay_bytes, (int) devs.size(), params.place_mode, params.place_device,
                params.place_w, &note);

        for (int il = 0; il < n_layer; ++il) {
            if (!plan_in[il].nbytes) {
                continue;
            }
            if (arena[il] < 0 || arena[il] >= (int) devs.size() || arena[il] >= LLAMA_MOE_STREAM_MAX_DEVICES) {
                LLAMA_LOG_WARN("%s: invalid arena device %d for layer %d, streaming disabled\n",
                               __func__, arena[il], il);
                return nullptr;
            }
            if (arena[il] != plan_in[il].device) {
                ++n_remote;
                bytes_remote += plan_in[il].nbytes;
                plan_in[il].device = arena[il];
            }
            if (diagnostics) {
                LLAMA_LOG_INFO("moe_stream_diag: layer=%d plan_nbytes=%" PRIu64 " arena_device=%d\n",
                               il, plan_in[il].nbytes, arena[il]);
            }
        }
        if (!note.empty()) {
            LLAMA_LOG_INFO("%s: %s\n", __func__, note.c_str());
        }
    }

    // plan_in[].nbytes carries the padded image size, so one slot size fits the worst-case layout of
    // every streamed layer (the families of a layer differ in size from layer to layer here)
    size_t slot_bytes = 0;
    for (const auto & s : plan_in) {
        slot_bytes = std::max(slot_bytes, (size_t) s.nbytes);
    }
    if (slot_bytes == 0) {
        LLAMA_LOG_INFO("%s: no streamable expert image; see source rejection counts above\n", __func__);
        return nullptr;
    }

    llama_moe_stream * ms = new llama_moe_stream();
    ms->min_tokens = params.min_tokens;
    ms->proc_setup    = (decltype(ms->proc_setup)) v_setup;
    ms->proc_teardown = (decltype(ms->proc_teardown)) v_teardown;
    ms->proc_stats    = (decltype(ms->proc_stats)) v_stats;
    ms->proc_reset    = (decltype(ms->proc_reset)) v_reset;
    ms->proc_begin    = (decltype(ms->proc_begin)) v_begin;

    // size the ring against the smallest free VRAM across the devices that own streamed layers,
    // so the arena cannot be the reason an allocation fails later
    size_t budget = 0;
    {
        bool first = true;
        for (const auto & c : devs) {
            bool used = false;
            for (const auto & s : plan_in) {
                if (s.nbytes && s.device == c.ord) {
                    used = true;
                    break;
                }
            }
            if (!used) {
                continue;
            }
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(c.dev, &free_b, &total_b);
            // leave the requested headroom plus slack for the compute buffers the graphs still need
            const size_t avail = llama_moe_plan::arena_budget(params.budget_bytes, free_b, uint64_t(2048) << 20);
            LLAMA_LOG_INFO("%s: device %d free=%.1f MiB total=%.1f MiB budget=%.1f MiB (%s) slot_allocation=%.1f MiB\n",
                           __func__, c.ord, free_b / 1048576.0, total_b / 1048576.0, avail / 1048576.0,
                           params.budget_bytes ? "explicit" : "auto, 2048 MiB reserve", (slot_bytes + FN_SLOT_SLACK) / 1048576.0);
            budget = first ? avail : std::min(budget, avail);
            first = false;
        }
        if (first) {
            LLAMA_LOG_WARN("%s: no CUDA device owns a planned arena\n", __func__);
            delete ms;
            return nullptr;
        }
    }
    if (budget == 0) {
        LLAMA_LOG_WARN("%s: no VRAM available within the arena budget\n", __func__);
        delete ms;
        return nullptr;
    }

    llama_moe_plan::config pcfg;
    pcfg.n_slots        = params.n_slots;
    pcfg.hard_max_slots = params.hard_max_slots;
    pcfg.slot_bytes     = slot_bytes;
    pcfg.budget_bytes   = budget;
    pcfg.slot_overhead_bytes = FN_SLOT_SLACK;

    const llama_moe_plan::result plan = llama_moe_plan::plan(plan_in, pcfg);
    if (!plan.enabled) {
        LLAMA_LOG_WARN("%s: expert streaming disabled: %s\n", __func__, plan.reason.c_str());
        delete ms;
        return nullptr;
    }

    ms->enabled  = true;
    ms->n_slots  = plan.n_slots;
    ms->note     = plan.reason;
    ms->bind.resize(n_layer);   // value-initialized: every bind starts unassigned (slot < 0)
    ms->n_bytes_per_ubatch = plan.stream_bytes;
    ms->n_remote_layers    = n_remote;
    ms->n_bytes_remote     = bytes_remote;

    ggml_context * ctx_arena = nullptr;
    {
        struct ggml_init_params ip;
        ip.mem_size   = llama_moe_stream_descriptor_bytes(n_layer);
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;   // descriptors only; the bytes live in the arena buffers
        ctx_arena = ggml_init(ip);
        if (!ctx_arena) {
            LLAMA_LOG_WARN("%s: cannot allocate %zu bytes for arena descriptors\n", __func__, ip.mem_size);
            delete ms;
            return nullptr;
        }
    }
    ms->ctx_arena = ctx_arena;

    // allocate one arena buffer per (device, slot) and remember its device base pointer
    std::map<int, std::vector<ggml_backend_buffer_t>> bufs;   // device -> slot -> arena buffer
    std::map<int, std::vector<void *>> base;                  // device -> slot -> dev ptr
    for (const auto & c : devs) {
        bool used = false;
        for (const auto & s : plan_in) {
            if (s.nbytes && s.device == c.ord) {
                used = true;
                break;
            }
        }
        if (!used) {
            continue;
        }
        ms->slot_bytes[c.ord] = slot_bytes;

        std::vector<void *> bases;
        std::vector<ggml_backend_buffer_t> dev_bufs;
        for (int k = 0; k < plan.n_slots; ++k) {
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(
                    ggml_backend_dev_buffer_type(c.dev), slot_bytes + FN_SLOT_SLACK);
            if (!buf) {
                LLAMA_LOG_WARN("%s: device %d could not allocate arena slot %d of %zu MiB\n",
                               __func__, c.ord, k, slot_bytes >> 20);
                break;
            }
            // WEIGHTS usage is what keeps a remote arena honest: ggml schedules an op with weights on
            // the backend that holds them, so the expert MUL_MAT_ID follows the arena and the
            // activations cross the link, never the gigabyte-scale image itself
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ms->slot_buf.push_back(buf);
            dev_bufs.push_back(buf);
            bases.push_back(ggml_backend_buffer_get_base(buf));
        }
        if ((int) bases.size() < plan.n_slots) {
            // not enough slots to run the ring: fall back to the baseline instead of a partial engine
            llama_moe_stream_destroy(ms);
            return nullptr;
        }
        base[c.ord] = bases;
        bufs[c.ord] = dev_bufs;
    }

    // build the per-layer arena descriptors and the engine layer/slot tables
    std::map<int, std::vector<ggml_cuda_moe_stream_layer>> eng_layers;
    std::map<int, std::vector<ggml_cuda_moe_stream_slot>>  eng_slots;

    for (int il = 0; il < n_layer; ++il) {
        if (plan.slot[il] < 0) {
            continue;
        }
        const int dev  = plan_in[il].device;
        const int slot = plan.slot[il];

        llama_moe_stream_layer_bind & b = ms->bind[il];
        b.layer = il;
        b.slot  = slot;
        b.nbytes = (int64_t) plan_in[il].nbytes;

        ggml_cuda_moe_stream_layer el;
        std::memset(&el, 0, sizeof(el));
        el.layer = il;

        for (int f = 0; f < 3; ++f) {
            ggml_tensor * src = orig[il][f];
            ggml_tensor * a   = ggml_new_tensor_3d(ctx_arena, src->type, src->ne[0], src->ne[1], src->ne[2]);
            if (ggml_nbytes(a) != ggml_nbytes(src)) {
                LLAMA_LOG_WARN("%s: arena descriptor size mismatch at layer %d family %d\n", __func__, il, f);
                llama_moe_stream_destroy(ms);
                return nullptr;
            }
            void * addr = (char *) base[dev][slot] + lay_off[il][f];
            if (ggml_backend_tensor_alloc(bufs[dev][slot], a, addr) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_WARN("%s: cannot bind arena descriptor at layer %d family %d\n", __func__, il, f);
                llama_moe_stream_destroy(ms);
                return nullptr;
            }
            b.arena[f]    = a;
            el.host[f]    = src->data;
            el.nbytes[f]  = ggml_nbytes(src);
            el.dst_off[f] = lay_off[il][f];
            el.pad_bytes[f] = lay_pad[il][f];
            if (diagnostics && el.pad_bytes[f]) {
                LLAMA_LOG_INFO("moe_stream_diag: layer=%d family=%d cuda_padding=%zu bytes\n", il, f, el.pad_bytes[f]);
            }
        }
        eng_layers[dev].push_back(el);

        const int kick = plan.post_kick[il];
        if (kick >= 0) {
            b.kick_layer = plan_in[kick].il;
            b.kick_slot  = plan.slot[kick];
        }
    }

    for (const auto & kv : eng_layers) {
        const int dev = kv.first;
        std::vector<ggml_cuda_moe_stream_slot> & slots = eng_slots[dev];
        for (int k = 0; k < plan.n_slots; ++k) {
            ggml_cuda_moe_stream_slot s;
            std::memset(&s, 0, sizeof(s));
            s.slot = k;
            s.dev  = base[dev][k];
            s.cap  = slot_bytes;
            for (int f = 0; f < 3; ++f) {
                // slot capacity is advertised as the largest family footprint any layer needs there
                size_t mx = 0;
                for (const auto & l : kv.second) {
                    mx = std::max(mx, l.nbytes[f]);
                }
                s.nbytes[f] = mx;
            }
            slots.push_back(s);
        }

        ggml_cuda_moe_stream_params ep;
        std::memset(&ep, 0, sizeof(ep));
        ep.n_slots   = plan.n_slots;
        ep.pin_host  = params.pin_host;
        ep.pin_bytes = 0;
        ep.device    = dev;

        auto setup = (decltype(ms->proc_setup)) v_setup;
        const int rc = setup(&ep, kv.second.data(), (int) kv.second.size(),
                             slots.data(), (int) slots.size());
        if (rc != 0) {
            LLAMA_LOG_WARN("%s: streamed expert engine rejected device %d (rc=%d)\n", __func__, dev, rc);
            llama_moe_stream_destroy(ms);
            return nullptr;
        }

        if (std::find(ms->devices.begin(), ms->devices.end(), dev) == ms->devices.end()) {
            ms->devices.push_back(dev);
        }
        ms->registered = true;
    }
    ms->device = ms->devices.empty() ? -1 : ms->devices.front();

    for (int pos : plan.bootstrap) {
        const int dev = plan_in[pos].device;
        ms->boot_slot[dev].push_back(plan.slot[pos]);
        ms->boot_layer[dev].push_back(plan_in[pos].il);
    }

    LLAMA_LOG_INFO("%s: streamed experts armed: %d layer(s), %d slot(s) of %.1f MiB, %zu MiB cross "
                   "PCIe per micro-batch (min %d tokens), %d layer(s) remote\n", __func__,
                   plan.n_streamed, plan.n_slots, slot_bytes / 1048576.0,
                   plan.stream_bytes >> 20, (int) params.min_tokens, n_remote);
    if (bytes_remote) {
        LLAMA_LOG_INFO("%s: %zu MiB per micro-batch is read on a device other than the layer owner, "
                       "activation hops are additional\n", __func__, bytes_remote >> 20);
    }
    LLAMA_LOG_INFO("%s: bytes per streamed token = slot image / micro-batch tokens, so cold prefill "
                   "throughput scales directly with -ub\n", __func__);
    return ms;
}

void llama_moe_stream_destroy(llama_moe_stream * ms) {
    if (!ms) {
        return;
    }
    if (!ms->registered) {
        // nothing was set up, so there is no worker that could still touch the arenas
        ms->proc_teardown = nullptr;
    } else if (ms->proc_teardown && !ms->destroyed) {
        // every engine owns a worker thread that can still write into its arenas, so all of them have to
        // be stopped and joined before the first arena buffer is freed. tearing down only one device
        // would leak its worker and its pinned mirrors and leave a writer aimed at freed device memory
        for (const int dev : ms->devices) {
            ms->proc_teardown(dev);
        }
        ms->destroyed = true;
    }
    for (auto buf : ms->slot_buf) {
        ggml_backend_buffer_free(buf);
    }
    ms->slot_buf.clear();
    if (ms->ctx_arena) {
        ggml_free(ms->ctx_arena);
        ms->ctx_arena = nullptr;
    }
    delete ms;
}

bool llama_moe_stream_enabled(const llama_moe_stream * ms) {
    return ms && ms->enabled && ms->proc_begin;
}

const llama_moe_stream * llama_moe_stream_select(const llama_moe_stream * ms, int64_t n_tokens) {
    if (!llama_moe_stream_enabled(ms)) {
        return nullptr;
    }
    if (n_tokens < ms->min_tokens) {
        return nullptr;
    }
    return ms;
}

bool llama_moe_stream_bind(const llama_moe_stream * ms, int il, llama_moe_stream_layer_bind & out) {
    if (!ms || !ms->enabled || il < 0 || il >= (int) ms->bind.size()) {
        return false;
    }
    const llama_moe_stream_layer_bind & b = ms->bind[il];
    if (b.slot < 0 || !b.arena[0] || !b.arena[1] || !b.arena[2]) {
        return false;
    }
    out = b;
    return true;
}

void llama_moe_stream_arm(llama_moe_stream * ms, int64_t n_tokens) {
    if (llama_moe_stream_select(ms, n_tokens) == nullptr || ms->armed) {
        return;
    }
    ms->armed = true;
    // one ring per device: priming only a single device leaves every other device's first pre-fence
    // waiting for a fill that nobody ever queued, which ends in the submit timeout, not in a fallback
    for (const int dev : ms->devices) {
        const auto & sl = ms->boot_slot[dev];
        const auto & ly = ms->boot_layer[dev];
        if (!sl.empty()) {
            ms->proc_begin(dev, sl.data(), ly.data(), (int) sl.size());
        }
    }
}

void llama_moe_stream_report(llama_moe_stream * ms, int64_t n_tokens, double t_ms) {
    if (!llama_moe_stream_enabled(ms) || !ms->armed) {
        return;
    }
    ms->armed = false;
    ms->n_ubatch_streamed++;
    ms->n_tokens_streamed += (uint64_t) n_tokens;

    uint64_t fill = 0, bytes = 0, stage = 0, pre = 0, post = 0, errs = 0;
    uint64_t wait_ns = 0, stage_ns = 0, submit_ns = 0;
    {
        // the counters live per engine: read and reset all of them, otherwise a multi-device placement
        // reports the traffic of one link and every bandwidth conclusion drawn from it is wrong
        for (const int d : ms->devices) {
            ggml_cuda_moe_stream_stats st;
            std::memset(&st, 0, sizeof(st));
            if (ms->proc_stats) {
                ms->proc_stats(d, &st);
            }
            fill      += st.n_fill;
            bytes     += st.n_fill_bytes;
            stage     += st.n_stage_bytes;
            pre       += st.n_fence_pre;
            post      += st.n_fence_post;
            errs      += st.n_error;
            wait_ns   += st.n_slot_wait_ns;
            stage_ns  += st.n_stage_ns;
            submit_ns += st.n_copy_submit_ns;
            if (ms->proc_reset) {
                ms->proc_reset(d);
            }
        }
    }

    const double gb = bytes / 1e9;
    LLAMA_LOG_INFO("%s: ubatch %d tokens in %.1f ms | expert fills %llu | H2D %.2f GB | stage %.2f GB "
                   "| fences %llu/%llu | slot wait %.1f ms | stage %.1f ms | submit %.1f ms | "
                   "errors %llu | devices %d\n",
                   __func__, (int) n_tokens, t_ms,
                   (unsigned long long) fill, gb, stage / 1e9,
                   (unsigned long long) pre, (unsigned long long) post,
                   wait_ns / 1e6, stage_ns / 1e6, submit_ns / 1e6, (unsigned long long) errs,
                   (int) ms->devices.size());
    if (bytes > 0) {
        LLAMA_LOG_INFO("%s: achieved H2D %.2f GB/s over %.1f ms of micro-batch time\n",
                       __func__, gb / (t_ms / 1000.0), t_ms);
    }
}
