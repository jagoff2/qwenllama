// Streamed MoE expert prefill engine - CUDA implementation.
//
// Contract (see ggml/include/ggml-moe-stream.h):
//   * host RAM holds the authoritative expert quant weights of the MoE layers that do not fit in VRAM
//   * a ring of `n_slots` device arena slots mirrors one layer image each, byte for byte
//   * a fill copies one layer image host -> slot on a dedicated copy stream
//   * GGML_OP_MOE_STREAM_FENCE orders the fills against the compute stream:
//       phase 0 (pre)  : make the compute stream wait for the fill that put `layer` into `slot`
//       phase 1 (post) : record that the slot is free and queue the fill of `kick_layer` into `kick_slot`
//
// Ordering rules enforced here:
//   * a fill never starts before the compute that read that slot finished (cudaEventQuery on ev_used)
//   * the pre-fence waits on the host until the fill it needs has actually been *submitted* to the copy
//     stream, because cudaStreamWaitEvent captures the event state at call time
//   * a failed fill poisons the engine and the next pre-fence aborts instead of reading a torn slot
//
// Everything is inert when the engine is not active, so the disabled path is the stock code path.

#include "common.cuh"
#include "moe-stream.cuh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#define FN_MOE_MAX_DEVICES 16

// how long a pre-fence tolerates a missing submission before declaring the pipeline wedged
#define FN_MOE_SUBMIT_TIMEOUT_MS 120000

// staging granularity: one expert family is cut into <= FN_MOE_STAGE_CHUNKS chunks of at least
// FN_MOE_STAGE_CHUNK_MIN bytes so the stage copy and the PCIe DMA overlap
#define FN_MOE_STAGE_CHUNKS      8
#define FN_MOE_STAGE_CHUNK_MIN   (32u << 20)
#define FN_MOE_STAGE_CHUNK_MAX   (128u << 20)

namespace {

uint64_t fn_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct fn_slot {
    int32_t idx = -1;
    bool    set = false;

    char * dev = nullptr;   // device arena base
    char * pin = nullptr;   // pinned staging mirror of the same bytes (nullptr => copy straight from host)
    size_t cap = 0;         // slot capacity in bytes
    size_t total = 0;       // bytes of the image currently assigned to the slot

    cudaEvent_t ev_filled = nullptr;   // recorded on the copy stream once a fill was submitted
    cudaEvent_t ev_used   = nullptr;   // recorded on the compute stream at the post-fence

    int  held_layer      = -1;   // layer currently valid in dev, -1 == nothing
    int  submitted_layer = -1;   // layer whose fill has been submitted to the copy stream
    int  queued_layer    = -1;   // layer the pending job will write, kept until it is published
    bool pending         = false; // a fill is queued or in flight for this slot

    // Release accounting. cudaEventQuery(ev_used) on its own is not a safe gate: the event reports the
    // state of the most recent record, so a second fill for one slot would see the *previous* read's
    // completed event and start copying while the compute stream is still reading. Each read therefore
    // grants one credit at its post-fence and each copy consumes one before it writes: a copy may start
    // only once a release for the bytes it is about to replace exists, and cudaEventQuery then proves
    // that release really completed on the device. A fresh slot carries one credit because nothing has
    // ever read it.
    uint64_t credits = 1;   // releases granted since the last copy, i.e. how many copies may start
    uint64_t n_fill  = 0;   // fills published into this slot
    uint64_t n_bytes = 0;
};

struct fn_job {
    int slot;
    int layer;
};

struct fn_engine {
    int  device   = -1;
    bool active   = false;
    bool pin_host = false;
    bool poisoned = false;

    std::vector<fn_slot> slots;
    std::map<int, ggml_cuda_moe_stream_layer> layers;   // layer -> host sources

    cudaStream_t copy_stream = nullptr;

    std::thread             worker;
    std::mutex              m;
    std::condition_variable cv_fill;   // signalled when a slot's submission state changes
    // one queue per slot, not one shared FIFO: a fill that is waiting for a reader must never hold up a
    // fill of a different slot, and per-slot queues make a duplicate fill trivially detectable
    std::vector<std::vector<fn_job>> q_slot;
    int                     rr = 0;    // round-robin start so no slot starves
    bool                    stop = false;

    ggml_cuda_moe_stream_stats st;
};

fn_engine *                 g_eng[FN_MOE_MAX_DEVICES] = {};
std::mutex                  g_eng_mtx;

fn_engine * engine_get(int device) {
    if (device < 0 || device >= FN_MOE_MAX_DEVICES) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_eng_mtx);
    return g_eng[device];
}

void engine_poison(fn_engine * e, const char * what, cudaError_t err) {
    std::lock_guard<std::mutex> lock(e->m);
    if (!e->poisoned) {
        e->poisoned = true;
    }
    e->st.n_error++;
    GGML_LOG_ERROR("%s: %s (%s) - disabling streamed experts\n", __func__, what,
                   err != cudaSuccess ? cudaGetErrorName(err) : "logic");
}

// split a large memcpy across threads: the stage copy is pure DDR traffic and one core cannot
// saturate the memory subsystem on its own
void fn_parallel_memcpy(void * dst, const void * src, size_t n) {
    if (n == 0) {
        return;
    }
    const size_t chunk_min = 8u << 20;
    if (n < chunk_min * 2) {
        std::memcpy(dst, src, n);
        return;
    }
    unsigned nt = std::thread::hardware_concurrency();
    if (nt > 8) {
        nt = 8;
    }
    if (nt < 2) {
        std::memcpy(dst, src, n);
        return;
    }
    const size_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (unsigned i = 0; i < nt; ++i) {
        const size_t off = (size_t) i * chunk;
        if (off >= n) {
            break;
        }
        const size_t len = std::min(chunk, n - off);
        ts.emplace_back([dst, src, off, len] {
            std::memcpy((char *) dst + off, (const char *) src + off, len);
        });
    }
    for (auto & t : ts) {
        t.join();
    }
}

// the worker owns: wait for slot release -> optionally stage into pinned memory -> async H2D -> record
void worker_main(fn_engine * e) {
    if (cudaSetDevice(e->device) != cudaSuccess) {
        engine_poison(e, "worker cannot select device", cudaErrorInvalidDevice);
        return;
    }

    const auto queued = [e] {
        for (const auto & q : e->q_slot) {
            if (!q.empty()) {
                return true;
            }
        }
        return false;
    };

    for (;;) {
        fn_job job;
        ggml_cuda_moe_stream_layer ldesc;
        fn_slot * sp = nullptr;

        {
            std::unique_lock<std::mutex> lock(e->m);
            e->cv_fill.wait(lock, [&] { return e->stop || queued(); });
            if (e->stop) {
                return;
            }

            // a slot may be written once a read of it has been released (credit) - a fill that only
            // re-publishes an image the slot already holds needs no credit because it writes nothing.
            // a slot without credit stays queued while the others are served, so one waiting reader can
            // never hold up the whole device
            int pick = -1;
            const int n = (int) e->q_slot.size();
            for (int k = 0; k < n; ++k) {
                const int d = (e->rr + k) % n;
                if (e->q_slot[d].empty()) {
                    continue;
                }
                const fn_slot & c = e->slots[d];
                if (c.credits > 0 || c.held_layer == e->q_slot[d].front().layer) {
                    pick = d;
                    break;
                }
            }
            if (pick < 0) {
                // every queued fill is aimed at a slot a reader is still live on
                e->cv_fill.wait_for(lock, std::chrono::microseconds(200), [&] { return e->stop; });
                continue;
            }
            e->rr = (pick + 1) % n;
            job = e->q_slot[pick].front();
            e->q_slot[pick].erase(e->q_slot[pick].begin());

            if (job.slot < 0 || job.slot >= (int) e->slots.size()) {
                e->st.n_error++;
                continue;
            }
            sp = &e->slots[job.slot];

            if (sp->held_layer == job.layer) {
                // a re-armed ring can ask for an image the slot already holds: publish it as ready
                // without moving a byte. this is also what keeps a redundant fill from consuming the
                // credit that the next real fill needs
                sp->submitted_layer = job.layer;
                sp->pending         = false;
                sp->queued_layer    = -1;
                e->st.n_worker_skip++;
                e->cv_fill.notify_all();
                continue;
            }

            auto it = e->layers.find(job.layer);
            if (it == e->layers.end()) {
                e->st.n_error++;
                sp->pending = false;
                continue;
            }
            ldesc = it->second;
            ++e->st.n_worker_wake;
        }

        // 1. the slot must no longer be read by compute. the credit picked above proved that a release
        //    for these bytes exists; the event proves that the release completed on the device
        const uint64_t t_wait0 = fn_now_ns();
        bool ready = false;
        while (!ready) {
            const cudaError_t q = cudaEventQuery(sp->ev_used);
            if (q == cudaSuccess) {
                ready = true;
                break;
            }
            if (q != cudaErrorNotReady) {
                engine_poison(e, "cudaEventQuery(ev_used) failed", q);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(e->m);
                if (e->stop) {
                    sp->pending       = false;
                    sp->queued_layer  = -1;
                    e->cv_fill.notify_all();
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        {
            std::lock_guard<std::mutex> lock(e->m);
            e->st.n_slot_wait_ns += fn_now_ns() - t_wait0;
        }

        {
            std::lock_guard<std::mutex> lock(e->m);
            if (sp->credits > 0) {
                sp->credits--;
            }
        }

        // 2. host -> device, either straight from the authoritative pointer or via the pinned mirror.
        //    The destination offsets come from the caller-supplied layout (llama.cpp bound the arena
        //    descriptors with the same offsets), so a raw byte copy lands exactly where the descriptors
        //    point. The pinned mirror uses the same offsets: it mirrors the slot byte for byte.
        const uint64_t t_sub0 = fn_now_ns();
        bool ok = true;
        size_t need = 0;
        for (int f = 0; f < GGML_MOE_STREAM_MAX_FAMILIES; ++f) {
            need += ldesc.nbytes[f];
        }
        if (need == 0 || need > sp->cap) {
            engine_poison(e, "layer image does not fit the arena slot", cudaErrorInvalidValue);

            std::lock_guard<std::mutex> lock(e->m);
            sp->pending         = false;
            sp->submitted_layer = -2;
            sp->queued_layer    = -1;
            e->cv_fill.notify_all();
            continue;
        }
        for (int f = 0; f < GGML_MOE_STREAM_MAX_FAMILIES; ++f) {
            const size_t n = ldesc.nbytes[f];
            if (n == 0) {
                continue;
            }
            const char * src = (const char *) ldesc.host[f];
            if (src == nullptr) {
                engine_poison(e, "layer family has no host data", cudaErrorInvalidValue);
                ok = false;
                break;
            }
            const size_t dst = ldesc.dst_off[f];
            const size_t pad = ldesc.pad_bytes[f];
            if (dst > sp->cap || n > sp->cap - dst || pad > sp->cap - dst - n) {
                engine_poison(e, "family range falls outside the arena slot", cudaErrorInvalidValue);
                ok = false;
                break;
            }
            // A shorter image reuses bytes from the previous layer. Restore CUDA's zero tail.
            if (pad) {
                const cudaError_t err = cudaMemsetAsync(sp->dev + dst + n, 0, pad, e->copy_stream);
                if (err != cudaSuccess) {
                    engine_poison(e, "cannot clear expert tensor padding", err);
                    ok = false;
                    break;
                }
            }
            if (sp->pin == nullptr) {
                const cudaError_t err = cudaMemcpyAsync(sp->dev + dst, src, n,
                                                        cudaMemcpyHostToDevice, e->copy_stream);
                if (err != cudaSuccess) {
                    engine_poison(e, "cudaMemcpyAsync H2D failed", err);
                    ok = false;
                    break;
                }
                continue;
            }
            // cut the family into chunks: the DDR staging copy of the next chunk runs while the copy
            // engine is still transferring the previous one. without this the fill is stage+DMA
            // serialized and throughput drops to 1/(1/bw_stage + 1/bw_dma)
            size_t pos = 0;
            while (pos < n) {
                size_t len = n - pos;
                if (len > FN_MOE_STAGE_CHUNK_MAX) {
                    size_t nc = (n + FN_MOE_STAGE_CHUNKS - 1) / FN_MOE_STAGE_CHUNKS;
                    if (nc < FN_MOE_STAGE_CHUNK_MIN) {
                        nc = FN_MOE_STAGE_CHUNK_MIN;
                    }
                    len = std::min(len, nc);
                }
                const uint64_t t_stg0 = fn_now_ns();
                fn_parallel_memcpy(sp->pin + dst + pos, src + pos, len);
                {
                    std::lock_guard<std::mutex> lock(e->m);
                    e->st.n_stage_bytes += len;
                    e->st.n_stage_ns    += fn_now_ns() - t_stg0;
                }
                const cudaError_t err = cudaMemcpyAsync(sp->dev + dst + pos, sp->pin + dst + pos, len,
                                                        cudaMemcpyHostToDevice, e->copy_stream);
                if (err != cudaSuccess) {
                    engine_poison(e, "cudaMemcpyAsync H2D failed", err);
                    ok = false;
                    break;
                }
                pos += len;
            }
            if (!ok) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(e->m);
            e->st.n_copy_submit_ns += fn_now_ns() - t_sub0;
        }

        if (!ok) {
            std::lock_guard<std::mutex> lock(e->m);
            sp->pending         = false;
            sp->submitted_layer = -2;   // a pre-fence for this layer must abort, not proceed
            sp->queued_layer    = -1;
            e->cv_fill.notify_all();
            continue;
        }

        // 3. publish: the event state captured by a later cudaStreamWaitEvent now includes this copy
        const cudaError_t err = cudaEventRecord(sp->ev_filled, e->copy_stream);
        if (err != cudaSuccess) {
            engine_poison(e, "cudaEventRecord(ev_filled) failed", err);
        }

        {
            std::lock_guard<std::mutex> lock(e->m);
            sp->submitted_layer = job.layer;
            sp->held_layer      = job.layer;
            sp->total           = need;
            sp->pending         = false;
            sp->queued_layer    = -1;
            sp->n_fill++;
            e->st.n_fill++;
            e->st.n_fill_bytes += need;
            if (sp->pin == nullptr) {
                e->st.n_stage_bytes += need;   // no staging: the host pointer went straight to DMA
            }
        }
        e->cv_fill.notify_all();
    }
}

void engine_release(fn_engine * e) {
    if (!e) {
        return;
    }
    cudaSetDevice(e->device);
    {
        std::lock_guard<std::mutex> lock(e->m);
        e->stop = true;
    }
    e->cv_fill.notify_all();
    if (e->worker.joinable()) {
        e->worker.join();
    }
    for (auto & s : e->slots) {
        if (s.ev_filled) {
            cudaEventDestroy(s.ev_filled);
        }
        if (s.ev_used) {
            cudaEventDestroy(s.ev_used);
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
        }
        s = fn_slot{};
    }
    if (e->copy_stream) {
        cudaStreamDestroy(e->copy_stream);
    }
    delete e;
}

bool queue_fill(fn_engine * e, int slot, int layer) {
    std::lock_guard<std::mutex> lock(e->m);
    if (e->poisoned || e->stop) {
        return false;
    }
    if (slot < 0 || slot >= (int) e->slots.size()) {
        e->st.n_error++;
        return false;
    }
    if (e->layers.find(layer) == e->layers.end()) {
        e->st.n_error++;
        return false;
    }
    fn_slot & s = e->slots[slot];
    if (s.held_layer == layer && !s.pending) {
        // the slot already carries exactly what is needed, no bytes to move
        e->st.n_worker_skip++;
        return true;
    }
    if (s.pending) {
        // re-arming the ring at the start of the next micro-batch overlaps a fill that is still queued or
        // in flight. the same image again is a no-op worth suppressing; a different image means the ring
        // and the plan disagree, and no release accounting can say which reader the second copy would
        // race, so report the error instead of choosing between two unsafe behaviours
        if (s.queued_layer == layer) {
            e->st.n_worker_skip++;
            return true;
        }
        for (const fn_job & j : e->q_slot[slot]) {
            if (j.layer == layer) {
                e->st.n_worker_skip++;
                return true;
            }
        }
        e->st.n_error++;
        return false;
    }
    s.submitted_layer = -1;
    s.held_layer      = -1;
    s.pending         = true;
    s.queued_layer    = layer;
    e->q_slot[slot].push_back(fn_job{ slot, layer });
    e->cv_fill.notify_all();
    return true;
}

} // namespace

void ggml_cuda_op_moe_stream_fence(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t * p     = (const int32_t *) dst->op_params;
    const int       layer = p[0];
    const int       phase = p[1];
    const int       slot  = p[2];

    fn_engine * e = engine_get(ctx.device);
    if (e == nullptr) {
        // the op is only emitted when the engine was active at graph build time
        GGML_ABORT("%s: streamed expert engine is not configured for device %d (layer %d)",
                   __func__, ctx.device, layer);
    }

    cudaStream_t stream = ctx.stream();

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        // a fence cannot be captured: the fills are produced by a host worker, not by the graph
        GGML_ABORT("%s: streamed expert fences must not run inside CUDA graph capture", __func__);
    }

    if (slot < 0 || slot >= (int) e->slots.size()) {
        engine_poison(e, "fence references an unknown slot", cudaErrorInvalidValue);
        GGML_ABORT("%s: invalid expert arena slot %d", __func__, slot);
    }

    fn_slot * sp = &e->slots[slot];

    if (phase == 0) {
        // wait until the fill that carries `layer` has been submitted to the copy stream, then order
        // the compute stream behind it
        const uint64_t t0 = fn_now_ns();
        for (;;) {
            bool   done = false;
            bool   dead = false;
            {
                std::unique_lock<std::mutex> lock(e->m);
                if (e->poisoned) {
                    dead = true;
                } else if (sp->submitted_layer == layer) {
                    done = true;
                } else if (sp->submitted_layer == -2) {
                    dead = true;
                } else {
                    e->cv_fill.wait_for(lock, std::chrono::milliseconds(50));
                }
            }
            if (dead) {
                GGML_ABORT("%s: streamed expert pipeline failed before layer %d; refusing to read a "
                           "possibly torn expert slot", __func__, layer);
            }
            if (done) {
                break;
            }
            if ((fn_now_ns() - t0) / 1000000 > FN_MOE_SUBMIT_TIMEOUT_MS) {
                GGML_ABORT("%s: timed out waiting for expert slot %d (layer %d)", __func__, slot, layer);
            }
        }

        const cudaError_t err = cudaStreamWaitEvent(stream, sp->ev_filled, 0);
        if (err != cudaSuccess) {
            engine_poison(e, "cudaStreamWaitEvent(ev_filled) failed", err);
            GGML_ABORT("%s: cannot order expert slot %d: %s", __func__, slot, cudaGetErrorName(err));
        }
        {
            std::lock_guard<std::mutex> lock(e->m);
            e->st.n_fence_pre++;
        }
        return;
    }

    // post: this slot is no longer read by compute once the stream gets here
    const cudaError_t err = cudaEventRecord(sp->ev_used, stream);
    if (err != cudaSuccess) {
        engine_poison(e, "cudaEventRecord(ev_used) failed", err);
        GGML_ABORT("%s: cannot release expert slot %d: %s", __func__, slot, cudaGetErrorName(err));
    }
    {
        std::lock_guard<std::mutex> lock(e->m);
        // the record above must be issued before the credit becomes visible, otherwise a worker could pick
        // this slot while the event still reports the previous, already completed read
        sp->credits++;
        e->st.n_fence_post++;
    }

    const int kick_slot  = p[3];
    const int kick_layer = p[4];
    if (kick_layer >= 0) {
        if (!queue_fill(e, kick_slot, kick_layer)) {
            // leave the pipeline alive but do not silently lose the fill: the matching pre-fence for
            // kick_layer will not find a submission and will abort with a precise message
            engine_poison(e, "cannot queue expert fill", cudaErrorUnknown);
        }
    }
}

int ggml_cuda_moe_stream_setup_impl(const struct ggml_cuda_moe_stream_params * params,
                                    const struct ggml_cuda_moe_stream_layer  * layers, int n_layers,
                                    const struct ggml_cuda_moe_stream_slot   * slots,  int n_slots) {
    if (params == nullptr || slots == nullptr || n_slots <= 0 || params->device < 0 ||
        params->device >= FN_MOE_MAX_DEVICES) {
        return -1;
    }

    std::unique_lock<std::mutex> lock(g_eng_mtx);
    fn_engine * prev = g_eng[params->device];
    if (prev) {
        // replacing a live engine: tear it down first so its worker cannot touch freed arenas
        g_eng[params->device] = nullptr;
    }
    lock.unlock();

    if (prev) {
        engine_release(prev);
    }

    fn_engine * e = new fn_engine();
    e->device   = params->device;
    e->pin_host = params->pin_host != 0;

    if (cudaSetDevice(e->device) != cudaSuccess) {
        delete e;
        return -2;
    }

    if (cudaStreamCreateWithFlags(&e->copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete e;
        return -3;
    }

    e->slots.resize(n_slots);
    e->q_slot.assign(n_slots, std::vector<fn_job>());
    for (int i = 0; i < n_slots; ++i) {
        const ggml_cuda_moe_stream_slot & in = slots[i];
        fn_slot & s = e->slots[i];
        s.idx = in.slot;
        s.dev = (char *) in.dev;
        s.set = true;
        if (s.dev == nullptr) {
            engine_poison(e, "slot with null device base", cudaErrorInvalidValue);
            engine_release(e);
            return -4;
        }
        const size_t cap = in.cap;
        if (cap == 0) {
            engine_poison(e, "arena slot with zero capacity", cudaErrorInvalidValue);
            engine_release(e);
            return -4;
        }
        s.cap   = cap;
        s.total = cap;
        if (cudaEventCreateWithFlags(&s.ev_filled, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.ev_used,   cudaEventDisableTiming) != cudaSuccess) {
            engine_poison(e, "cannot create events", cudaErrorMemoryAllocation);
            engine_release(e);
            return -5;
        }
        if (e->pin_host && cap > 0) {
            if (cudaHostAlloc((void **) &s.pin, cap, cudaHostAllocDefault) != cudaSuccess) {
                // running without the pinned mirror is correct, only slower: degrade, do not fail
                GGML_LOG_WARN("%s: pinned staging for slot %d (%.1f MiB) failed, falling back to "
                              "direct copies from pageable host memory\n", __func__, i, cap / 1048576.0);
                s.pin = nullptr;
                e->pin_host = false;
            }
        }
    }

    for (int i = 0; i < n_layers; ++i) {
        e->layers[layers[i].layer] = layers[i];
    }
    if (e->layers.empty()) {
        engine_release(e);
        return -6;
    }

    e->active = true;
    e->worker = std::thread(worker_main, e);

    g_eng_mtx.lock();
    g_eng[e->device] = e;
    g_eng_mtx.unlock();

    GGML_LOG_INFO("%s: device %d armed with %d expert arena slot(s), %.1f MiB each, pinned staging %s\n",
                  __func__, e->device, n_slots, e->slots[0].total / 1048576.0,
                  e->pin_host ? "on" : "off");
    return 0;
}

void ggml_cuda_moe_stream_teardown_impl(int device) {
    fn_engine * e = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_eng_mtx);
        if (device >= 0 && device < FN_MOE_MAX_DEVICES) {
            e = g_eng[device];
            g_eng[device] = nullptr;
        }
    }
    engine_release(e);
}

void ggml_cuda_moe_stream_stats_impl(int device, struct ggml_cuda_moe_stream_stats * out) {
    if (out == nullptr) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    fn_engine * e = engine_get(device);
    if (e == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->m);
    *out = e->st;
}

void ggml_cuda_moe_stream_reset_impl(int device) {
    fn_engine * e = engine_get(device);
    if (e == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->m);
    e->st = ggml_cuda_moe_stream_stats{};
}

int ggml_cuda_moe_stream_active_impl(int device) {
    fn_engine * e = engine_get(device);
    return (e != nullptr && e->active) ? 1 : 0;
}

void ggml_cuda_moe_stream_begin_impl(int device, const int32_t * slots, const int32_t * layers, int n) {
    fn_engine * e = engine_get(device);
    if (e == nullptr || !e->active) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        queue_fill(e, slots[i], layers[i]);
    }
}

void ggml_cuda_moe_stream_end_of_graph(int device) {
    // nothing per-graph to retire: the ring state is owned by the engine, not by the graph
    GGML_UNUSED(device);
}
