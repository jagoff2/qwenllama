// Streamed MoE expert ring: protocol safety + analytical timing model.
//
// The CUDA engine (ggml/src/ggml-cuda/moe-stream.cu) and the graph fences form a small
// producer/consumer protocol on top of the slot plan (src/llama-moe-stream-plan.h):
//
//   fill(m)      : host image of streamed ordinal m -> slot (m mod S), on a copy stream
//   pre-fence(m) : compute stream waits until fill(m) has been submitted (and, on the device, until
//                  the copy itself is complete)
//   post-fence(m): records that slot (m mod S) is no longer read by the compute stream, queues fill
//                  of ordinal m + S into that same slot
//
// This test replays that protocol in software against the real planner output and checks the
// properties that matter. No GPU, no model, no ggml: it runs anywhere.
//
//   safety   a slot is never written while the graph is still reading it, and every read observes
//            exactly the layer the pre-fence waited for
//   liveness a missing fill is detected as a wedge, never silently read as data
//   traffic  exactly one expert image per streamed layer crosses PCIe per micro-batch
//   model    the same recurrence with this box's bandwidths predicts prefill throughput
//
// ref: bench/flashnext-stream-prefill/RESULTS.md (performance model)

#include "llama-moe-stream-plan.h"
#include "ggml-moe-stream.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
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

//
// abstract replay of the protocol
//
// slot state mirrors the engine:
//   content  - layer image currently valid in the slot
//   filled   - layer whose fill has completed (-1 when the slot no longer holds a usable image)
//   busy     - the compute stream is between the pre-fence and the post-fence of a layer, i.e. the
//              slot is being read and must not be written
//   held     - layer being read while busy
//

struct sim_state {
    std::vector<int>  content;
    std::vector<int>  filled;
    std::vector<bool> busy;
    std::vector<int>  held;
    std::vector<std::pair<int, int>> queue;   // (slot, layer) fills waiting for a release
    int      n_fills   = 0;
    int      n_skips   = 0;
    int      torn      = 0;   // a fill started while the slot was being read
    int      stale     = 0;   // a read observed a layer other than the requested one
    int      wedged    = 0;   // a pre-fence could not be satisfied
    uint64_t bytes     = 0;
    bool     ignore_release = false;   // negative control only
};

// the worker may start a fill only once the previous reader of the slot is done (cudaEventQuery on
// the release event). completing the fill publishes the image
static void sim_fill_if_ready(sim_state & st, const std::vector<uint64_t> & nbytes_of) {
    for (size_t i = 0; i < st.queue.size(); ++i) {
        const int s = st.queue[i].first;
        const int l = st.queue[i].second;

        if (st.busy[s]) {
            if (!st.ignore_release) {
                continue;   // still waiting for the release event
            }
            ++st.torn;      // the negative control writes over a live read
        }

        st.queue.erase(st.queue.begin() + i);
        st.filled[s]  = l;
        st.content[s] = l;
        ++st.n_fills;
        st.bytes += nbytes_of[l];
        return;   // one worker thread: one fill at a time
    }
}

// enqueue a fill unless the slot already holds exactly that image
static void sim_queue(sim_state & st, int slot, int layer) {
    if (st.content[slot] == layer && !st.busy[slot]) {
        ++st.n_skips;
        st.filled[slot] = layer;
        return;
    }
    st.filled[slot] = -1;
    st.queue.push_back({ slot, layer });
}

// returns false when the micro-batch wedged
static bool sim_microbatch(sim_state & st, const std::vector<int> & slot_of,
                           const std::vector<int> & boot, const std::vector<int> & kick,
                           const std::vector<uint64_t> & nbytes_of) {
    const int n = (int) slot_of.size();

    // bootstrap: the first min(S, n) ordinals have no predecessor release point
    for (int t : boot) {
        sim_queue(st, slot_of[t], t);
    }

    for (int t = 0; t < n; ++t) {
        const int s = slot_of[t];

        sim_fill_if_ready(st, nbytes_of);

        // pre-fence
        if (st.filled[s] != t) {
            ++st.wedged;
            return false;   // the engine aborts instead of multiplying with a stale image
        }
        if (st.content[s] != t) {
            ++st.stale;
        }

        // the mul_mat_id chain reads the slot
        st.busy[s] = true;
        st.held[s] = t;

        // post-fence: release the slot, then kick the ordinal that will reuse it
        st.busy[s] = false;
        if (kick[t] >= 0) {
            sim_queue(st, slot_of[kick[t]], kick[t]);
            sim_fill_if_ready(st, nbytes_of);
        }
    }

    // a correctly built plan leaves nothing queued at the end of a micro-batch
    return st.queue.empty();
}

static sim_state sim_new(int S) {
    sim_state st;
    st.content.assign(S, -1);
    st.filled.assign(S, -1);
    st.busy.assign(S, false);
    st.held.assign(S, -1);
    return st;
}

static void test_protocol(void) {
    for (int S : { 1, 2, 3, 4, 7 }) {
        for (int n : { 1, 2, 5, 41 }) {
            std::vector<int> slot, boot, kick;
            build_ring(n, S, slot, boot, kick);

            std::vector<uint64_t> nbytes(n);
            uint64_t want = 0;
            for (int t = 0; t < n; ++t) {
                nbytes[t] = 144u * 1024u * (t + 1);   // distinct, not a power of two
                want     += nbytes[t];
            }

            sim_state st = sim_new(S);
            for (int ub = 0; ub < 4; ++ub) {
                const bool ok = sim_microbatch(st, slot, boot, kick, nbytes);

                CHECK(ok, "S=%d n=%d ub=%d wedged", S, n, ub);
                CHECK(st.wedged == 0, "S=%d n=%d ub=%d wedge count %d", S, n, ub, st.wedged);
                CHECK(st.stale == 0, "S=%d n=%d ub=%d stale reads %d", S, n, ub, st.stale);
                CHECK(st.torn == 0, "S=%d n=%d ub=%d torn writes %d", S, n, ub, st.torn);
                if (!ok) {
                    break;
                }
                if (ub == 0) {
                    CHECK(st.bytes == want, "first micro-batch must move every image: %llu / %llu",
                          (unsigned long long) st.bytes, (unsigned long long) want);
                    CHECK(st.n_fills == n, "first micro-batch fills %d of %d", st.n_fills, n);
                } else {
                    // the ring is reused: images that already sit in the right slot are not re-sent, so
                    // a later micro-batch never moves more than one image per layer
                    CHECK(st.bytes <= (uint64_t) (ub + 1) * want,
                          "S=%d n=%d ub=%d moved %llu bytes", S, n, ub,
                          (unsigned long long) st.bytes);
                    CHECK(st.n_skips >= 0, "counter sanity");
                }
            }
        }
    }
}

// negative control: with the release check removed, the same replay must produce torn writes. without
// this case the safety assertions above could be vacuous
static void test_protocol_negative(void) {
    std::vector<int> slot, boot, kick;
    build_ring(8, 2, slot, boot, kick);
    std::vector<uint64_t> nbytes(8, 1024);

    sim_state st = sim_new(2);
    st.ignore_release = true;

    // the kick of ordinal t always targets the slot ordinal t reads, so a fill issued without waiting
    // for the release event would overwrite the image that is being multiplied with
    for (int t = 0; t < 8; ++t) {
        if (kick[t] < 0) {
            continue;
        }
        CHECK(slot[kick[t]] == slot[t], "kick target slot differs at ordinal %d", t);
    }

    // the queue-drain with ignore_release must count torn writes when a fill targets a busy slot
    sim_state st2 = sim_new(2);
    st2.ignore_release = true;
    st2.busy[0] = true;
    st2.held[0] = 7;
    st2.queue.push_back({ 0, 3 });
    sim_fill_if_ready(st2, nbytes);
    CHECK(st2.torn == 1, "filling a busy slot must be counted as torn, got %d", st2.torn);
    CHECK(st2.n_fills == 1, "the negative control still performed the fill");
}

// liveness: a lost kick must wedge the sequence loudly rather than read a stale image
static void test_protocol_wedged(void) {
    std::vector<int> slot, boot, kick;
    build_ring(8, 2, slot, boot, kick);

    std::vector<uint64_t> nbytes(8, 1024);
    std::vector<int> broken = kick;
    broken[2] = -1;   // lose the fill of ordinal 4

    sim_state st = sim_new(2);
    const bool ok = sim_microbatch(st, slot, boot, broken, nbytes);

    CHECK(!ok, "a lost kick must wedge the sequence");
    CHECK(st.wedged == 1, "expected exactly one wedge, got %d", st.wedged);
    CHECK(st.stale == 0, "a wedged sequence must never read a stale image, got %d", st.stale);
}

//
// asynchronous replay
//
// The three progress points of the real engine are independent: the host submits graph nodes ahead of
// the device, the device executes the compute stream, and the worker copies on its own stream. The
// replay above collapses them, so it cannot see the one hazard that matters for the release event:
// cudaEventQuery reports the state of the most recent cudaEventRecord, and the record of the *current*
// read is issued after its mul_mat_id nodes. A second job for the same slot (the ring is re-primed at
// the start of every streamed micro-batch, so a re-arm can overlap a job that is still queued) therefore
// sees the previous read's completed event and starts a copy while the compute stream is reading.
//
// The engine gates that with a release generation: a fill may start only when a release was recorded
// after the previous fill of that slot published, and that record has completed. The negative case below
// runs the same schedule with the generation removed and must observe the overlap.
//

enum { AX_READ = 0, AX_RECORD = 1 };

struct ax_op { int slot; int layer; int kind; };

struct ax_slot {
    int  content   = -1;   // image validly in the slot
    int  submitted = -1;   // published fill, what a pre-fence waits for
    int  queued    = -1;   // layer of the pending job
    int  copy      = -1;   // layer in flight on the copy stream
    int  armed     = -1;   // layer the worker picked, waiting for the release event
    bool pending   = false;
    bool ev_recorded = false;   // a cudaEventRecord(ev_used) was issued
    bool ev_ready    = false;   // and the device has passed it

    int  credits     = 1;    // releases granted since the last copy: how many copies may start
    uint64_t n_fill  = 0;
};

struct ax_state {
    std::vector<ax_slot>          s;
    std::vector<ax_op>            devq;   // compute stream, in submission order
    std::vector<std::vector<int>> q;      // one fill queue per slot, like the engine
    int  overlap = 0;   // a copy started or ran while a read of that slot was pending
    int  stale   = 0;   // a read executed against a slot that no longer holds its layer
    int  wedged  = 0;
    int  fills   = 0;
    uint64_t bytes = 0;
    int  violation = 0;   // a second, different fill for a slot that is still owned (the engine errors)
    bool gate   = true;   // release-credit gate (the shipped engine) or event-only gating (pre-fix)
    bool dedupe = true;   // queue_fill drops an identical pending job
};

static ax_state ax_new(int S) {
    ax_state st;
    st.s.assign(S, ax_slot());
    st.q.assign(S, std::vector<int>());
    return st;
}

static bool ax_queued(const ax_state & st) {
    for (const auto & v : st.q) {
        if (!v.empty()) {
            return true;
        }
    }
    return false;
}

static bool ax_read_pending(const ax_state & st, int slot) {
    for (const ax_op & o : st.devq) {
        if (o.slot == slot && o.kind == AX_READ) {
            return true;
        }
    }
    return false;
}

static bool ax_armed(const ax_state & st) {
    for (const ax_slot & a : st.s) {
        if (a.armed >= 0) {
            return true;
        }
    }
    return false;
}

static bool ax_any_copy(const ax_state & st) {
    for (const ax_slot & a : st.s) {
        if (a.copy >= 0) {
            return true;
        }
    }
    return false;
}

static void ax_queue(ax_state & st, int slot, int layer, bool dedupe) {
    ax_slot & a = st.s[slot];
    if (a.content == layer && !a.pending) {
        a.submitted = layer;
        return;
    }
    if (dedupe && a.pending) {
        if (layer == a.queued) {
            return;   // the identical fill is already in flight
        }
        bool same = false;
        for (int l : st.q[slot]) {
            if (l == layer) {
                same = true;
            }
        }
        if (same) {
            return;
        }
        ++st.violation;   // the engine reports an error rather than pick between two unsafe copies
        return;
    }
    a.submitted = -1;
    a.pending   = true;
    a.queued    = layer;
    st.q[slot].push_back(layer);
}

// the worker: one transfer at a time, it picks the head of the first slot queue whose release generation
// was observed, and a slot it has already picked keeps it until its release event completes
static void ax_worker(ax_state & st) {
    if (ax_any_copy(st)) {
        return;
    }
    for (size_t s = 0; s < st.s.size(); ++s) {
        ax_slot & a = st.s[s];
        if (a.armed < 0) {
            continue;
        }
        if (a.ev_recorded && !a.ev_ready) {
            return;   // cudaEventQuery(ev_used) == cudaErrorNotReady, the worker waits here
        }
        if (ax_read_pending(st, (int) s)) {
            ++st.overlap;   // the copy starts while the compute stream still has a read for this slot
        }
        if (st.gate && a.credits > 0) {
            a.credits--;    // the release this copy was allowed to consume
        }
        a.copy  = a.armed;
        a.armed = -1;
        return;
    }
    for (size_t s = 0; s < st.s.size(); ++s) {
        if (st.q[s].empty()) {
            continue;
        }
        const ax_slot & a = st.s[s];
        const int       want = st.q[s].front();
        if (st.gate && a.credits == 0 && a.content != want) {
            continue;   // gated: a reader of this slot may still be live, other slots are served anyway
        }
        st.q[s].erase(st.q[s].begin());
        if (st.gate && a.content == want) {
            // the slot already holds this image: publish it without writing, so no credit is spent
            st.s[s].submitted = want;
            st.s[s].pending   = false;
            st.s[s].queued    = -1;
            return;
        }
        st.s[s].armed = want;
        return;
    }
}

// one device step: the copy engine retires one transfer and the compute stream advances one node
static void ax_device(ax_state & st, const std::vector<uint64_t> & nb) {
    for (size_t s = 0; s < st.s.size(); ++s) {
        ax_slot & a = st.s[s];
        if (a.copy < 0) {
            continue;
        }
        const int l = a.copy;
        a.copy      = -1;
        a.content   = l;
        a.submitted = l;
        a.pending   = false;
        a.queued    = -1;
        ++a.n_fill;
        ++st.fills;
        st.bytes        += nb[l];
        break;
    }
    if (!st.devq.empty()) {
        const ax_op o = st.devq.front();
        st.devq.erase(st.devq.begin());
        if (o.kind == AX_READ) {
            if (st.s[o.slot].content != o.layer) {
                ++st.stale;
            }
        } else {
            st.s[o.slot].ev_recorded = true;
            st.s[o.slot].ev_ready    = true;
        }
    }
}

static ax_state ax_run(int S, int n, int ubatches, bool gate, bool dedupe, int inject,
                       uint64_t * want_out) {
    std::vector<int> slot, boot, kick;
    build_ring(n, S, slot, boot, kick);

    std::vector<uint64_t> nbytes(n);
    uint64_t want = 0;
    for (int t = 0; t < n; ++t) {
        nbytes[t] = 2048u * (t + 1);
        want     += nbytes[t];
    }
    if (want_out) {
        *want_out = want;
    }

    ax_state st = ax_new(S);
    st.gate   = gate;
    st.dedupe = dedupe;

    for (int ub = 0; ub < ubatches; ++ub) {
        // llama_moe_stream_arm re-primes the ring at the start of every streamed micro-batch
        for (int t : boot) {
            ax_queue(st, slot[t], t, dedupe);
        }
        if (inject == 1) {
            // a re-arm that lands on top of a job that is still queued for the same slot, same image
            for (int t : boot) {
                ax_queue(st, slot[t], t, false);
            }
        } else if (inject == 2) {
            // ... and the same with a different image, which no release accounting can reason about
            for (int t : boot) {
                ax_queue(st, slot[t], t + 1000, true);
            }
        }

        for (int t = 0; t < n && st.wedged == 0; ++t) {
            const int s = slot[t];

            int spins = 0;
            while (st.s[s].submitted != t) {
                ax_worker(st);
                ax_device(st, nbytes);
                if (++spins > 500) {
                    ++st.wedged;
                    break;
                }
            }
            if (st.wedged) {
                break;
            }

            // the host submits the mul_mat_id nodes and the post-fence record back to back: a read of a
            // 1.5 GiB expert image takes milliseconds, the host microseconds, so between the two
            // submissions the release event still reports the *previous* read. a fill that starts there
            // looks legal to an event-only gate and races the reader
            st.devq.push_back({ s, t, AX_READ });
            ax_worker(st);

            st.devq.push_back({ s, t, AX_RECORD });
            st.s[s].credits++;          // the post-fence grants one release for this slot
            st.s[s].ev_ready = false;   // a fresh record is not-ready until the device reaches it
            ax_worker(st);

            // the device stays behind the host within one layer and catches up between layers, which is
            // the real relationship: submission is microseconds, a layer is milliseconds
            for (int k = 0; k < 4; ++k) {
                ax_device(st, nbytes);
            }

            if (kick[t] >= 0) {
                ax_queue(st, slot[kick[t]], kick[t], dedupe);
                ax_worker(st);
            }
        }

        for (int i = 0; i < 512 && (!st.devq.empty() || ax_queued(st) || ax_any_copy(st) ||
                                    ax_armed(st)); ++i) {
            ax_worker(st);
            ax_device(st, nbytes);
        }
    }
    return st;
}

static void test_protocol_async(void) {
    uint64_t want = 0;

    // negative control: event-only gating (the pre-fix engine) must let a re-armed duplicate fill overlap
    // a live read, otherwise the assertions below are vacuous
    {
        const ax_state st = ax_run(2, 8, 3, false, false, /* inject */ 1, &want);
        CHECK(st.overlap >= 1,
              "event-only gating must reproduce the re-arm overlap, got %d (the fix would be untested)",
              st.overlap);
    }

    // the shipped engine: generation gate and duplicate suppression
    for (int S : { 1, 2, 3, 5 }) {
        for (int n : { 3, 8, 41 }) {
            const ax_state clean = ax_run(S, n, 3, true, true, 0, &want);
            CHECK(clean.wedged == 0, "S=%d n=%d the gate must not wedge a legal ring", S, n);
            CHECK(clean.stale == 0, "S=%d n=%d stale reads %d", S, n, clean.stale);
            CHECK(clean.overlap == 0, "S=%d n=%d overlapping fills %d", S, n, clean.overlap);

            // the injection bypasses duplicate suppression on purpose: the credit gate and the
            // already-holds path must absorb it without wedging, without a stale read and without a race
            const ax_state inj = ax_run(S, n, 3, true, true, 1, &want);
            CHECK(inj.wedged == 0, "S=%d n=%d an injected duplicate job must not wedge the ring", S, n);
            CHECK(inj.stale == 0, "S=%d n=%d stale reads with an injected job: %d", S, n, inj.stale);
            CHECK(inj.overlap == 0, "S=%d n=%d a credit-gated fill overlapped a read: %d", S, n, inj.overlap);
            CHECK(inj.bytes == clean.bytes, "S=%d n=%d a redundant re-arm must not move bytes: %llu / %llu",
                  S, n, (unsigned long long) inj.bytes, (unsigned long long) clean.bytes);
            CHECK(clean.bytes <= (uint64_t) 3 * want, "S=%d n=%d traffic over three micro-batches %llu",
                  S, n, (unsigned long long) clean.bytes);
        }
    }

    // the gate has to work on its own: the graph builder is allowed to queue the same image twice and
    // still be safe even if the duplicate suppression were removed
    for (int S : { 1, 2, 3 }) {
        const ax_state noguard_off = ax_run(S, 12, 3, true, false, 1, nullptr);
        CHECK(noguard_off.overlap == 0, "S=%d the credit gate alone must prevent the overlap, got %d",
              S, noguard_off.overlap);
        CHECK(noguard_off.wedged == 0, "S=%d the credit gate alone wedged the ring", S);
    }

    // a second, different fill for a slot that is still owned is refused loudly, never raced
    for (int S : { 1, 2, 3 }) {
        const ax_state bad = ax_run(S, 12, 2, true, true, 2, nullptr);
        CHECK(bad.violation >= 1, "S=%d a conflicting fill must be reported, got %d", S, bad.violation);
        CHECK(bad.overlap == 0, "S=%d a conflicting fill must never race a reader, got %d", S, bad.overlap);
        CHECK(bad.stale == 0, "S=%d a conflicting fill must never read stale bytes, got %d", S, bad.stale);
    }

    // duplicate suppression on its own terms: the same (slot, layer) queued twice while pending is one job
    {
        ax_state st = ax_new(2);
        ax_queue(st, 0, 3, true);
        ax_queue(st, 0, 3, true);
        CHECK(st.q[0].size() == 1, "an identical pending fill must be suppressed, got %d jobs",
              (int) st.q[0].size());
        ax_queue(st, 1, 3, true);
        CHECK(st.q[1].size() == 1, "a different slot must still be queued");
    }
}

// the plan itself: the kick of ordinal t always targets the slot ordinal t reads, which is what makes
// the release of that slot the earliest possible start of the next fill
static void test_plan_kick_slot_identity(void) {
    for (int S : { 1, 2, 3, 5 }) {
        for (int n : { 1, 3, 12, 48 }) {
            std::vector<int> slot, boot, kick;
            build_ring(n, S, slot, boot, kick);
            for (int t = 0; t < n; ++t) {
                if (kick[t] < 0) {
                    CHECK(t + S >= n, "ordinal %d must have a kick when %d + %d < %d", t, t, S, n);
                    continue;
                }
                CHECK(slot[kick[t]] == slot[t], "kick target slot differs at ordinal %d (S=%d)", t, S);
                CHECK(kick[t] == t + S, "kick target must be t+S at %d (S=%d)", t, S);
            }
        }
    }
}

//
// timing model
//
// per streamed ordinal m, one device:
//   release(m)  = read_end(m - S)                        (0 while m < S)
//   submit(m)   = max(worker_free, release(m))           the worker waits for the release event
//   worker_free = submit(m) + stage                      one worker thread stages serially
//   done(m)     = submit(m) + (chunked ? max(stage, dma) : stage + dma)
//   start(m)    = max(read_end(m - 1), done(m))          the pre-fence blocks the compute stream
//   read_end(m) = start(m) + c(m)
//
// stage      = bytes / stage_bw              host copy into the pinned mirror
// dma        = bytes / dma_bw                PCIe host -> device
// c(m)       = n_tokens * gflop_layer / gpu  expert GEMM time of the layer for this micro-batch
//

struct model_in {
    int    n;              // streamed layers on this device
    int    S;              // arena slots
    double bytes;          // expert image bytes per layer
    double stage_bw;       // bytes/us, host copy into the pinned mirror
    double dma_bw;         // bytes/us, PCIe host -> device
    double tokens;         // micro-batch size
    double gflop_layer;    // expert FLOP per token per layer (GFLOP)
    double tflops;         // device throughput for the expert GEMMs (TFLOP/s)
    bool   chunked;        // staging pipelined against the DMA
};

struct model_out {
    double total_us   = 0.0;
    double stall_us   = 0.0;   // time the compute stream waited on a fill
    double traffic    = 0.0;
    double stage_us   = 0.0;
    double dma_us     = 0.0;
    double c_us       = 0.0;
    int    overlaps   = 0;     // fills that began before the previous reader finished
};

static model_out model_run(const model_in & in) {
    model_out o;

    std::vector<double> read_end(in.n, 0.0);

    o.stage_us = in.bytes / in.stage_bw;
    o.dma_us   = in.bytes / in.dma_bw;
    o.c_us     = in.tokens * in.gflop_layer * 1e3 / in.tflops;   // GFLOP/(TFLOP/s) -> us

    const double busy = in.chunked ? std::max(o.stage_us, o.dma_us) : o.stage_us + o.dma_us;

    double worker_free = 0.0;
    double prev_end    = 0.0;

    for (int m = 0; m < in.n; ++m) {
        const double rel = (m - in.S >= 0) ? read_end[m - in.S] : 0.0;
        const double sub = std::max(worker_free, rel);
        worker_free      = sub + o.stage_us;

        if (m - in.S >= 0 && sub < read_end[m - in.S]) {
            ++o.overlaps;
        }

        const double done  = sub + busy;
        const double start = std::max(prev_end, done);

        o.stall_us += std::max(0.0, done - prev_end);

        read_end[m] = start + o.c_us;
        prev_end    = read_end[m];
    }

    o.total_us = prev_end;
    o.traffic  = (double) in.n * in.bytes;
    return o;
}

static void test_model(void) {
    // this box: i9-13900K, dual-channel DDR5, GPU0 in a gen5 x8 slot, GPU1 in gen4 x4.
    // the bandwidths are estimates pending measurement; the recurrence does not depend on them
    const double gib = 1024.0 * 1024.0 * 1024.0;

    model_in in;
    in.n           = 41;              // layers that land on GPU0 with a bandwidth-proportional split
    in.bytes       = 1.4943 * gib;    // measured mean expert image size per layer
    in.stage_bw    = 4.0e4;           // 40 GB/s
    in.dma_bw      = 4.5e4;           // 45 GB/s, gen5 x8
    in.tokens      = 512.0;
    in.gflop_layer = 98.3e-3;         // 10 experts x 3 x 2 x 640 x 2560 = 98.3 MFLOP per token
    in.tflops      = 12.0;            // q4_k x f16 expert GEMMs on a 5060 Ti, dequant limited
    in.chunked     = true;

    in.S = 1;  const model_out s1 = model_run(in);
    in.S = 2;  const model_out s2 = model_run(in);
    in.S = 3;  const model_out s3 = model_run(in);

    CHECK(s1.overlaps == 0 && s2.overlaps == 0 && s3.overlaps == 0,
          "the plan must never overlap a live slot");
    CHECK(s2.total_us <= s1.total_us, "S=2 (%.1f ms) must not be slower than S=1 (%.1f ms)",
          s2.total_us / 1e3, s1.total_us / 1e3);
    CHECK(s3.total_us <= s2.total_us, "S=3 (%.1f ms) must not be slower than S=2 (%.1f ms)",
          s3.total_us / 1e3, s2.total_us / 1e3);

    // with a single slot the fill of the next layer cannot start before the previous layer finished
    // reading, so the slot staging time is on the critical path in full
    CHECK(s1.total_us >= in.n * s1.stage_us - 1e-6,
          "S=1 total %.1f ms below the serialized staging bound %.1f ms",
          s1.total_us / 1e3, in.n * s1.stage_us / 1e3);

    // the total must sit inside a sandwich of the bottleneck resource and the fully additive cost
    for (int S : { 1, 2, 3 }) {
        model_in m = in;
        m.S = S;
        const model_out r = model_run(m);
        const double bottleneck = std::max(std::max(r.stage_us, r.dma_us), r.c_us);
        const double lo = in.n * bottleneck;
        const double hi = in.n * (std::max(r.stage_us, r.dma_us) + r.c_us) + std::max(r.stage_us, r.dma_us);
        CHECK(r.total_us >= lo - 1e-6, "S=%d total %.1f ms below the resource bound %.1f ms",
              S, r.total_us / 1e3, lo / 1e3);
        CHECK(r.total_us <= hi + 1e-6, "S=%d total %.1f ms above the additive bound %.1f ms",
              S, r.total_us / 1e3, hi / 1e3);
    }

    // staging pipelining must never hurt
    in.chunked = false;
    const model_out p2 = model_run(in);
    in.chunked = true;
    CHECK(s2.total_us <= p2.total_us + 1e-6, "chunked staging %.1f ms vs serialized %.1f ms",
          s2.total_us / 1e3, p2.total_us / 1e3);
    CHECK(p2.total_us >= s2.total_us, "serialized staging must be reported honestly");

    // bytes per token = image / B: a bigger micro-batch is the single biggest lever until the device
    // becomes compute bound
    std::printf("\n  model: %d layers x %.3f GiB, stage %.0f GB/s, DMA %.0f GB/s, "
                "%.1f MFLOP/token/layer at %.0f TFLOP/s\n",
                in.n, in.bytes / gib, in.stage_bw * 1e6 / 1e9, in.dma_bw * 1e6 / 1e9,
                in.gflop_layer * 1e3, in.tflops);
    std::printf("  %-8s %-6s %-11s %-11s %-9s %-10s\n",
                "ubatch", "slots", "MiB/token", "tok/s", "stage ms", "compute ms");

    double prev_tps = -1.0;
    for (int B : { 512, 1024, 2048, 4096, 8192 }) {
        model_in m = in;
        m.S       = 3;
        m.chunked = true;
        m.tokens  = B;
        const model_out r   = model_run(m);
        const double    tps = B / (r.total_us * 1e-6);

        std::printf("  %-8d %-6d %-11.1f %-11.0f %-9.2f %-10.2f\n", B, m.S,
                    in.bytes / (1024.0 * 1024.0) / B, tps, r.stage_us / 1e3, r.c_us / 1e3);

        CHECK(tps >= prev_tps - 1e-9, "throughput must not fall as the micro-batch grows (%d tokens)", B);
        prev_tps = tps;
    }

    // the regime check: while staging is the bottleneck, making the GEMMs cheaper cannot help, and
    // once the device is the bottleneck, moving bytes faster cannot help either
    {
        model_in m = in;
        m.S = 3;
        m.chunked = true;
        m.tokens  = 4096;
        const model_out base = model_run(m);
        CHECK(base.stage_us > base.c_us, "4096 tokens should still be staging bound in this model");
        m.tflops = in.tflops * 4.0;
        const model_out fast = model_run(m);

        // steady state: one layer per staging interval regardless of how fast the GEMMs are
        const double rate_base = (base.total_us - base.c_us) / in.n;
        const double rate_fast = (fast.total_us - fast.c_us) / in.n;
        CHECK(std::fabs(rate_base - rate_fast) < 1e-6,
              "transfer bound: per-layer interval %.3f us vs %.3f us with a 4x faster device",
              rate_base, rate_fast);
        CHECK(fast.total_us < base.total_us && base.total_us - fast.total_us <= base.c_us + 1e-6,
              "a faster device may only remove the tail of the last layer's compute");
        m.tflops = in.tflops;
        m.tokens = 16384;
        const model_out big = model_run(m);
        CHECK(big.c_us > big.stage_us, "at 16384 tokens the model should be compute bound");
        CHECK(big.total_us > 0.0, "sanity");
    }
}

static void test_reused_slot_padding(void) {
    ggml_cuda_moe_stream_layer large = {};
    ggml_cuda_moe_stream_layer small = {};
    const uint64_t large_spans[3] = { 512, 512, 2048 + 240 };
    const uint64_t small_spans[3] = { 512, 512, 1024 + 240 };
    uint64_t large_off[3], small_off[3];
    const uint64_t capacity = layout_families(large_spans, 512, large_off);
    const uint64_t small_size = layout_families(small_spans, 512, small_off);
    for (int f = 0; f < 3; ++f) {
        large.pad_bytes[f] = small.pad_bytes[f] = f == 2 ? 240 : 0;
        large.nbytes[f] = large_spans[f] - large.pad_bytes[f];
        small.nbytes[f] = small_spans[f] - small.pad_bytes[f];
        large.dst_off[f] = large_off[f];
        small.dst_off[f] = small_off[f];
    }

    // Descriptor initialization clears tails once, before any layer fills the shared slot.
    std::vector<uint8_t> arena(capacity, 0);
    auto fill = [&](const ggml_cuda_moe_stream_layer & layer, uint8_t pattern, bool clear_padding) {
        for (int f = 0; f < 3; ++f) {
            std::memset(arena.data() + layer.dst_off[f], pattern, layer.nbytes[f]);
            if (clear_padding) {
                std::memset(arena.data() + layer.dst_off[f] + layer.nbytes[f], 0, layer.pad_bytes[f]);
            }
        }
    };
    fill(large, 0x7f, false);
    fill(small, 0x11, false);
    const size_t tail = small.dst_off[2] + small.nbytes[2];
    uint16_t stale_scale = 0;
    std::memcpy(&stale_scale, arena.data() + tail, sizeof(stale_scale));
    CHECK((stale_scale & 0x7c00) == 0x7c00 && (stale_scale & 0x03ff) != 0,
          "negative control: previous larger image must leave an FP16 NaN in the shorter tensor tail");
    fill(small, 0x11, true);
    CHECK(std::all_of(arena.begin() + tail, arena.begin() + tail + small.pad_bytes[2],
                     [](uint8_t b) { return b == 0; }), "every reused quantized tail must be zero");
    CHECK(arena[tail - 1] == 0x11, "padding clear must preserve the final weight byte");

    config cfg;
    cfg.slot_bytes = small_size - small.pad_bytes[2];
    layer_src src;
    src.il = src.device = 0;
    src.nbytes = small_size;
    CHECK(!plan({ src }, cfg).enabled, "an image must reserve backend padding inside its slot");
}

int main(void) {
    test_protocol();
    test_protocol_negative();
    test_protocol_wedged();
    test_protocol_async();
    test_plan_kick_slot_identity();
    test_reused_slot_padding();
    test_model();

    std::printf("\ntest-moe-stream-sim: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
