#include "llama-moe-stream-plan.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <map>

namespace llama_moe_plan {

const char * source_rejection(const source_traits & s, int max_devices) {
    if (!s.gate_input)                 { return "missing_gate_input"; }
    if (s.fused_gate_up)               { return "unsupported_fused_gate_up"; }
    if (!s.have_all)                   { return "missing_expert_family"; }
    if (s.scales)                      { return "unsupported_expert_scales"; }
    if (s.biases)                      { return "unsupported_expert_biases"; }
    if (s.device < 0)                  { return "no_cuda_layer_owner"; }
    if (s.device >= max_devices)       { return "device_ordinal_out_of_range"; }
    if (s.data_mask != 7)              { return "missing_expert_data"; }
    if (!s.contiguous)                 { return "noncontiguous_expert_layout"; }
    if (s.host_mask == 7)              { return ""; }
    if (s.device_mask == 7)            { return "device_resident"; }
    if (s.host_mask != 0)              { return "mixed_host_nonhost_experts"; }
    return "nonhost_expert_source";
}

uint64_t arena_budget(uint64_t requested_bytes, uint64_t free_bytes, uint64_t reserve_bytes) {
    return requested_bytes ? std::min(requested_bytes, free_bytes)
                           : (free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0);
}

int clamp_slots(uint64_t slot_bytes, uint64_t budget_bytes, int want, int hard_max) {
    if (slot_bytes == 0) {
        return 0;
    }
    if (want < 1) {
        want = 1;
    }
    if (hard_max >= 1 && want > hard_max) {
        want = hard_max;
    }
    if (budget_bytes == 0) {
        return want;
    }
    const uint64_t fit = budget_bytes / slot_bytes;
    if (fit < 1) {
        return 0;
    }
    return fit < (uint64_t) want ? (int) fit : want;
}

uint64_t layout_families(const uint64_t nbytes[3], uint64_t align, uint64_t dst_off[3]) {
    uint64_t off = 0;

    for (int f = 0; f < 3; ++f) {
        dst_off[f] = 0;
        if (nbytes[f] == 0) {
            continue;
        }
        if (align > 1) {
            off = (off + align - 1) / align * align;
        }
        dst_off[f] = off;
        off += nbytes[f];
    }

    return off;
}

void build_ring(int n, int n_slots, std::vector<int> & slot, std::vector<int> & bootstrap,
                std::vector<int> & post_kick) {
    slot.assign(n < 0 ? 0 : n, -1);
    bootstrap.clear();
    post_kick.assign(n < 0 ? 0 : n, -1);

    if (n <= 0 || n_slots < 1) {
        return;
    }

    for (int t = 0; t < n; ++t) {
        slot[t] = t % n_slots;
    }

    const int boot = n < n_slots ? n : n_slots;
    for (int t = 0; t < boot; ++t) {
        bootstrap.push_back(t);
    }

    for (int t = 0; t < n; ++t) {
        const int nxt = t + n_slots;
        if (nxt < n) {
            post_kick[t] = nxt;
        }
    }
}

result plan(const std::vector<layer_src> & layers, const config & cfg) {
    result res;
    res.slot.assign(layers.size(), -1);
    res.ordinal.assign(layers.size(), -1);
    res.post_kick.assign(layers.size(), -1);

    // group by owning device, preserving model-layer order inside each group
    std::map<int, std::vector<size_t>> by_dev;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].nbytes == 0) {
            res.n_resident++;
            continue;
        }
        if (layers[i].il < 0 || layers[i].device < 0) {
            res.reason = "layer entry is not initialised";
            return res;
        }
        if (layers[i].nbytes > cfg.slot_bytes) {
            res.reason = "slot_bytes smaller than a layer image";
            return res;
        }
        by_dev[layers[i].device].push_back(i);
        res.stream_bytes += layers[i].nbytes;
    }

    res.n_streamed = (int) layers.size() - res.n_resident;
    res.n_devices  = (int) by_dev.size();

    if (res.n_streamed == 0) {
        res.reason = "no host-resident expert layers to stream";
        return res;
    }

    if (cfg.slot_overhead_bytes > std::numeric_limits<uint64_t>::max() - cfg.slot_bytes) {
        res.reason = "slot allocation size overflow";
        return res;
    }
    const uint64_t allocation_bytes = cfg.slot_bytes + cfg.slot_overhead_bytes;
    res.slot_bytes = cfg.slot_bytes;
    res.n_slots    = clamp_slots(allocation_bytes, cfg.budget_bytes, cfg.n_slots, cfg.hard_max_slots);
    if (res.n_slots < 1) {
        res.reason = "device budget cannot hold a single arena slot";
        return res;
    }
    res.arena_bytes = (uint64_t) res.n_slots * allocation_bytes;

    for (const auto & kv : by_dev) {
        const std::vector<size_t> & pos = kv.second;

        std::vector<int> slot;
        std::vector<int> boot;
        std::vector<int> kick;
        build_ring((int) pos.size(), res.n_slots, slot, boot, kick);

        for (size_t t = 0; t < pos.size(); ++t) {
            res.slot[pos[t]]    = slot[t];
            res.ordinal[pos[t]] = (int) t;
        }
        for (int t : boot) {
            res.bootstrap.push_back((int) pos[t]);
        }
        for (size_t t = 0; t < pos.size(); ++t) {
            if (kick[t] >= 0) {
                res.post_kick[pos[t]] = (int) pos[kick[t]];
            }
        }
    }

    res.enabled = true;
    res.reason  = "ok";
    return res;
}


std::vector<int> place_arena(const std::vector<int> & layer_dev,
                             const std::vector<uint64_t> & nbytes,
                             int n_dev, int mode, int primary_dev,
                             const std::vector<float> & weights,
                             std::string * note) {
    if (note) {
        note->clear();
    }

    const int n = (int) layer_dev.size();
    std::vector<int> out = layer_dev;

    if (n_dev <= 0 || mode == PLACE_LAYER) {
        return out;
    }

    std::vector<int> idx;
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (layer_dev[i] < 0) {
            continue;
        }
        idx.push_back(i);
        total += i < (int) nbytes.size() ? nbytes[i] : 0;
    }
    if (idx.empty() || total == 0) {
        return out;
    }

    if (mode == PLACE_PRIMARY) {
        int target = primary_dev;
        if (target < 0 || target >= n_dev) {
            // no explicit choice: the device carrying the most streamed bytes is the natural host
            std::vector<uint64_t> per(n_dev, 0);
            for (int i : idx) {
                per[layer_dev[i]] += i < (int) nbytes.size() ? nbytes[i] : 0;
            }
            target = 0;
            for (int d = 1; d < n_dev; ++d) {
                if (per[d] > per[target]) {
                    target = d;
                }
            }
        }
        int moved = 0;
        for (int i : idx) {
            if (out[i] != target) {
                out[i] = target;
                ++moved;
            }
        }
        if (note) {
            *note = "all arenas on device " + std::to_string(target) + ", " +
                    std::to_string(moved) + " of " + std::to_string((int) idx.size()) +
                    " layers moved off their owning device";
        }
        return out;
    }

    if (mode != PLACE_SPLIT) {
        return out;
    }

    // byte-proportional split: every device gets a share of the streamed bytes, layers are handed to
    // the device that is furthest below its target. within one layer image of the ideal split, and it
    // keeps a device with weight 0 out of the transfer entirely
    std::vector<double> w(n_dev, 0.0);
    double sum = 0.0;
    for (int d = 0; d < n_dev; ++d) {
        const float v = d < (int) weights.size() ? weights[d] : 1.0f;
        w[d] = v > 0.0f ? (double) v : 0.0;
        sum += w[d];
    }
    if (sum <= 0.0) {
        for (double & v : w) {
            v = 1.0;
        }
        sum = (double) n_dev;
    }

    std::vector<uint64_t> assigned(n_dev, 0);
    int moved = 0;
    for (int i : idx) {
        const uint64_t b = i < (int) nbytes.size() ? nbytes[i] : 0;
        int best = -1;
        double best_deficit = 0.0;
        for (int d = 0; d < n_dev; ++d) {
            if (w[d] <= 0.0) {
                continue;
            }
            const double deficit = (double) total * (w[d] / sum) - (double) assigned[d];
            if (best < 0 || deficit > best_deficit) {
                best = d;
                best_deficit = deficit;
            }
        }
        if (best < 0) {
            best = layer_dev[i];   // every device excluded: leave the layer where it is
        }
        out[i]        = best;
        assigned[best] += b;
        if (best != layer_dev[i]) {
            ++moved;
        }
    }

    if (note) {
        *note = "arena split by weight over " + std::to_string(n_dev) + " device(s), " +
                std::to_string(moved) + " of " + std::to_string((int) idx.size()) +
                " layers moved, bytes per device:";
        for (int d = 0; d < n_dev; ++d) {
            *note += " " + std::to_string(assigned[d] >> 20) + " MiB";
        }
    }
    return out;
}

double expected_unique_experts(int64_t n_expert, int64_t k, int64_t n_tokens) {
    if (n_expert <= 0 || k <= 0 || n_tokens <= 0) {
        return 0.0;
    }
    if (k >= n_expert) {
        return (double) n_expert;
    }
    // exact: n_expert * (1 - prod_{j<k} (1 - j/n_expert)) is the collision-aware form, but for
    // k << n_expert the independent-pick form is accurate to well under one expert
    const double p_miss = std::pow(1.0 - (double) k / (double) n_expert, (double) n_tokens);
    return (double) n_expert * (1.0 - p_miss);
}

} // namespace llama_moe_plan
