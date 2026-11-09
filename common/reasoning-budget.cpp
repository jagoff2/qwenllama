#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;
    const llama_vocab * vocab;
    std::vector<std::string> texts;
    std::string tail;
    std::string suffix;
    size_t max_text = 0;

    token_matcher(const std::vector<llama_tokens> & seqs, const llama_vocab * vocab)
        : seqs(collect(seqs)), ac(build_trie(this->seqs)), vocab(vocab) {
        if (vocab) {
            for (const auto & seq : this->seqs) {
                std::string text;
                for (const auto token : seq) {
                    text += common_token_to_piece(vocab, token, true);
                }
                max_text = std::max(max_text, text.size());
                texts.push_back(std::move(text));
            }
        }
    }

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        int32_t p = ac.match_pattern(state);
        suffix.clear();
        if (vocab && max_text > 0) {
            const size_t previous = tail.size();
            tail += common_token_to_piece(vocab, token, true);
            size_t match_end = 0;
            size_t match_size = 0;
            for (size_t i = 0; i < texts.size(); ++i) {
                if (texts[i].empty()) {
                    continue;
                }
                const size_t pos = tail.rfind(texts[i]);
                const size_t end = pos == std::string::npos ? 0 : pos + texts[i].size();
                if (end > previous && (end > match_end || (end == match_end && texts[i].size() > match_size))) {
                    p = (int32_t) i;
                    match_end = end;
                    match_size = texts[i].size();
                }
            }
            if (match_end > 0) {
                suffix = tail.substr(match_end);
                tail = suffix;
            }
            if (tail.size() >= max_text) {
                tail.erase(0, tail.size() - max_text + 1);
            }
        }
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; tail.clear(); suffix.clear(); }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none
    bool awaiting_output;
    std::string pending_output;
    std::vector<std::string> output_markers;
    llama_tokens end_tokens;
    llama_tokens closing_tokens;
};

static bool reasoning_whitespace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || cp == 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f ||
           cp == 0x200b || cp == 0x205f || cp == 0x2060 || cp == 0x3000 || cp == 0xfeff;
}

static void reasoning_accept_output(common_reasoning_budget_ctx * ctx, const std::string & piece) {
    if (!ctx->awaiting_output) {
        return;
    }
    ctx->pending_output += piece;
    while (!ctx->pending_output.empty()) {
        const auto cp = common_parse_utf8_codepoint(ctx->pending_output, 0);
        if (cp.status == utf8_parse_result::INCOMPLETE) {
            return;
        }
        if (cp.status == utf8_parse_result::INVALID || reasoning_whitespace(cp.codepoint)) {
            ctx->pending_output.erase(0, cp.status == utf8_parse_result::INVALID ? 1 : cp.bytes_consumed);
            continue;
        }
        size_t matched = 0;
        bool partial = false;
        for (const auto & marker : ctx->output_markers) {
            if (ctx->pending_output.compare(0, marker.size(), marker) == 0) {
                matched = std::max(matched, marker.size());
            } else if (marker.compare(0, ctx->pending_output.size(), ctx->pending_output) == 0) {
                partial = true;
            }
        }
        if (matched > 0) {
            ctx->pending_output.erase(0, matched);
        } else if (partial) {
            return;
        } else {
            ctx->awaiting_output = false;
            ctx->pending_output.clear();
        }
    }
}

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    const int32_t start_match = ctx->start_matcher.advance(token);
    const int32_t end_match = ctx->end_matcher.advance(token);

    // A closing delimiter also arms the guard when the opening tag was not replayed.
    if (ctx->state != REASONING_BUDGET_FORCING && end_match >= 0 &&
            (start_match < 0 || ctx->end_matcher.suffix.size() <= ctx->start_matcher.suffix.size())) {
        ctx->state = REASONING_BUDGET_DONE;
        ctx->end_match = end_match;
        ctx->awaiting_output = true;
        ctx->pending_output.clear();
        reasoning_accept_output(ctx, ctx->end_matcher.suffix);
        COM_TRC("%s", "deactivated (natural end)\n");
        return;
    }

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (start_match >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = end_match;
                ctx->awaiting_output = true;
                ctx->pending_output.clear();
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (start_match >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                ctx->awaiting_output = false;
                ctx->pending_output.clear();
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            } else if (ctx->awaiting_output && ctx->vocab) {
                const auto attr = llama_vocab_get_attr(ctx->vocab, token);
                if (!(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED))) {
                    reasoning_accept_output(ctx, common_token_to_piece(ctx->vocab, token, false));
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_IDLE || (ctx->state == REASONING_BUDGET_DONE && !ctx->awaiting_output)) {
        return;
    }
    for (size_t i = 0; i < cur_p->size; i++) {
        if (!common_reasoning_budget_is_token_allowed(smpl, cur_p->data[i].id)) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
    cur_p->sorted = false;
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->awaiting_output = false;
    ctx->pending_output.clear();
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    auto * ctx = new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs, vocab),
            /* .end_matcher   = */ token_matcher(end_seqs, vocab),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
            /* .awaiting_output = */ initial_state == REASONING_BUDGET_DONE,
            /* .pending_output  = */ {},
            /* .output_markers  = */ {},
            /* .end_tokens      = */ {},
            /* .closing_tokens  = */ {},
    };
    ctx->output_markers = ctx->start_matcher.texts;
    ctx->output_markers.insert(ctx->output_markers.end(), ctx->end_matcher.texts.begin(), ctx->end_matcher.texts.end());
    const size_t configured_markers = ctx->output_markers.size();
    for (size_t i = 0; i < configured_markers; ++i) {
        const auto & marker = ctx->output_markers[i];
        if (marker.size() > 2 && marker.front() == '<' && marker[1] != '/' && marker.back() == '>') {
            const std::string closing = "</" + marker.substr(1);
            ctx->output_markers.push_back(closing);
        }
    }
    if (vocab) {
        for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
            const auto attr = llama_vocab_get_attr(vocab, token);
            const bool special = attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED);
            const std::string piece = special ? common_token_to_piece(vocab, token, true) : std::string();
            const bool end_control = (attr & LLAMA_TOKEN_ATTR_CONTROL) &&
                    (piece.find("end") != std::string::npos || piece.find("eot") != std::string::npos ||
                     piece.find("eom") != std::string::npos || piece.find("fim_suffix") != std::string::npos || piece == "</s>");
            if (llama_vocab_is_eog(vocab, token) || end_control) {
                ctx->end_tokens.push_back(token);
            }
            if ((attr & LLAMA_TOKEN_ATTR_USER_DEFINED) && piece.compare(0, 2, "</") == 0) {
                ctx->closing_tokens.push_back(token);
            }
            if (!piece.empty() && (piece.front() == '<' || piece.front() == '[')) {
                ctx->output_markers.push_back(piece);
            }
        }
    }
    ctx->output_markers.erase(std::remove(ctx->output_markers.begin(), ctx->output_markers.end(), ""), ctx->output_markers.end());
    return llama_sampler_init(&common_reasoning_budget_i, ctx);
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

bool common_reasoning_budget_is_token_allowed(const struct llama_sampler * smpl, llama_token token) {
    if (!smpl) {
        return true;
    }
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->state == REASONING_BUDGET_FORCING) {
        return !std::binary_search(ctx->end_tokens.begin(), ctx->end_tokens.end(), token) &&
               (ctx->force_pos >= ctx->forced_tokens.size() || token == ctx->forced_tokens[ctx->force_pos]);
    }
    const bool boundary = ctx->state == REASONING_BUDGET_DONE && ctx->awaiting_output;
    if (ctx->state != REASONING_BUDGET_COUNTING && ctx->state != REASONING_BUDGET_WAITING_UTF8 && !boundary) {
        return true;
    }
    return !std::binary_search(ctx->end_tokens.begin(), ctx->end_tokens.end(), token) &&
           !(boundary && std::binary_search(ctx->closing_tokens.begin(), ctx->closing_tokens.end(), token));
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
