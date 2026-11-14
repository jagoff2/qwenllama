#include "arg.h"
#include "common.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <limits>

using json = nlohmann::ordered_json;

static std::string fingerprint_bytes(const void * memory, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(memory);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return std::to_string(hash);
}


struct loop_schedule_interval {
    int start;
    int end;
    json config;
};

static int schedule_bound(const json & value, const char * name) {
    if (!value.is_number_integer()) {
        throw std::runtime_error(std::string("loop_schedule ") + name + " must be an integer");
    }
    const double number = value.get<double>();
    if (number < 0 || number > 255) {
        throw std::runtime_error(std::string("loop_schedule ") + name + " must be in 0..255");
    }
    return value.get<int>();
}

static json schedule_config(const json & value) {
    if (!value.is_array() || value.size() < 7 || value.size() > 11) {
        throw std::runtime_error("scheduled config must contain 7..11 numbers");
    }
    json result = json::array();
    for (size_t i = 0; i < value.size(); ++i) {
        if (!value[i].is_number()) throw std::runtime_error("scheduled config entries must be numbers");
        const double number = value[i].get<double>();
        if (!std::isfinite(number)) throw std::runtime_error("scheduled config entries must be finite");
        if (i < 3 || i >= 6) {
            if (std::trunc(number) != number || number < double(INT32_MIN) || number > double(INT32_MAX)) {
                throw std::runtime_error("scheduled config integer fields must be integral int32 values");
            }
            result.push_back(int32_t(number));
        } else {
            if (std::abs(number) > double(std::numeric_limits<float>::max())) {
                throw std::runtime_error("scheduled config float fields must fit finite float32 values");
            }
            result.push_back(value[i]);
        }
    }
    return result;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    auto mp = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto cp = common_context_params_to_llama(params);
    cp.n_seq_max = 1;
    cp.n_rs_seq = 0;
    cp.n_outputs_max = 1;
    cp.n_outputs_max_per_seq = 1;
    std::cout << json({{"ready", true}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        llama_context * ctx = nullptr;
        llama_sampler * grammar = nullptr;
        try {
            auto request = json::parse(line);
            auto tokens = common_tokenize(vocab, request.at("prompt").get<std::string>(), false, true);
            if (request.value("tokenize_only", false)) {
                std::cout << json({{"tokens", tokens.size()}}).dump() << std::endl;
                continue;
            }
            std::vector<loop_schedule_interval> schedule;
            if (request.contains("loop_schedule")) {
                const auto & supplied = request.at("loop_schedule");
                if (!supplied.is_array()) throw std::runtime_error("loop_schedule must be an array");
                for (const auto & entry : supplied) {
                    if (!entry.is_object() || !entry.contains("start") || !entry.contains("end") || !entry.contains("config")) {
                        throw std::runtime_error("loop_schedule entries require start, end, and config");
                    }
                    const int start = schedule_bound(entry.at("start"), "start");
                    const int end = schedule_bound(entry.at("end"), "end");
                    if (start > end) throw std::runtime_error("loop_schedule start must not exceed end");
                    schedule.push_back({start, end, schedule_config(entry.at("config"))});
                }
                std::sort(schedule.begin(), schedule.end(), [](const auto & a, const auto & b) {
                    return a.start < b.start;
                });
                for (size_t i = 1; i < schedule.size(); ++i) {
                    if (schedule[i].start <= schedule[i-1].end) {
                        throw std::runtime_error("loop_schedule intervals must not overlap (endpoints are inclusive)");
                    }
                }
            }
            const bool schedule_active = !schedule.empty();
            const char * config_file = std::getenv("QWEN4EXP_LOOP_CONFIG");
            if (schedule_active && (!config_file || !*config_file)) {
                throw std::runtime_error("nonempty loop_schedule requires QWEN4EXP_LOOP_CONFIG");
            }
            // Missing/empty schedules preserve the original config serialization and decode path.
            json default_config;
            if (schedule_active) default_config = schedule_config(request.at("config"));
            json prediction_configs = json::array();
            json effective_prediction_configs = json::array();
            auto write_prediction_config = [&](size_t prediction_index) {
                const json * chosen = schedule_active ? &default_config : nullptr;
                int interval_index = -1;
                if (schedule_active) {
                    for (size_t i = 0; i < schedule.size(); ++i) {
                        if (prediction_index >= size_t(schedule[i].start) && prediction_index <= size_t(schedule[i].end)) {
                            chosen = &schedule[i].config;
                            interval_index = int(i);
                            break;
                        }
                    }
                }
                if (config_file) {
                    std::ofstream out(config_file);
                    for (const auto & v : (chosen ? *chosen : request.at("config"))) out << v << ' ';
                    out << '\n';
                    out.close();
                    if (!out) throw std::runtime_error("cannot write loop config");
                }
                if (schedule_active) {
                    prediction_configs.push_back({{"prediction_index", prediction_index},
                            {"interval_index", interval_index}, {"config", *chosen}});
                    effective_prediction_configs.push_back(*chosen);
                }
            };
            // Prediction 0 is produced by prompt decode, including its context-reserve graph.
            write_prediction_config(0);
            const auto t0 = std::chrono::steady_clock::now();
            ctx = llama_init_from_model(model, cp);
            if (!ctx) throw std::runtime_error("context creation failed");
            // Zero actual state bytes, including any recycled allocation containing NaNs.
            llama_memory_clear(llama_get_memory(ctx), true);
            if (tokens.empty() || tokens.size() > cp.n_ubatch) throw std::runtime_error("prompt must fit one microbatch");
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
                throw std::runtime_error("decode failed");
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            const int n_vocab = llama_vocab_n_tokens(vocab);
            if (!logits) throw std::runtime_error("no logits");
            for (int i = 0; i < n_vocab; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            uint64_t fingerprint = 14695981039346656037ULL;
            const uint8_t * logit_bytes = reinterpret_cast<const uint8_t *>(logits);
            for (size_t j = 0; j < size_t(n_vocab)*sizeof(float); ++j) {
                fingerprint ^= logit_bytes[j];
                fingerprint *= 1099511628211ULL;
            }
            json scores = json::object();
            float best = -INFINITY;
            std::string answer;
            for (char ch : std::string("ABCD")) {
                std::string label(1, ch);
                auto ids = common_tokenize(vocab, label, false, false);
                if (ids.size() != 1) throw std::runtime_error("label is not one token");
                scores[label] = logits[ids[0]];
                if (logits[ids[0]] > best) { best = logits[ids[0]]; answer = label; }
            }
            int top = (int) (std::max_element(logits, logits + n_vocab) - logits);
            double exp_sum = 0;
            for (int j = 0; j < n_vocab; ++j) exp_sum += std::exp(double(logits[j] - logits[top]));
            double label_mass = 0;
            for (const auto & score : scores.items()) label_mass += std::exp(score.value().get<double>() - logits[top]) / exp_sum;
            json result = {{"answer", answer}, {"logits", scores}, {"tokens", tokens.size()},
                           {"top_token", top}, {"top_piece", common_token_to_piece(ctx, top)},
                           {"label_probability_mass", label_mass},
                           {"logits_fingerprint_fnv1a64", std::to_string(fingerprint)},
                           {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()}};
            // Development-only likelihood scoring; never label this as generation.
            // Correct prior target tokens are supplied by the evaluator here.
            if (request.contains("score_text")) {
                if (request.contains("max_new_tokens")) throw std::runtime_error("scoring and generation are separate requests");
                auto targets = common_tokenize(vocab, request.at("score_text").get<std::string>(), false, false);
                if (targets.empty() || tokens.size() + targets.size() > cp.n_ctx) {
                    throw std::runtime_error("invalid teacher-forced target length");
                }
                double log_probability = 0;
                json per_token = json::array();
                std::vector<llama_token> digit_ids;
                for (char digit = '0'; digit <= '9'; ++digit) {
                    auto ids = common_tokenize(vocab, std::string(1,digit), false, false);
                    if (ids.size() != 1) throw std::runtime_error("digit is not one token");
                    digit_ids.push_back(ids[0]);
                }
                for (size_t step = 0; step < targets.size(); ++step) {
                    logits = llama_get_logits_ith(ctx, -1);
                    for (int j = 0; j < n_vocab; ++j) if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite scored logits");
                    const int best_id = (int) (std::max_element(logits, logits+n_vocab)-logits);
                    const double maximum = logits[best_id];
                    double denominator = 0;
                    for (int j = 0; j < n_vocab; ++j) denominator += std::exp(double(logits[j])-maximum);
                    const double lp = double(logits[targets[step]])-maximum-std::log(denominator);
                    log_probability += lp;
                    int rank = 1, digit_rank = 1;
                    json digit_logits = json::array();
                    for (int j = 0; j < n_vocab; ++j) rank += logits[j] > logits[targets[step]];
                    for (const auto id : digit_ids) {
                        digit_rank += logits[id] > logits[targets[step]];
                        digit_logits.push_back(logits[id]);
                    }
                    per_token.push_back({{"token", targets[step]}, {"piece", common_token_to_piece(ctx,targets[step])},
                                         {"log_probability", lp}, {"greedy_token", best_id},
                                         {"logits_fingerprint_fnv1a64", fingerprint_bytes(logits, size_t(n_vocab)*sizeof(float))},
                                         {"target_rank", rank}, {"digit_rank", digit_rank}, {"digit_logits",digit_logits}});
                    if (step+1 < targets.size()) {
                        if (schedule_active) write_prediction_config(step+1);
                        if (llama_decode(ctx,llama_batch_get_one(&targets[step],1))) {
                            throw std::runtime_error("teacher-forcing decode failed");
                        }
                    }
                }
                result["teacher_forced"] = true;
                result["target_log_probability"] = log_probability;
                result["target_tokens"] = per_token;
                result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            }
            if (request.contains("max_new_tokens")) {
                int limit = request.at("max_new_tokens").get<int>();
                if (limit < 1 || tokens.size() + limit > cp.n_ctx) throw std::runtime_error("generation budget exceeds context");
                std::string generated;
                std::vector<llama_token> generated_ids;
                std::vector<llama_token_data> candidates(n_vocab);
                if (request.contains("grammar")) {
                    const std::string grammar_text = request.at("grammar").get<std::string>();
                    grammar = llama_sampler_init_grammar(vocab, grammar_text.c_str(), "root");
                    if (!grammar) throw std::runtime_error("grammar initialization failed");
                }
                bool stopped = false;
                for (int step = 0; step < limit; ++step) {
                    logits = llama_get_logits_ith(ctx, -1);
                    for (int j = 0; j < n_vocab; ++j) if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite generated logits");
                    llama_token token;
                    if (grammar) {
                        for (int j = 0; j < n_vocab; ++j) candidates[j] = {j, logits[j], 0};
                        llama_token_data_array data = {candidates.data(), candidates.size(), -1, false};
                        llama_sampler_apply(grammar, &data);
                        auto selected = std::max_element(candidates.begin(), candidates.end(),
                                [](const auto & a, const auto & b) { return a.logit < b.logit; });
                        if (!std::isfinite(selected->logit)) throw std::runtime_error("no finite grammar candidate");
                        token = selected->id;
                        llama_sampler_accept(grammar, token);
                    } else {
                        token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
                    }
                    if (llama_vocab_is_eog(vocab, token)) { stopped = true; break; }
                    generated += common_token_to_piece(ctx, token);
                    generated_ids.push_back(token);
                    if (step + 1 < limit) {
                        if (schedule_active) write_prediction_config(size_t(step)+1);
                        if (llama_decode(ctx, llama_batch_get_one(&token, 1))) {
                            throw std::runtime_error("generation decode failed");
                        }
                    }
                }
                result["generated_text"] = generated;
                result["generated_ids"] = generated_ids;
                result["stopped_eog"] = stopped;
                result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                if (grammar) { llama_sampler_free(grammar); grammar = nullptr; }
            }
            if (schedule_active) {
                json normalized_schedule = json::array();
                for (const auto & entry : schedule) {
                    normalized_schedule.push_back({{"start", entry.start}, {"end", entry.end}, {"config", entry.config}});
                }
                result["loop_schedule"] = normalized_schedule;
                result["loop_default_config"] = default_config;
                result["prediction_configs"] = prediction_configs;
                result["effective_prediction_configs"] = effective_prediction_configs;
            }
            if (request.value("capture_state", false)) {
                const size_t state_size = llama_state_get_size(ctx);
                std::vector<uint8_t> state(state_size);
                const size_t written = llama_state_get_data(ctx, state.data(), state.size());
                if (written != state_size) throw std::runtime_error("serialized state size changed");
                result["state_bytes"] = state_size;
                result["state_fingerprint_fnv1a64"] = fingerprint_bytes(state.data(), state.size());
            }
            llama_free(ctx);
            ctx = nullptr;
            std::cout << result.dump() << std::endl;
        } catch (const std::exception & e) {
            if (grammar) llama_sampler_free(grammar);
            if (ctx) llama_free(ctx);
            std::cout << json({{"error", e.what()}}).dump() << std::endl;
        }
    }
    llama_model_free(model);
    llama_backend_free();
}
