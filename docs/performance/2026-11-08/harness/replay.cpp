#include "arg.h"
#include "common.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

using json = nlohmann::ordered_json;
static int64_t now_us() {
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (int64_t) (double(q.QuadPart) * 1e6 / f.QuadPart);
}

int main(int argc, char ** argv) {
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER, nullptr)) return 1;
    params.warmup = false;
    llama_backend_init();
    llama_numa_init(params.numa);
    const char * config_path = getenv("DECODE_REPLAY_CONFIG");
    if (!config_path) return 2;
    json config; std::ifstream(config_path) >> config;
    const auto prompt = config.at("prompt").get<std::vector<llama_token>>();
    const auto replay = config.at("tokens").get<std::vector<llama_token>>();
    const int warm = config.value("warm", 64);
    auto init = common_init_from_params(params, true);
    auto * model = init->model();
    if (!model) return 3;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> reference;
    const std::string reference_path = config.at("reference");
    {
        std::ifstream f(reference_path, std::ios::binary | std::ios::ate);
        if (f) {
            const size_t bytes = f.tellg();
            if (bytes != replay.size() * nv * sizeof(float)) return 4;
            reference.resize(bytes / sizeof(float));
            f.seekg(0); f.read((char *) reference.data(), bytes);
        }
    }
    for (const auto & run : config.at("runs")) {
        if (run.contains("phase_layout")) _putenv_s("LLAMA_MOE_CACHE_PHASE_LAYOUT", run.at("phase_layout").get<bool>() ? "1" : "0");
        if (run.contains("admission_mib")) _putenv_s("LLAMA_MOE_CACHE_ADMISSION_MIB", std::to_string(run.at("admission_mib").get<int>()).c_str());
        if (run.contains("pipeline")) _putenv_s("GGML_CUDA_CACHE_PIPELINE", run.at("pipeline").get<bool>() ? "1" : "0");
        if (run.contains("compact")) _putenv_s("LLAMA_FN_CACHE_COMPACT_DECODE", run.at("compact").get<bool>() ? "1" : "0");
        if (run.contains("batch_d2h")) _putenv_s("GGML_SCHED_BATCH_D2H", run.at("batch_d2h").get<bool>() ? "1" : "0");
        if (run.contains("serial")) _putenv_s("GGML_SCHED_SERIAL_CACHE_PREP", run.at("serial").get<bool>() ? "1" : "0");
        params.cpuparams.n_threads = run.at("threads");
        params.cpuparams_batch.n_threads = 24;
        auto cp = common_context_params_to_llama(params);
        if (run.contains("cache_slots")) _putenv_s("LLAMA_MOE_CACHE_SLOTS", std::to_string(run.at("cache_slots").get<int>()).c_str());
        if (run.contains("stream_mode")) cp.moe_stream_gpu_mode = run.at("stream_mode") == "layer" ? 0 : 1;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) return 5;
        json result = run;
        result["tokens"] = json::array();
        {
            common_threadpools pools;
            pools.init(ctx, params);
            auto batch = llama_batch_init(params.n_batch, 0, 1);
            const int64_t prefill_start = now_us();
            for (size_t first = 0; first < prompt.size(); first += params.n_batch) {
                common_batch_clear(batch);
                const size_t end = std::min(prompt.size(), first + params.n_batch);
                for (size_t i = first; i < end; ++i) common_batch_add(batch, prompt[i], i, {0}, i + 1 == prompt.size());
                if (llama_decode_with_purpose(ctx, batch, LLAMA_DECODE_PROMPT)) return 6;
                llama_synchronize(ctx);
            }
            result["prefill_us"] = now_us() - prefill_start;
            const bool save_reference = reference.empty();
            if (save_reference) reference.resize(replay.size() * nv);
            double measured_us = 0;
            double max_delta = 0, sq_delta = 0, sq_ref = 0;
            size_t different = 0, argmax_different = 0, nonfinite = 0;
            for (size_t i = 0; i < replay.size(); ++i) {
                common_batch_clear(batch);
                common_batch_add(batch, replay[i], prompt.size() + i, {0}, true);
                const int64_t start = now_us();
                if (llama_decode_with_purpose(ctx, batch, LLAMA_DECODE_GENERATION)) return 7;
                const float * logits = llama_get_logits_ith(ctx, 0);
                const int64_t end = now_us();
                if (i >= size_t(warm)) measured_us += end - start;
                float * ref = reference.data() + i * nv;
                if (save_reference) std::copy(logits, logits + nv, ref);
                const int best = std::max_element(logits, logits + nv) - logits;
                const int ref_best = std::max_element(ref, ref + nv) - ref;
                argmax_different += best != ref_best;
                double token_max = 0;
                for (int j = 0; j < nv; ++j) {
                    const double d = double(logits[j]) - ref[j];
                    if (!std::isfinite(d)) { ++nonfinite; continue; }
                    token_max = std::max(token_max, std::abs(d));
                    sq_delta += d * d; sq_ref += double(ref[j]) * ref[j];
                    different += logits[j] != ref[j];
                }
                max_delta = std::max(max_delta, token_max);
                result["tokens"].push_back({{"index", i}, {"input", replay[i]}, {"greedy", best},
                    {"start_us", start}, {"end_us", end}, {"us", end-start}, {"max_logit_delta", token_max}});
            }
            result["measured_tokens"] = replay.size() - warm;
            result["measured_us"] = measured_us;
            result["tps"] = (replay.size() - warm) * 1e6 / measured_us;
            result["max_logit_delta"] = max_delta;
            result["relative_l2"] = std::sqrt(sq_delta / std::max(sq_ref, 1e-30));
            result["different_logits"] = different;
            result["argmax_different"] = argmax_different;
            result["nonfinite"] = nonfinite;
            result["first_decode_us"] = result["tokens"][0]["us"];
            result["prefill_plus_first_decode_us"] = result["prefill_us"].get<int64_t>() + result["first_decode_us"].get<int64_t>();
            if (save_reference) {
                std::ofstream f(reference_path, std::ios::binary);
                f.write((const char *) reference.data(), reference.size() * sizeof(float));
            }
            size_t next_position = prompt.size() + replay.size();
            if (config.value("reentry", false)) {
                result["reentry"] = json::array();
                for (const int requested_rows : {49, 4096}) {
                    const int rows = std::min(requested_rows, params.n_batch);
                    const int64_t reentry_start = now_us();
                    common_batch_clear(batch);
                    for (int i = 0; i < rows; ++i) common_batch_add(batch, prompt[i % prompt.size()], next_position+i, {0}, i+1 == rows);
                    if (llama_decode_with_purpose(ctx, batch, LLAMA_DECODE_PROMPT)) return 9;
                    llama_synchronize(ctx);
                    json stage = {{"prompt_tokens", rows}, {"prefill_us", now_us()-reentry_start}};
                    next_position += rows;
                    const std::string stage_path = reference_path + ".reentry-" + std::to_string(rows);
                    std::vector<float> stage_reference(size_t(64) * nv);
                    std::ifstream reference_file(stage_path, std::ios::binary | std::ios::ate);
                    const bool save_stage = !reference_file;
                    if (!save_stage) {
                        if (size_t(reference_file.tellg()) != stage_reference.size() * sizeof(float)) return 12;
                        reference_file.seekg(0);
                        reference_file.read((char *) stage_reference.data(), stage_reference.size() * sizeof(float));
                    }
                    double stage_sq_delta = 0, stage_sq_ref = 0, stage_max = 0;
                    int stage_argmax = 0;
                    int64_t stage_decode_us = 0;
                    for (int i = 0; i < 64; ++i) {
                        common_batch_clear(batch);
                        common_batch_add(batch, replay[i % replay.size()], next_position++, {0}, true);
                        const int64_t started = now_us();
                        if (llama_decode_with_purpose(ctx, batch, LLAMA_DECODE_GENERATION)) return 10;
                        const float * logits = llama_get_logits_ith(ctx, 0);
                        const int64_t elapsed = now_us() - started;
                        if (i == 0) stage["first_decode_us"] = elapsed;
                        if (i >= 32) stage_decode_us += elapsed;
                        float * ref = stage_reference.data() + size_t(i) * nv;
                        if (save_stage) std::copy(logits, logits + nv, ref);
                        stage_argmax += (std::max_element(logits, logits + nv) - logits) !=
                                        (std::max_element(ref, ref + nv) - ref);
                        for (int j = 0; j < nv; ++j) {
                            if (!std::isfinite(logits[j])) return 11;
                            const double delta = double(logits[j]) - ref[j];
                            stage_sq_delta += delta * delta;
                            stage_sq_ref += double(ref[j]) * ref[j];
                            stage_max = std::max(stage_max, std::abs(delta));
                        }
                    }
                    if (save_stage) {
                        std::ofstream f(stage_path, std::ios::binary);
                        f.write((const char *) stage_reference.data(), stage_reference.size() * sizeof(float));
                    }
                    stage["decode_tokens"] = 64;
                    stage["last32_tps"] = 32e6 / stage_decode_us;
                    stage["relative_l2"] = std::sqrt(stage_sq_delta / std::max(stage_sq_ref, 1e-30));
                    stage["max_logit_delta"] = stage_max;
                    stage["argmax_different"] = stage_argmax;
                    stage["nonfinite"] = 0;
                    result["reentry"].push_back(stage);
                }
            }
            if (config.value("flush_profiles", false) && run == config.at("runs").back()) {
                for (const char * variable : {"GGML_HANDOFF_PROFILE", "GGML_CPU_OP_PROFILE"}) {
                    const char * path = getenv(variable);
                    if (path) std::ofstream(std::string(path) + ".flush") << "flush outside timing\n";
                }
                for (int flush = 0; flush < 4; ++flush) {
                    common_batch_clear(batch);
                    common_batch_add(batch, replay.back(), next_position++, {0}, true);
                    if (llama_decode_with_purpose(ctx, batch, LLAMA_DECODE_GENERATION)) return 8;
                }
                llama_synchronize(ctx);
            }
            llama_batch_free(batch);
        }
        const std::string output = run.at("output");
        std::ofstream(output) << result.dump(2) << '\n';
        std::cout << "REPLAY_RESULT " << result.at("label") << " tps=" << result.at("tps")
                  << " max_delta=" << result.at("max_logit_delta") << std::endl;
        llama_free(ctx);
    }
    init.reset();
    llama_backend_free();
    return 0;
}
