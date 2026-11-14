#include "arg.h"
#include "common.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
using json = nlohmann::ordered_json;
int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    auto mp = common_model_params_to_llama(params);
    auto * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) return 2;
    auto cp = common_context_params_to_llama(params);
    cp.n_seq_max = 1; cp.n_rs_seq = 0; cp.n_outputs_max = 1; cp.n_outputs_max_per_seq = 1;
    const auto * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::cout << json({{"ready",true},{"vocab",nv}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin,line)) {
        llama_context * ctx = nullptr;
        try {
            const auto req = json::parse(line);
            const auto started = std::chrono::steady_clock::now();
            const char * path = std::getenv("QWEN4EXP_LOOP_CONFIG");
            if (!path) throw std::runtime_error("missing config path");
            { std::ofstream f(path); for (const auto & x : req.at("config")) f << x << ' '; f << '\n'; }
            ctx = llama_init_from_model(model,cp);
            if (!ctx) throw std::runtime_error("context init");
            llama_memory_clear(llama_get_memory(ctx),true);
            auto prompt = common_tokenize(vocab,req.at("prompt").get<std::string>(),false,true);
            auto continuation = common_tokenize(vocab,req.value("continuation",std::string("1, 2, 3, 4, 5, 6, 7, 8")),false,true);
            if (prompt.empty() || prompt.size() > cp.n_ubatch) throw std::runtime_error("prompt length");
            if (llama_decode(ctx,llama_batch_get_one(prompt.data(),(int)prompt.size()))) throw std::runtime_error("prefill");
            std::ofstream output(req.at("logits_file").get<std::string>(),std::ios::binary);
            if (!output) throw std::runtime_error("open logits");
            json top_ids=json::array();
            json fingerprints=json::array();
            for (size_t step=0;step<=continuation.size();step++) {
                const float * logits=llama_get_logits_ith(ctx,-1);
                if (!logits) throw std::runtime_error("no logits");
                for (int j=0;j<nv;j++) if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite logits");
                output.write(reinterpret_cast<const char *>(logits),nv*sizeof(float));
                uint64_t hash=14695981039346656037ULL;
                const auto * bytes=reinterpret_cast<const unsigned char *>(logits);
                for (size_t j=0;j<nv*sizeof(float);j++) {hash ^= bytes[j]; hash *= 1099511628211ULL;}
                fingerprints.push_back(std::to_string(hash));
                top_ids.push_back(std::max_element(logits,logits+nv)-logits);
                if (step<continuation.size() && llama_decode(ctx,llama_batch_get_one(&continuation[step],1))) throw std::runtime_error("continuation");
            }
            output.close();
            if (!output) throw std::runtime_error("write logits");
            llama_synchronize(ctx);
            auto result=json({{"id",req.at("id")},{"vocab",nv},{"prompt_tokens",prompt.size()},
                {"graph_reused",llama_perf_context(ctx).n_reused},
                {"rows",continuation.size()+1},{"top_ids",top_ids},{"hashes",fingerprints},
                {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}});
            llama_free(ctx); ctx=nullptr;
            std::cout<<result.dump()<<std::endl;
        } catch (const std::exception & e) {
            if (ctx) llama_free(ctx);
            std::cout<<json({{"error",e.what()}}).dump()<<std::endl;
        }
    }
    llama_model_free(model); llama_backend_free();
}
