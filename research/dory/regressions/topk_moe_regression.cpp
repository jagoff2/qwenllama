#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
constexpr int EXPERTS = 512, SELECTED = 10;
struct result { std::vector<float> weights; std::vector<int32_t> ids; json layout; };
struct resources {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t ordinary = nullptr, aliases = nullptr;
    ~resources() {
        if (ordinary) ggml_backend_buffer_free(ordinary);
        if (aliases) ggml_backend_buffer_free(aliases);
        if (ctx) ggml_free(ctx);
    }
};
static ggml_tensor * root(ggml_tensor * t) { while (t->view_src) t = t->view_src; return t; }
static uint32_t bits(float f) { uint32_t v; std::memcpy(&v, &f, 4); return v; }
static uint32_t ordered(float f) { const uint32_t v = bits(f); return v & 0x80000000u ? ~v : v | 0x80000000u; }
static uint64_t ulps(float a, float b) {
    const uint64_t x = ordered(a), y = ordered(b);
    return x > y ? x-y : y-x;
}
static json compare(const result & actual, const result & expected) {
    if (actual.weights.size() != expected.weights.size() || actual.ids.size() != expected.ids.size())
        throw std::runtime_error("reference shape mismatch");
    size_t nonfinite = 0, different_bits = 0, id_mismatches = 0;
    double max_abs = 0, max_relative = 0, error2 = 0, reference2 = 0;
    uint64_t max_ulps = 0;
    for (size_t i=0; i<actual.weights.size(); ++i) {
        const float a=actual.weights[i], e=expected.weights[i];
        if (!std::isfinite(a) || !std::isfinite(e)) { ++nonfinite; continue; }
        const double delta = double(a)-e;
        max_abs=std::max(max_abs,std::abs(delta));
        max_relative=std::max(max_relative,std::abs(delta)/std::max(std::abs(double(e)),1e-30));
        max_ulps=std::max(max_ulps,ulps(a,e));
        different_bits += bits(a) != bits(e);
        error2 += delta*delta; reference2 += double(e)*e;
    }
    for(size_t i=0;i<actual.ids.size();++i) id_mismatches += actual.ids[i] != expected.ids[i];
    return {{"nonfinite",nonfinite},{"id_mismatches",id_mismatches},{"weight_bit_mismatches",different_bits},
            {"max_abs",max_abs},{"max_relative",max_relative},{"max_ulp",max_ulps},
            {"nmse",error2/std::max(reference2,1e-30)},
            {"bitwise_equal",nonfinite==0 && id_mismatches==0 && different_bits==0},
            {"within_tolerance",nonfinite==0 && id_mismatches==0 && max_abs<2e-5}};
}

static result run(ggml_backend_t backend, int rows, bool bias_on, bool norm_on,
                  const std::string & gate, const std::string & alias, const std::string & profile,
                  bool trailing_scale) {
    resources r;
    r.ctx=ggml_init({2*1024*1024,nullptr,true});
    if(!r.ctx) throw std::runtime_error("ggml_init failed");
    auto * raw=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,EXPERTS,rows);
    // Computed logits exercise the real overlap checker, which skips NONE nodes.
    auto * logits=ggml_scale(r.ctx,raw,1.0f);
    ggml_set_name(logits,"computed_logits");
    auto * probs=gate=="softmax" ? ggml_soft_max(r.ctx,logits) :
                 ggml_sqrt(r.ctx,ggml_softplus(r.ctx,logits));
    auto * bias=ggml_new_tensor_1d(r.ctx,GGML_TYPE_F32,EXPERTS);
    auto * selection=bias_on ? ggml_add(r.ctx,probs,bias) : probs;
    auto * ids=ggml_argsort_top_k(r.ctx,selection,SELECTED);
    auto * weights=ggml_get_rows(r.ctx,ggml_reshape_3d(r.ctx,probs,1,EXPERTS,rows),ids);
    if(norm_on) {
        weights=ggml_reshape_2d(r.ctx,weights,SELECTED,rows);
        auto * sum=ggml_sum_rows(r.ctx,weights);
        sum=ggml_clamp(r.ctx,sum,6.103515625e-5f,INFINITY);
        weights=ggml_reshape_3d(r.ctx,ggml_div(r.ctx,weights,sum),1,SELECTED,rows);
    }
    // Default deliberately ends on the normalization RESHAPE: this catches the
    // CUDA matcher's former end-of-graph overread. Use this identity tail only
    // when comparing numerical results against a DLL with that matcher bug.
    if(trailing_scale) weights=ggml_scale(r.ctx,weights,1.0f);
    // Only the selected-ID view and final weights are observable. The public
    // set_output helper also marks view roots, demanding the entire argsort
    // buffer and intentionally preventing top-k fusion. This fixture reads
    // only the ten selected IDs, matching the model's downstream consumers.
    ids->flags |= GGML_TENSOR_FLAG_OUTPUT;
    weights->flags |= GGML_TENSOR_FLAG_OUTPUT;
    ggml_set_name(ids,"selected_ids"); ggml_set_name(weights,"selected_weights");
    auto * graph=ggml_new_graph_custom(r.ctx,128,false);
    ggml_build_forward_expand(graph,weights);
    // Standalone backend execution bypasses the scheduler, which normally marks
    // compute nodes before CUDA's fusion eligibility checks.
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)
        ggml_graph_node(graph,i)->flags |= GGML_TENSOR_FLAG_COMPUTE;
    const size_t logits_bytes=size_t(EXPERTS)*rows*sizeof(float);
    const size_t weights_bytes=size_t(SELECTED)*rows*sizeof(float);
    // The selected-ID view occupies 40 bytes at each 2048-byte row stride.
    // In the combined layout weights start after the last selected-ID row.
    const size_t combined_offset=size_t(rows-1)*EXPERTS*sizeof(float)+512;
    json layout={{"alias",alias},{"logits_bytes",logits_bytes},{"weights_bytes",weights_bytes}};
    if(alias!="separate") {
        r.aliases=ggml_backend_alloc_buffer(backend,logits_bytes+weights_bytes+4096);
        if(!r.aliases) throw std::runtime_error("alias allocation failed");
        auto * base=static_cast<char *>(ggml_backend_buffer_get_base(r.aliases));
        auto bind=[&](ggml_tensor * t,size_t offset) {
            if(ggml_backend_tensor_alloc(r.aliases,root(t),base+offset)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("alias tensor binding failed");
        };
        bind(logits,0);
        if(alias=="logits_ids" || alias=="logits_both_disjoint") bind(ids,0);
        if(alias=="logits_weights") bind(weights,0);
        if(alias=="logits_both_disjoint") bind(weights,combined_offset);
        layout["weights_offset"]=alias=="logits_both_disjoint" ? combined_offset : 0;
        layout["selected_ids_and_weights_disjoint"]=true;
    }
    r.ordinary=ggml_backend_alloc_ctx_tensors(r.ctx,backend);
    if(!r.ordinary) throw std::runtime_error("graph allocation failed");
    int checked_tensors=0,checked_views=0;
    for(auto * t=ggml_get_first_tensor(r.ctx);t;t=ggml_get_next_tensor(r.ctx,t)) {
        if(!t->buffer || !t->data)
            throw std::runtime_error(std::string("unbound graph tensor: ")+ggml_get_name(t));
        ++checked_tensors;
        if(t->view_src) {
            if(t->buffer!=t->view_src->buffer ||
               t->data!=static_cast<char *>(t->view_src->data)+t->view_offs)
                throw std::runtime_error("incorrect backend view binding");
            ++checked_views;
        }
    }
    const uintptr_t weight_begin=reinterpret_cast<uintptr_t>(weights->data);
    const uintptr_t weight_end=weight_begin+weights_bytes;
    for(int row=0;row<rows;++row) {
        const uintptr_t id_begin=reinterpret_cast<uintptr_t>(ids->data)+size_t(row)*ids->nb[1];
        const uintptr_t id_end=id_begin+SELECTED*sizeof(int32_t);
        if(id_begin<weight_end && weight_begin<id_end)
            throw std::runtime_error("simultaneously live weights and selected IDs overlap");
    }
    layout["checked_tensors"]=checked_tensors;
    layout["checked_views"]=checked_views;
    layout["selected_ids_and_weights_disjoint"]=true;
    ggml_backend_buffer_clear(r.ordinary,0);
    if(r.aliases) ggml_backend_buffer_clear(r.aliases,0);
    std::vector<float> values(size_t(EXPERTS)*rows), biases(EXPERTS);
    for(int e=0;e<EXPERTS;++e) biases[e]=float((e*37)%101-50)*0.00013f;
    for(int row=0;row<rows;++row) {
        for(int e=0;e<EXPERTS;++e) {
            const int permutation=(e*331+row*37)%EXPERTS;
            values[size_t(row)*EXPERTS+e]=-4.0f+8.0f*float(permutation)/511.0f;
        }
        if(profile=="near_tie") {
            // Nine clear winners; a tenth-place pair separated by one F32 ULP.
            for(int k=0;k<9;++k) values[size_t(row)*EXPERTS+(k*47+row*13)%EXPERTS]=8.0f+0.125f*k;
            values[size_t(row)*EXPERTS+510]=7.0f;
            values[size_t(row)*EXPERTS+511]=std::nextafter(7.0f,INFINITY);
        }
    }
    ggml_backend_tensor_set(raw,values.data(),0,values.size()*sizeof(float));
    ggml_backend_tensor_set(bias,biases.data(),0,biases.size()*sizeof(float));
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed");
    ggml_backend_synchronize(backend);
    result out;
    out.layout=layout;
    out.weights.resize(size_t(SELECTED)*rows);
    out.ids.resize(size_t(SELECTED)*rows);
    ggml_backend_tensor_get(weights,out.weights.data(),0,out.weights.size()*sizeof(float));
    // ids is a strided top-k VIEW, not a compact [10,rows] allocation.
    for(int row=0;row<rows;++row)
        ggml_backend_tensor_get(ids,out.ids.data()+size_t(row)*SELECTED,
                                size_t(row)*ids->nb[1],SELECTED*sizeof(int32_t));
    return out;
}

int main(int argc,char **argv) {
    ggml_backend_t cpu=nullptr,target=nullptr;
    try {
        std::string backend="CUDA0",fusion="on",gate_filter="all",profile="ordinary",reference_path;
        bool trailing_scale=false;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if(i+1>=argc) throw std::runtime_error("option needs a value");
            const std::string value=argv[++i];
            if(key=="--backend") backend=value;
            else if(key=="--fusion") fusion=value;
            else if(key=="--gate") gate_filter=value;
            else if(key=="--profile") profile=value;
            else if(key=="--reference") reference_path=value;
            else if(key=="--trailing-scale") {
                if(value!="on" && value!="off") throw std::runtime_error("invalid trailing-scale value");
                trailing_scale=value=="on";
            }
            else throw std::runtime_error("unknown option: "+key);
        }
        if((fusion!="on"&&fusion!="off") || (gate_filter!="all"&&gate_filter!="softmax"&&gate_filter!="sqrt_softplus") ||
           (profile!="ordinary"&&profile!="near_tie")) throw std::runtime_error("invalid option value");
        // Fresh processes are required: CUDA fusion settings are cached statically.
        _putenv_s("GGML_CUDA_DISABLE_FUSION",fusion=="off"?"1":"0");
        _putenv_s("GGML_CUDA_DISABLE_TOPK_FUSION","");
        _putenv_s("GGML_CUDA_DISABLE_FUSION_OPS","");
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
        std::map<std::string,result> references;
        if(!reference_path.empty()) {
            std::ifstream in(reference_path);
            if(!in) throw std::runtime_error("cannot open explicit reference");
            std::string line;
            while(std::getline(in,line)) {
                if(line.empty()) continue;
                const auto entry=json::parse(line);
                if(!entry.contains("key")) continue;
                result ref;
                ref.weights=entry.at("weights").get<std::vector<float>>();
                ref.ids=entry.at("ids").get<std::vector<int32_t>>();
                if(!references.emplace(entry.at("key").get<std::string>(),std::move(ref)).second)
                    throw std::runtime_error("duplicate reference key");
            }
        }
        // Bundled runtime registers CPU/CUDA backends statically; no dynamic loader.
        cpu=ggml_backend_init_by_name("CPU",nullptr);
        target=ggml_backend_init_by_name(backend.c_str(),nullptr);
        if(!cpu||!target) throw std::runtime_error("backend unavailable");
        int count=0,failed=0,reference_nonidentical=0;
        for(const auto & gate:{std::string("softmax"),std::string("sqrt_softplus")}) {
            if(gate_filter!="all"&&gate_filter!=gate) continue;
            for(int rows:{1,49,64}) for(bool bias:{false,true}) for(bool norm:{false,true}) {
                const auto oracle=run(cpu,rows,bias,norm,gate,"separate",profile,trailing_scale);
                for(const auto & alias:{"separate","logits_weights","logits_ids","logits_both_disjoint"}) {
                    const std::string key=gate+"/"+std::to_string(rows)+"/bias"+std::to_string(bias)+
                        "/norm"+std::to_string(norm)+"/"+alias+"/"+profile;
                    const auto actual=run(target,rows,bias,norm,gate,alias,profile,trailing_scale);
                    const auto cpu_comparison=compare(actual,oracle);
                    json entry={{"key",key},{"backend",backend},{"fusion",fusion},{"experts",EXPERTS},
                                {"selected",SELECTED},{"rows",rows},{"bias",bias},{"norm",norm},{"gate",gate},
                                {"profile",profile},{"trailing_scale",trailing_scale},
                                {"layout",actual.layout},{"cpu",cpu_comparison},
                                {"weights",actual.weights},{"ids",actual.ids}};
                    if(!reference_path.empty()) {
                        const auto found=references.find(key);
                        if(found==references.end()) throw std::runtime_error("reference missing "+key);
                        entry["reference"]=compare(actual,found->second);
                        reference_nonidentical+=!entry["reference"]["bitwise_equal"].get<bool>();
                    }
                    failed+=!cpu_comparison["within_tolerance"].get<bool>();
                    ++count;
                    std::cout<<entry.dump()<<std::endl;
                }
            }
        }
        ggml_backend_free(target); target=nullptr;
        ggml_backend_free(cpu); cpu=nullptr;
        std::cout<<json({{"summary",true},{"cases",count},{"trailing_scale",trailing_scale},
                        {"cpu_tolerance_failures",failed},
                        {"reference_nonidentical",reference_nonidentical}}).dump()<<std::endl;
        return failed?1:0;
    } catch(const std::exception & e) {
        if(target) ggml_backend_free(target);
        if(cpu) ggml_backend_free(cpu);
        std::cout<<json({{"error",e.what()}}).dump()<<std::endl;
        return 2;
    }
}
