#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>

static bool reject_allocation = false;
static ggml_backend_buffer_type_t original_type;

static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t size) {
    if (reject_allocation) {
        return nullptr;
    }
    auto * buffer = original_type->iface.alloc_buffer(original_type, size);
    if (buffer) {
        buffer->buft = type;
    }
    return buffer;
}

static ggml_cgraph * graph(ggml_context * context, int rows) {
    auto * result = ggml_new_graph(context);
    auto * input = ggml_new_tensor_1d(context, GGML_TYPE_F32, rows);
    ggml_set_input(input);
    auto * output = ggml_scale(context, input, 2.0f);
    ggml_set_output(output);
    ggml_build_forward_expand(result, output);
    return result;
}

int main() {
    original_type = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type type = *original_type;
    type.iface.alloc_buffer = allocate;
    ggml_init_params params = {1 << 20, nullptr, true};
    auto * small_context = ggml_init(params);
    auto * large_context = ggml_init(params);
    auto * small = graph(small_context, 32);
    auto * large = graph(large_context, 4096);
    auto * allocator = ggml_gallocr_new(&type);
    if (!ggml_gallocr_reserve(allocator, small)) return 1;
    reject_allocation = true;
    if (ggml_gallocr_reserve(allocator, large)) return 2;
    if (ggml_gallocr_alloc_graph(allocator, large)) return 3;
    reject_allocation = false;
    if (!ggml_gallocr_alloc_graph(allocator, large)) return 4;
    if (!ggml_graph_node(large, ggml_graph_n_nodes(large) - 1)->data) return 5;
    ggml_gallocr_free(allocator);
    ggml_free(large_context);
    ggml_free(small_context);
    std::puts("allocation failure, identical-shape retry, and recovery: PASS");
}
