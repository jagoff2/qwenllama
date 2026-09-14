// Streamed MoE expert prefill engine - CUDA side.
//
// See ggml/include/ggml-moe-stream.h for the contract. This file declares the engine that the
// GGML_OP_MOE_STREAM_FENCE implementation drives, plus the small query API llama.cpp reaches through
// ggml_backend_reg_get_proc_address().
#ifndef GGML_CUDA_MOE_STREAM_CUH
#define GGML_CUDA_MOE_STREAM_CUH

#include "ggml-moe-stream.h"

#include <cuda_runtime.h>

struct ggml_backend_cuda_context;

// fence op implementation: orders the engine against the compute stream
void ggml_cuda_op_moe_stream_fence(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// engine lifecycle, called through the device registry
int  ggml_cuda_moe_stream_setup_impl(const struct ggml_cuda_moe_stream_params * params,
                                     const struct ggml_cuda_moe_stream_layer  * layers, int n_layers,
                                     const struct ggml_cuda_moe_stream_slot   * slots,  int n_slots);
void ggml_cuda_moe_stream_teardown_impl(int device);
void ggml_cuda_moe_stream_stats_impl(int device, struct ggml_cuda_moe_stream_stats * out);
void ggml_cuda_moe_stream_reset_impl(int device);
int  ggml_cuda_moe_stream_active_impl(int device);
void ggml_cuda_moe_stream_begin_impl(int device, const int32_t * slots, const int32_t * layers, int n);

// Called by the CUDA backend once per computed graph so the engine can drop per-graph state.
void ggml_cuda_moe_stream_end_of_graph(int device);

#endif // GGML_CUDA_MOE_STREAM_CUH
