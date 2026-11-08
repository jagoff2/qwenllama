# Shared expert decode cache

This opt-in Qwen3.8-Flash-Next path reuses the streamed-prefill expert buffers for generation. GPU-resident cached experts run on CUDA; uncached experts retain the CPU path. The original GGUF expert tensors remain authoritative. There is no additional expert-weight CUDA allocation and no allocator teardown at a phase change.

## Controls and supported configuration

Add `--fn-decode-expert-cache` to a server configuration with `--fn-stream-experts`. Disable it with `--no-fn-decode-expert-cache`; the default is disabled. The environment equivalent is `LLAMA_ARG_FN_DECODE_EXPERT_CACHE=1`.

The deployed suite configuration used for capacity planning is `-ncmoe 46 --fn-stream-slots 2 --fn-stream-gpu-mode primary --fn-stream-budget-mib 4096 --fn-stream-min-tokens 1024`, with one server sequence and 261888 configured context tokens. Its arena device is CUDA0. Dense weights, GPU-resident expert layers 46 and 47, KV placement, and the existing prefill configuration retain their original ownership.

The cache accepts separate, contiguous host expert tensors with 512 experts, top-10 routing, embedding width 2560 and expert width 640. Supported quantizations are Q4_K, Q5_K, Q5_1 and Q8_0. Creation probes the actual CPU buffer through masked-MMID support checks; repacked or unsupported buffers fall back. A metadata-only eight-row MMID probe also checks every expert family against the actual arena device's forced MMVQ capability. Devices that cannot execute these tiles fall back before cache execution. The implementation requires a single arena device. Multiple sequences and MTP contexts are excluded; active LoRA uses the ordinary expert path.

The server calls `llama_decode_with_purpose()` with `LLAMA_DECODE_PROMPT` or `LLAMA_DECODE_GENERATION`. Legacy `llama_decode()` retains automatic behavior and does not infer a generation phase solely from a small batch. The generation cache supports up to 49 rows, covering the default ngram-map-k verification length, using CUDA MoE tiles of at most eight rows. Larger generation batches use the ordinary path.

## Capacity

Capacity below is derived from this model's tensor metadata and the allocator layout, rather than a measurement of available free VRAM.

| Expert class | Host layers | Bytes per complete expert triplet |
|---|---:|---:|
| Q4_K up/gate, Q5_1 down | 43 | 3,072,000 |
| Q4_K up/gate, Q8_0 down | 2 | 3,584,000 |
| Q5_K up/gate, Q8_0 down | 1 | 3,993,600 |

Two existing allocations provide approximately 3932 MiB physically: each has a 1950 MiB expert payload, CUDA tail padding, and 16 MiB original slack. Decode keeps that original slack unused and reserves 32 MiB of each payload for scratch and mappings. One MiB of the first scratch span is dedicated to CUDA activation quantization.

Each of the 46 host layers receives 27 resident slots and one permanently zero dummy slot. The three family tensors form a contiguous layer bank. Deterministic size-descending balanced placement keeps every bank within one original allocation. The 1242 resident experts plus dummy slots consume 3825.413086 MiB including alignment and padding; 64 MiB is reserved for scratch, leaving approximately 10.587 MiB of unused payload. The planner rejects configurations that cannot fit these banks. It does not allocate a second cache on CUDA1.

## Phase and copy protocol

Before reusing an arena, the context synchronizes scheduler readers and the CUDA engine parks its worker and drains active transfers. A context-specific owner identifier guards the engine. Large prompt microbatches return the memory to the whole-layer prefill ring and invalidate decode mappings. Small prompt microbatches keep a warm cache and use the ordinary prompt path.

The completed prompt exports at most the last 256 routing rows per layer. The first generation phase ranks that bounded history and seeds up to 27 experts per layer. This seed finishes before publishing its mappings and can copy the full cache capacity. Subsequent admissions use aged frequency scores, a minimum score of two, a replacement margin of `max(1, resident_score / 8)`, and a global 64 MiB copy budget per generation microbatch. Scores halve every 32 evaluated generation rows. Candidates are ranked by score improvement per byte; equal scores do not cause replacement.

An expert becomes visible only after all three family copies complete. Eviction removes its prior mapping before another graph can read the destination slot. Each graph receives matching immutable CPU and GPU mapping snapshots: zero means CPU execution and the GPU dummy expert; a positive slot means cached GPU execution and exact zero CPU output. The graph input owner uploads the GPU map asynchronously on its backend stream. Host map storage remains unchanged until graph completion. Unfinished admissions remain unmapped while later graph work runs. Weight bank addresses and allocation ownership remain stable across graph reuse.

Router observations are contiguous graph outputs, expanded after router weights so the existing router fusion remains available. After graph completion, the context queues route reads for every layer, synchronizes the scheduler once for those reads, and then updates host statistics. The host read buffers stay alive until synchronization.

Runtime copy failures fail the request and disable subsequent use of that cache context. Unsupported layouts or an unsuccessful full-memory graph reservation use the baseline path. Reservation restores the current output count even when graph construction throws. Failure cleanup drains admissions and leaves cache mode before destroying cache descriptors, allowing the original pool to serve prefill again. Generation fallback routes are counted separately as `baseline_routes`, never as cache hits or misses, and do not trigger cache admissions. Prompt histogram collection remains independent of these counters.

## Hybrid execution and bounded scratch

Each cached layer first copies its activations and original expert IDs into distinct CPU tensors. The graph then enqueues the independent CUDA cache branch before visiting the CPU masked FFN branch. Actual graph edges keep the CPU preparation tensors alive until their readers finish. The two down-projection results are added before the usual expert weighting and reduction.

The scheduler can omit its usual CUDA-to-CPU split barrier only after a narrow post-allocation check: discrete CUDA followed by CPU, one scheduler copy, no evaluation callback, no incoming CPU split copies, distinct device/host storage for all computational inputs and outputs, and only the expected cache FFN operations. Original sources of GPU input copies must also reside on a device. If any condition fails, ordinary synchronization remains in force. The scheduler logs `fn-cache: CPU/GPU overlap active` when this guarded path is first used.

CUDA expert work uses MMVQ tiles of at most eight rows, including repeated references to the all-zero dummy expert. Intermediate cache tensors and tile concatenations use fixed addresses inside the reserved arena scratch. The layout is reused between layers after the preceding layer's dependencies complete. Activation quantization uses the dedicated one-MiB span, avoiding a separate backend-pool allocation for these cache MMVQs. Bounds checks reject layouts that exceed either reserved span.

Cache CUDA splits remain eligible for ordinary CUDA graph capture and replay. Their fixed weight, mapping and scratch addresses remain stable while mappings change in place between completed executions. Cache up/gate fusion is disabled to preserve the bounded MMVQ dispatch. Existing backend capture compatibility checks still apply; streamed-prefill fences continue to prevent capture of their own splits.

The fixed cache banks and scratch do not imply unchanged total VRAM usage. Full-memory reservation covers graph tensors, including attention and recurrent/indexer state at the configured context, but does not execute kernels or reserve every runtime CUDA temporary, cuBLAS workspace or graph executable. Existing backend pools can retain their peak allocation, and capture can add graph-executable storage within the remaining VRAM. Whole-GPU telemetry therefore measures sampled total usage, including other device users; it does not isolate cache allocations or guarantee the instantaneous peak was sampled.

The backend graph cache has no explicit entry-count or byte budget. It checks for entries unused for at least ten seconds on a sweep at most once every five seconds. Those checks still run during streamed prefill because graph lookup precedes capture-compatibility rejection. A first-node key reused by an incompatible prefill split is marked recently used, so its old executable can remain resident until the key becomes idle. Eviction bounds idle retention, not the total allocation of recently used graph shapes. Runtime headroom must therefore be validated separately from expert-bank geometry.

## Validation limits

The planner tests cover the real mixed-quant geometry, bank alignment and non-overlap, scratch bounds, insufficient capacity, arithmetic overflow, deterministic packing, and admission hysteresis. Full-memory graph reservation is attempted with the configured context capacity before cache execution.

These structural checks do not establish an end-to-end decode speedup. Historical route traces came from an older short MTP run and do not establish hit rates or throughput for the current long-generation ngram workload. Model-level numerical parity, phase-transition correctness, peak VRAM and prompt/generation performance require separate validation before changing the deployed launcher to enable the feature. This document does not report final benchmark results.

The 2026-10-05 smoke validation also repeated the original deployed executable with identical launch settings and requests. That baseline control itself failed the predefined `0.002` absolute log-probability tolerance: maximum differences were `0.951548`, `1.274541` and `2.551224` for the smoke, short follow-up and large-prompt cases. Generated tokens diverged after common prefixes of 56 and 59 tokens in the latter two cases. The cause of this run-to-run variation remains unisolated. The candidate's numerical parity gate remains **not passed**. The saved deployed build matches the main parent source and relevant compiler flags; see [baseline provenance](../../bench/shared-expert-cache/provenance.md) for the scope of those checks and the control artifacts.
