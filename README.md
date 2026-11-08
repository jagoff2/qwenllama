# qwenllama: fast Flash-Next prefill and GPU-local decode

This llama.cpp fork runs Qwen Flash-Next (`qwen4exp`) models with CPU-resident MoE weights. Large prompts stream expert weights through temporary GPU arenas. During generation, a phase switch places expert caches on each layer's owning GPU, reducing CPU expert reads and GPU handoffs while retaining the original fast prefill configuration.

On the tested i9-13900K and two RTX 5060 Ti 16 GB GPUs, controlled steady decode improved from **23.67 to 27.43 tokens/s** with eight CPU threads. The same 8192-token prefill remained about **18.2 seconds**. The switch adds **1.98 seconds** before steady generation, so short answers can see little benefit. **40 tokens/s has not been achieved.** [Recorded measurements](docs/performance/2026-11-08/README.md).

## Launch the installed build

The active Windows launcher is:

```powershell
& 'G:\qwenllamaMTP\perf-20260911\bench\flashnext-stream-prefill\launch-best-decode-no-draft-think-guard.ps1'
```

An exact snapshot is tracked at [examples/qwen38-codex/launch-local.ps1](examples/qwen38-codex/launch-local.ps1). It starts `perf-20260911/dist/fn-fast-prefill-local-decode-20261108/bin/llama-server.exe` with the original chat template, reasoning guard, and no draft model. The server binds to `127.0.0.1`.

The snapshot contains absolute paths for this machine. On another installation, edit its executable, GGUF model, projector, and chat-template paths. Weights and compiled binaries are external assets. [Launch configuration and rollback](examples/qwen38-codex/README.md).

## What is enabled

| Setting | Active value |
|---|---|
| Model | Qwen3.8-Flash-Next UD-Q4_K_XL, four GGUF shards |
| CPU / memory | i9-13900K; 192 GiB DDR5-5200, dual channel |
| GPUs | 2 x RTX 5060 Ti, 16 GB each |
| Context capacity / parallel slots | 261888 / 1 |
| Batch / microbatch | 4096 / 4096 |
| KV cache | F16 keys and values |
| Layer placement | Layer split, tensor split 1,1; 46 CPU expert layers |
| Prefill arenas | Primary GPU, 2 slots, 4096 MiB configured budget, 1024-token streaming threshold |
| CPU threads | 8 for decode, 24 for batch execution |
| Decode cache | 44 resident experts per CPU expert layer, on its owning GPU |
| New controls | `GGML_CPU_MOE_N1=1`, `LLAMA_MOE_CACHE_PHASE_LAYOUT=1` |

The N=1, Top-10 masked MMID path compacts selected expert routes, skips experts already handled by the GPU cache, and distributes remaining rows through one CPU work queue. It reuses the existing quantization and dot-product routines. It does not replace large-batch prefill. Its isolated gain at eight threads was small; larger local caches and reduced handoff work produced the substantial gain.

At an explicit prompt-to-generation boundary, the context synchronizes execution, releases temporary graph and expert pools, and recreates decode pools on both GPUs. Model weights, KV/state buffers, and prompt routing history survive. A later prompt batch above 49 tokens restores the original prefill allocation; shorter replay keeps the decode cache. This path is opt-in and restricted to QWEN4EXP without LoRA and with explicit prompt/generation purpose.

## Measured performance

The main comparison used one model load, four fresh contexts in off/on/on/off order, identical 8192-token prompts, and 320 forced decode tokens. Both sides used eight threads and the N=1 kernel. The steady metric excludes the first 64 tokens and includes synchronized logits retrieval.

| Metric | Original pools | GPU-local decode pools |
|---|---:|---:|
| Steady decode, mean of two runs | 23.672 t/s | 27.429 t/s |
| Initial prefill | 18.313 s | 18.176 s |
| First decode step | 0.687 s | 2.665 s |
| All 320 decode steps | 21.975 t/s | 21.915 t/s |
| Scheduler splits per token | 208 | 187 |
| CPU MoE operation time per token | 13.140 ms | 10.765 ms |
| Logical CPU expert bytes per token | 818.941 MB | 642.641 MB |
| GPU input-preparation host API time | 4.111 ms | 2.034 ms |

Steady decode improved **15.87%**. Initial prefill differences are small and are not claimed as a prefill improvement. At these steady rates, about 342 subsequent tokens amortize the additional transition cost. Actual results depend on context, routing, and output length. Reducing splits to 137 in an earlier experiment made decode slower by losing overlap, so split count alone is not the target.

Three controlled repeats per thread count found means of **21.307 t/s at 24 threads**, **23.582 t/s at 8**, and **21.020 t/s at 16**. This isolates the thread choice from the later cache-layout change. [Operation and thread summaries](docs/performance/2026-11-08/summary.json).

Real 512-token HTTP chat tests retained the production prefill and template. Initial prompt processing took 29.224 seconds with original pools and 29.308 seconds with phase switching. Full generation averaged 21.58 versus 22.09 t/s; the last 128 streamed tokens reached 23.01 versus 25.76 t/s. In serving, the phase transition appears between the first and second output tokens. Natural outputs differed. [HTTP validation](docs/performance/2026-11-08/server-smoke-phase-512/validation.json).

## Brief comparison with stock llama.cpp

The retained stock installation is b10870 / `1945e0920` at `C:\Users\user\qwenllama`, verified against the saved release archive. One identical chat request per build used 944 prompt tokens, 192 output tokens, eight decode threads, and an 8192-token context.

| Metric | Stock b10870 | This fast build |
|---|---:|---:|
| Steady decode, token counters 64-192 | 21.31 t/s | 27.78 t/s |
| Generation average, all output tokens | 21.39 t/s | 20.73 t/s |
| First-to-second-token gap | 0.091 s | 2.243 s |
| Whole request, excluding model loading | 15.111 s | 14.865 s |

The steady improvement was 30.3% in this single screen, but the transition consumed most of the benefit on the short answer. Outputs and compiler/backend versions differed. This prompt was below the arena-streaming threshold, so this comparison does not measure large-prompt prefill. [Full stock comparison and provenance](docs/performance/2026-11-08/stock-vs-fast/RESULT.md).

## Build and validation

See the [build guide](docs/build.md) for prerequisites. A CUDA build from a configured x64 Visual Studio development shell can use:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_OPENSSL=OFF
cmake --build build --target llama-server test-backend-ops --parallel 8
```

Update the tracked launch script to select the resulting executable and your asset paths. `GGML_NATIVE=ON` targets the build machine's CPU. Use the build guide when targeting other hardware.

The measured installed package uses rebuilt `ggml-base`, `ggml-cpu`, `llama`, and `llama-common` libraries with the original CUDA DLL and server executable. A fresh full source build is not the byte-identical measured package. [Binary hashes and activation record](docs/performance/2026-11-08/activation.json), [build provenance](docs/performance/2026-11-08/candidate-manifest.json).

Validation includes 32 dedicated N=1 fixtures within 176 total masked-operation cases, fixed-token ABBA replay, 49-token and 4096-token prompt re-entry, and initial/continued 512-token streamed HTTP chats. All tested re-entry sequences completed with finite logits. [Detailed evidence and harness snapshots](docs/performance/2026-11-08/README.md).

End-to-end logits are not bitwise identical, including between unchanged control contexts. No model-wide quality or perplexity equivalence is claimed. Allocation-failure rollback was reviewed but not fault-injected. Validation is specific to the listed hardware and single-slot configuration; allocating a 261888-token context is not a full-length throughput test.

Optional compact graphs, batched D2H transfers, serial cache preparation, and pinned refill experiments remain disabled in the active launcher. The source includes the optional CUDA refill experiment, but the installed package retains the original CUDA implementation. Buffered CPU and handoff diagnostics are intended for isolated single-context measurements.

Earlier memory-controller observations were about 25 GB/s over whole decode versus 68.43 GB/s in a synthetic read test. Whole-decode reads do not isolate CPU MoE bandwidth. Logical expert bytes are also not physical DRAM reads. GPU-source waits include execution time and cannot all be counted as removable transfer latency.

## Further documentation

- [Server API and options](tools/server/README.md)
- [Shared expert decode-cache design](docs/shared-expert-decode-cache.md)
- [Inherited Flash-Next/MTP work](FLASHNEXT_MTP.md); speculative decoding is disabled in this launcher
- [License](LICENSE)

Based on llama.cpp and the Flash-Next fork. This repository records a hardware-specific optimization and its measured limitations.
