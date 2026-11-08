# Decode and prefill measurements, 2026-11-08

These records support the [repository README](../../../README.md). They were captured before publication; this documentation update did not rerun model benchmarks or alter the installed binaries. Original absolute paths are retained as provenance.

## Controlled phase comparison

Hardware: i9-13900K, 192 GiB DDR5-5200 dual-channel memory, two RTX 5060 Ti 16 GB GPUs. Model: Qwen3.8-Flash-Next UD-Q4_K_XL. The production configuration retained context capacity 261888, batch/microbatch 4096, F16 KV, tensor split 1,1, 46 CPU expert layers, primary-GPU prefill arenas, eight decode threads, and 24 batch threads.

One model load, four fresh contexts in off/on/on/off order, identical 8192-token prompts and 320 forced decode tokens. Both conditions enabled N=1 MMID. The first 64 tokens were excluded from steady throughput. Each measured token included synchronized logits retrieval. Logit comparison and buffered profile writes occurred outside measured intervals. No concurrent build, model benchmark, or GPU profiler ran.

| Run | Phase layout | Steady t/s | Prefill seconds | First decode seconds |
|---|---|---:|---:|---:|
| A1 | Off | 23.664699 | 18.529418 | 0.735317 |
| B1 | On | 27.499568 | 18.278009 | 2.692858 |
| B2 | On | 27.359316 | 18.073504 | 2.636831 |
| A2 | Off | 23.679623 | 18.097332 | 0.638965 |

The mean steady gain is 15.872%. The first decode step costs 1.978 additional seconds. Including all 320 decode steps, rates are 21.975 t/s for original pools and 21.915 t/s for phase switching. The estimated amortization point is about 342 subsequent tokens at the measured steady rates, not a general guarantee.

- [Aggregate timings](phase-results-summary.json)
- [Replay input and case configuration](phase-fast-prefill-abba/config.json)
- [Executable/argument manifest](phase-fast-prefill-abba/manifest.json), [successful exit](phase-fast-prefill-abba/exit.json)
- Individual results: [A1](phase-fast-prefill-abba/result-0-phase0.json), [B1](phase-fast-prefill-abba/result-1-phase1.json), [B2](phase-fast-prefill-abba/result-2-phase1.json), [A2](phase-fast-prefill-abba/result-3-phase0.json)
- [Operation and thread summaries](summary.json), especially `phase-fast-prefill-abba` and `stock-thread-repeat`

## Mechanism and costs

Phase switching changes expert residency from 27 slots per layer in one GPU's pools to 44 slots on the layer's owning GPU. It preserves prompt routing history while replacing temporary graph and expert pools. Model and KV/state buffers remain allocated.

| Per-token mean | Original pools | GPU-local pools |
|---|---:|---:|
| CPU / CUDA0 / CUDA1 scheduler splits | 93 / 93 / 22 | 93 / 51 / 43 |
| CPU MoE operation time | 13.140 ms | 10.765 ms |
| Logical CPU expert reads | 818.941 MB | 642.641 MB |
| CPU backend compute wall time | 14.257 ms | 11.597 ms |
| GPU input-preparation host API wall time | 4.111 ms | 2.034 ms |
| GPU submission host API wall time | 0.954 ms | 0.770 ms |
| CPU-side waits for GPU sources | 18.052 ms | 17.243 ms |

CPU/GPU work overlaps. Host CUDA calls can block, and source waits include GPU execution. Do not add these values to separate device timings or treat all wait time as avoidable transfer overhead. Logical weight demand is not memory-controller DRAM bandwidth.

## Re-entry and numerical limits

Every ABBA context also executed a 49-token prompt plus 64 decode tokens, then a 4096-token prompt plus another 64 decode tokens. All completed with finite logits. Short prompt replay retained the decode pools; large re-entry restored the original prefill arenas and then rebuilt decode pools. The large re-entry prompt averaged 10.569 seconds with phase switching versus 9.140 seconds without, including pool restoration.

Optimized steady tails were 26.41 and 26.75 t/s after short replay and 23.95 and 24.30 t/s after the large prompt. Context and routing had changed, so these are separate workloads.

Initial full-vocabulary relative L2 against the saved reference was 0.0706-0.0724 for controls and 0.0770-0.0790 for phase-enabled runs. Greedy argmax differed on 8-9 versus 12-14 of 320 forced steps. All 64 greedy choices after each re-entry matched the first control, although logits differed. Unchanged original contexts also showed numerical variation. No task-quality or perplexity equivalence is established.

Cache creation or decode graph reservation failure is intended to restore the original layout and disable phase switching for that context. Normal transitions were exercised. Allocation-failure rollback was reviewed but not fault-injected.

The [dedicated N=1 log](kernel-test.txt) passes 32/32 cases. Those cases are included in the [full masked-operation log](kernel-all-test.txt), which passes 176/176. These fixtures establish equivalence for the tested operation inputs, not model-wide output identity. The archived kernel logs are converted from UTF-16 to UTF-8 with terminal color escapes removed for readability.

## Real HTTP serving

Both configurations completed an initial and continued streamed chat, each with 512 generated tokens. The production template, reasoning guard, startup warmup, and prefill settings were retained. Requests used temperature zero, a fixed seed, and `ignore_eos` to enforce length. Natural outputs differed.

| Initial chat | Original pools | Phase layout |
|---|---:|---:|
| Processed prompt tokens | 9643 | 9643 |
| Prompt time | 29.224 s | 29.308 s |
| Full generation average | 21.58 t/s | 22.09 t/s |
| Last 128 streamed tokens | 23.01 t/s | 25.76 t/s |

The continued chat processed 12820 prompt tokens with 5547 prefix tokens cached. Its prompt took 32.766 versus 34.397 seconds, including prefill pool restoration. Full generation averaged 21.46 versus 22.03 t/s. The phase-enabled first-to-second-token gaps were about 2.78 seconds initially and 2.66 seconds on continuation.

Server checkpoints split the initial prefill into batches of 1443, 1445, 2659, and 4092 despite configured capacity 4096. They retained the original primary-GPU, two-arena streaming path. The validation recorded no streaming error, phase fallback, or CUDA error.

- [Phase-enabled validation](server-smoke-phase-512/validation.json)
- [Matched original-pool validation](server-smoke-baseline-512/validation.json)
- [Separate stock b10870 comparison](stock-vs-fast/RESULT.md), below the arena threshold

## Package and reproduction records

[Activation hashes](activation.json) identify the installed files. [Candidate provenance](candidate-manifest.json) records the base commit, measured source-patch hash, compiler-era binary locations, original arguments, and selected environment. The full source patch from base `9eda97a9b647688e3328fc89c845f1c1d3662621` had SHA-256 `ebbc95edd96aec1ae3048a6b74155ea1b048b64267cd0c60ef70ccf9846f7df7` before documentation and launch-snapshot publication.

The package rebuilt `ggml-base`, `ggml-cpu`, `llama`, and `llama-common` with MSVC 19.37.32822.0. It retained the original CUDA DLL, server executable/implementation, `ggml`, and `mtmd`. The original CUDA build used toolkit 13.2 and architecture 120a. The source also contains a disabled pinned-refill experiment that is not in the installed CUDA DLL.

[The active launcher](../../../examples/qwen38-codex/launch-local.ps1) and [rollback snapshot](../../../examples/qwen38-codex/launch-pre-phase.ps1) contain the exact local asset paths. The two new enabled environment controls are `GGML_CPU_MOE_N1=1` and `LLAMA_MOE_CACHE_PHASE_LAYOUT=1`. Other scheduler/refill experiments remain disabled. Buffered diagnostic globals assume isolated single-context use.

Harness source is archived in [harness](harness/): [replay.cpp](harness/replay.cpp), [run.py](harness/run.py), [analysis](harness/analyze.py), [HTTP smoke runner](harness/server-smoke.py), [CMake project](harness/CMakeLists.txt), and [build script](harness/build.cmd). These preserve the original workspace layout and dependencies; they are not a standalone benchmark installer. In the existing local workspace, the controlled run was:

```powershell
Set-Location 'G:\qwenllamaMTP\perf-20260911\bench\decode-optimization-20261108'
.\build.cmd
python run.py --variant n1 --threads 8 --label NEW_UNIQUE_DIRECTORY --phase-matrix --profile --prepare --reentry
```

That runner expects the sibling `source` worktree, original installed binaries, model assets, and saved prompts/forced tokens under `../decode-diagnosis-20261108`. The exact input token arrays and re-entry configuration for the recorded run are retained in `phase-fast-prefill-abba/config.json`. The earlier full-vocabulary reference binary, raw multi-megabyte profiler CSVs, model weights, and executable binaries remain local and are not included in Git. Raw local results live under `G:\qwenllamaMTP\perf-20260911\bench\decode-optimization-20261108`.

[File hashes](copied-files.sha256.json) identify the included snapshots. The stock report also documents the discarded repetition-heavy pilot; its 34.46 t/s figure is excluded from the final result.
