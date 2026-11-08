# Qwen3.8 Flash-Next Codex launch snapshot

These files preserve the active local launch configuration identified on 2026-11-08:

- `launch-local.ps1` is a byte-for-byte copy of `G:\qwenllamaMTP\perf-20260911\bench\flashnext-stream-prefill\launch-best-decode-no-draft-think-guard.ps1`.
- `qwen38-codex-system-role-demoted.jinja` is a byte-for-byte copy of the template at `G:\qwenllamaMTP\qwen38-codex-system-role-demoted.jinja` referenced by that script, including its local host instructions.
- `qwen38-codex.jinja` retains the earlier 2026-09-19 template for reference.

The launch script retains the original absolute paths. To use another checkout or machine, update the executable, model shards, projector, and chat-template paths. The model and projector weights are external assets.

The launched executable is `G:\qwenllamaMTP\perf-20260911\dist\fn-decode-cache-think-guard-20261006\bin\llama-server.exe`. SHA-256 comparison confirmed that both executables and all eight DLLs in that directory match `G:\qwenllamaMTP\perf-20260911\llama.cpp\build\bin`. The build's CMake cache points to this checkout. The 29 source files in the saved decode-cache implementation manifest match the current files; its documentation file was updated later. All five source and test postimages in the saved reasoning-guard patch also match the current files.

This snapshot includes the shared expert decode cache, Qwen FIM-token end-of-generation correction, and reasoning guard that suppresses end-of-generation until visible answer or tool content follows reasoning. The launcher enables `--fn-decode-expert-cache`, disables speculative decoding, uses `--reasoning-format none`, and retains the 261888-token context, F16 KV cache, 4096-token batch and microbatch, and 4096 MiB streamed-expert budget.

Build settings recorded in the local CMake cache: Release, MSVC 19.37.32822.0, CUDA 13.2, architecture `120a`, shared libraries, CUDA and Flash Attention enabled, Flash Attention quantizations `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`, NCCL disabled, and OpenSSL disabled.

Validation on 2026-11-08 used the existing local test binaries: `test-moe-stream-plan.exe` passed 9568 checks, `test-moe-stream-sim.exe` passed 1108 checks, and `test-reasoning-budget.exe` passed its 12 sampler tests, UTF-8 checks, and output-before-EOG guard checks with `models/ggml-vocab-qwen35.gguf`. The launcher passed PowerShell syntax parsing. This publication did not rebuild the server or rerun the full Qwen model workload; the numerical-parity limitations recorded in [the decode-cache notes](../../docs/shared-expert-decode-cache.md) remain applicable.
