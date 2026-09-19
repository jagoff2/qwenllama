# Qwen3.8 Flash-Next Codex launch snapshot

These files preserve the local launch configuration used on 2026-09-19:

- `launch-local.ps1` preserves the command from `G:\qwenllamaMTP\perf-20260911\bench\flashnext-stream-prefill\launch-best.ps1`.
- `qwen38-codex.jinja` is an unchanged copy of the template at `G:\qwenllamaMTP\qwen38-codex.jinja` referenced by that script.

The launch script retains the original absolute paths. To use another checkout or machine, update the executable, model shards, projector, and chat-template paths. The model and projector weights are external assets.

The launched executable is `G:\qwenllamaMTP\perf-20260911\dist\namespace-tool-20260919\bin\llama-server.exe`. SHA-256 comparison confirmed that it and all eight adjacent DLLs match `G:\qwenllamaMTP\perf-20260911-nsfix\build\bin`. That build uses base commit `271fe98d46e38d79aeb45a052e3254f8b1e758f1` plus the Responses namespace-tool and content-conversion changes committed with this snapshot. The eight changed source, test, and server documentation files matched the build checkout byte-for-byte. The later root README update is preserved separately in the history.

Build settings recorded in the local CMake cache: Release, MSVC 19.37.32822.0, CUDA 13.2, architecture `120a`, shared libraries, CUDA and Flash Attention enabled, Flash Attention quantizations `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`, NCCL disabled, and OpenSSL disabled.

Validation before publication: the existing CPU `test-chat.exe` passed, and all 10 tests in `tools/server/tests/unit/test_compat_oai_responses.py` passed against the deployed executable with the local `stories260K-f32.gguf` test model and GPU layers disabled. This verification did not rerun the full Qwen model workload.
