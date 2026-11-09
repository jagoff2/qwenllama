# Qwen Flash-Next launch configuration

`launch-local.ps1` is the active fast-prefill/GPU-local-decode configuration, copied byte for byte from:

```text
G:\qwenllamaMTP\perf-20260911\bench\flashnext-stream-prefill\launch-best-decode-no-draft-think-guard.ps1
```

Run the installed configuration with:

```powershell
& 'G:\qwenllamaMTP\perf-20260911\bench\flashnext-stream-prefill\launch-best-decode-no-draft-think-guard.ps1'
```

The executable is `G:\qwenllamaMTP\perf-20260911\dist\fn-fast-prefill-local-decode-think-hard-20261109\bin\llama-server.exe`. The script enables the N=1 masked MMID path and phase-specific expert pools, with eight decode threads and 24 batch threads. It preserves the original 261888-token context capacity, F16 KV, 4096 batch/microbatch, primary-GPU arena prefill, thinking guard, and no-draft configuration.

The November 9 hard-guard update replaces only `llama-common.dll`. It blocks EOG and end markers during reasoning and after closure until answer or tool content starts, without counting repeated tags or Unicode whitespace as content. Final sampled tokens are checked against the guard. Synthetic HTTP and vocabulary-only sampler regressions passed. The exact chat template remains `G:\qwenllamaMTP\qwen38-codex-system-role-demoted.jinja`.

`launch-pre-phase.ps1` preserves the previous launcher and selects `dist/fn-decode-cache-think-guard-20261006/bin`. It is also available locally as `launch-best-decode-no-draft-think-guard.pre-phase-20261108.ps1`. Stop an existing server before starting either configuration on the same port. The rollback launcher restores the complete previous settings, including its earlier thread defaults; it is not the matched eight-thread benchmark control.

These are machine-specific snapshots, not portable installers. To use another checkout or machine, edit the executable, model shards, projector, and chat-template paths. The scripts clear process environment variables beginning with `LLAMA_`, `GGML_`, `QWEN4EXP_`, `OMP_`, and `DECODE_` before setting explicit controls. The active launcher binds to localhost.

`qwen38-codex-system-role-demoted.jinja` retains the template referenced by the launcher, including its local host instructions. `qwen38-codex.jinja` retains the earlier template for reference. Model and projector weights are not included.

The selected build reached 27.43 t/s in controlled steady replay, with the original fast prefill, but adds about 1.98 seconds at the generation transition. Short requests may not improve. [Results, provenance, and validation](../../docs/performance/2026-11-08/README.md).

Stock upstream b10870 remains installed separately at `C:\Users\user\qwenllama`. The [brief stock comparison](../../docs/performance/2026-11-08/stock-vs-fast/RESULT.md) launched its executable directly; neither launch script here is a stock upstream configuration.
