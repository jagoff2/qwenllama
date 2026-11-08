# Stock versus fast: brief chat benchmark

Stock llama.cpp remains installed at `C:\Users\user\qwenllama`: **b10870 / 1945e0920**, dated 2026-09-09, compiled with Clang 20.1.8. Seven checked engine/server/backend files match the retained release archive. The commit is present in local upstream/master history. [Provenance](stock-provenance.json).

The fast build is `perf-20260911/dist/fn-fast-prefill-local-decode-20261108/bin`, based on `9eda97a9b` plus the local optimization patch, compiled with MSVC 19.37.32822.0. Neither installation nor the active launcher was changed by this comparison.

One complete, neutral chat request per build: **944 prompt tokens, 192 generated tokens**, context 8192, 8 decode threads and 24 batch threads. Both used the same Qwen3.8-Flash-Next UD-Q4_K_XL model, batch/microbatch 4096, 46 CPU expert layers, tensor split 1,1, F16 KV, flash attention, no draft, and identical greedy sampling with normal EOS handling. Startup warmup and context checkpoints were disabled for this short screen. The fast build enabled its production cache/arena controls.

| Measurement | Stock b10870 | Fast branch |
|---|---:|---:|
| Prompt processing | 6.177 s | 5.634 s |
| Steady decode, token counters 64 through 192 | 21.31 t/s | 27.78 t/s |
| Generation average, all 192 tokens | 21.39 t/s | 20.73 t/s |
| First-to-second-token gap | 0.091 s | 2.243 s |
| Complete request, excluding model loading | 15.111 s | 14.865 s |
| Model startup | 17.219 s | 17.343 s |

The observed steady rate is **30.3% higher** in this screen. The phase transition consumes that advantage for such a short answer, leaving the generation average slightly lower. Overall request duration was only about 0.25 seconds shorter, including prompt processing.

This is a single-pass screen, not a confidence interval or fixed-token replay. Generated outputs differed. Both produced non-looping technical reasoning in the final chat test. Compiler/backend versions differ. The prompt is below the 1024-token arena-streaming threshold, so this does **not** measure large-prompt fast arena prefill. The phase switch and two-GPU, 44-slot decode cache were confirmed in the fast log.

All test servers were stopped. No concurrent model workload, build or profiler ran.

- [Final summary](summary-chat.json)
- [Identical request](request-chat.json)
- [Stock result](stock-chat/result.json), [launch manifest](stock-chat/manifest.json)
- [Fast result](fast-chat/result.json), [launch manifest](fast-chat/manifest.json)
- [Runner](quick.py)

The earlier native-completion probe truncated a saved prompt at an incomplete conversation boundary. It produced repetition: the verified fast attempt emitted token 248045 on 173 of 192 steps. Its 34.46 t/s figure is excluded because that workload strongly favors cache locality. Those artifacts remain in `stock`, `fast`, `fast-verified`, and `summary.json` for audit. Only the `*-chat` results above are the final comparison.

The older custom build at `G:\qwenllamaMTP\llama.cpp\build\bin` and the pre-phase rollback installation also remain. They were not labeled or tested as stock.
