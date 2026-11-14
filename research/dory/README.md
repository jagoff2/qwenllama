# Recurrence research artifacts

The branch-root [whitepaper](../../README.md) describes the model experiment and benchmark results.

| Directory or file | Contents |
|---|---|
| [model-study](model-study) | Frozen multiplication benchmark, per-problem outputs, scorers, model hashes, original source patch, and evaluation harness |
| [model-study/REPRODUCTION.md](model-study/REPRODUCTION.md) | Numerical reanalysis, original-runtime reconstruction, and new inference procedures |
| [config](config) | Stock, Loop, Neutral, and wide-neutral configurations |
| [launch.ps1](launch.ps1) | Parameterized launcher for a built integrated server |
| [build-regressions.ps1](build-regressions.ps1) | Windows build helper for standalone checks and the model evaluation harness |
| [regressions](regressions) | Standalone numerical and allocation regression sources |
| [evidence](evidence) | Curated integration measurements and provenance |
| [reference-scripts](reference-scripts) | Original laboratory build/test scripts, retaining their original absolute paths |

`reference-scripts` is an archival source exhibit, not a portable entry point. Its scripts refer to the original directory layout and separately built runtimes. Use the parameterized launcher/build helper and the standalone model reproducer for new work. The runtime evidence retains numerical results and hashes while excluding raw diagnostics, chat history, and private templates.

Run `python research/dory/verify-evidence.py` from the repository root to check the published integration summaries, token/fingerprint comparisons, source identities, and artifact manifest without a GPU. Use the separate model-study scorer for the primary 192-case result.

`evidence/artifact-sha256.json` hashes publication files relative to this directory. Text line endings are normalized to LF for this manifest; PNG bytes are hashed as-is. `model-study/original-artifact-sha256.json` preserves byte-for-byte hashes of copied original study files. Runtime binaries and GGUF weights are external.
