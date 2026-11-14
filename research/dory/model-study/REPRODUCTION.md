# Reproduce the model experiment

The source-only branch preserves the primary tasks, model outputs, configuration, analysis, and evaluation harness. No GGUF weights or Windows runtime binaries are committed. Numerical reanalysis needs Python 3.10+ and NumPy. Inference also needs the matching model, a built harness, and its runtime libraries.

## Reanalyze the recorded primary comparison

From the repository root, choose an output file that does not already exist:

```powershell
python research/dory/model-study/score_reproduction.py `
  --tasks research/dory/model-study/confirmation27_heldout192.jsonl `
  --stock research/dory/model-study/confirmation27_stock.jsonl `
  --candidate research/dory/model-study/confirmation27_norm28_31x4.jsonl `
  --output primary-rescore.json
```

Expected counts are 31 stock, 71 recurrence, 49 corrections, and 9 regressions. Expected exact paired p is 8.959402428981855e-08; the paired bootstrap interval is [14.0625, 28.125] percentage points with NumPy 2.2.6 and the recorded seed. This is a fresh numerical reanalysis of recorded results, not a new model inference run or independent verification of historical execution.

`original-artifact-sha256.json` identifies the original copied artifact bytes. `confirmation27_score.py` is retained as historical methodology: its frozen-provenance mode requires archived runtime files, including binaries excluded from this branch. Use the command above for a source-only numerical rescore.

## Original efficacy runtime

The primary result used source base `6b46b979bea3d7b08e21a9f540d61503c7a84ac9` from `https://github.com/noonr48/llama.cpp`. The current branch's main source tree is a later integration. To reconstruct the original model implementation, prepare a separate checkout of the original base and apply these patches in order:

```powershell
git apply C:/path/to/this-branch/research/dory/model-study/baseline_prerequisites.patch
git apply C:/path/to/this-branch/research/dory/model-study/recurrence_release27.patch
```

The first patch preserves pre-existing prerequisites shared by both original conditions. The second contains the recurrence and scheduler changes. Application to clean base versions of the affected files was checked; the experiment used an incremental build, not a verified complete clean build. [Original build record](original-build.md) describes the exact tools, arguments, and an explicitly unexecuted clean-build recipe. Its `releaseRoot` should point to this directory. Frozen source-only snapshots are under [original-source](original-source).

The historical primary runtime disabled operation fusion, graph reuse, and CUDA graphs, used CPU experts and 12 threads, and used these context arguments:

```text
-c 1024 -b 512 -ub 512 -ngl 99 --cpu-moe --fit off -t 12 -tb 12 -fa on --no-op-offload -lv 1
```

The reproducer sets those experimental switches and inherits PATH without changing it. Model hashes are in [model_hashes.json](model_hashes.json), and historical binary identities are in [confirmation27_runtime_manifest.json](confirmation27_runtime_manifest.json). Supply matching dependencies through installed tools or absolute paths; the scripts do not install tools or edit PATH.

## Build the evaluation harness against the integrated branch

For a configured Windows MSVC build of this branch, the helper compiles the evaluation harness and standalone regressions. Set the paths to your actual build and runtime:

```powershell
& ./research/dory/build-regressions.ps1 `
  -Build C:/path/to/configured-build `
  -Runtime C:/path/to/runtime-bin `
  -CommonLibrary C:/path/to/configured-build/common/llama-common.lib
```

MSVC, Windows SDK, and output paths are parameters. The default tool versions match the publication machine. The helper preserves PATH and uses absolute compiler paths. It produces `research/dory/out/latent-eval-schedule.exe` alongside the supplied DLLs. Running the original benchmark on this later implementation is a new efficacy experiment; do not label its outputs as the original frozen run.

## Run new paired inference

The runner verifies all four GGUF shards, creates a new output directory, performs first- and last-task neutralization controls, and runs both conditions with fresh contexts. Use the small smoke task file first, then the full task set:

```powershell
python research/dory/model-study/reproduce_recurrence.py `
  --binary research/dory/out/latent-eval-schedule.exe `
  --model C:/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf `
  --tasks research/dory/model-study/confirmation27_heldout192.jsonl `
  --output new-comparison

python research/dory/model-study/score_reproduction.py `
  --tasks research/dory/model-study/confirmation27_heldout192.jsonl `
  --stock new-comparison/stock.jsonl `
  --candidate new-comparison/norm28_31x4.jsonl `
  --output new-comparison/score.json
```

For a smoke run, replace `--tasks` with `research/dory/model-study/reproduction_smoke_tasks.jsonl` and choose a separate output directory. The independent scorer requires complete normally terminated output for every supplied task. Run GPU inference serially on the test machine.

## Try the integrated server

The public launcher accepts explicit executable/model paths and optional external template and multimodal-projector paths:

```powershell
& ./research/dory/launch.ps1 `
  -Server C:/path/to/runtime-bin/llama-server.exe `
  -Model C:/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf `
  -Mode Loop
```

`-Mode Stock` disables recurrence. `-Mode Neutral` executes recurrence and restores first-pass output/state for an identity control. Restart to change mode. Supply `-ChatTemplate` and `-MMProj` to use your existing external assets. The published launcher adapts the tested package launcher by parameterizing paths and making those assets optional. Its syntax and argument forwarding were checked during publication; a new full inference run was not performed for that adapter.
