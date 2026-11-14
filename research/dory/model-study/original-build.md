# Runtime and source reproduction

The tested inference route uses the included `confirmation27_bin/latent-eval-schedule.exe` and its adjacent DLLs. Python 3.10 or newer runs `reproduce_recurrence.py`; NumPy is needed only for scoring. The Windows runtime was tested with an i9-13900K, two RTX 5060 Ti cards, driver 596.36, and CUDA 13.2. CUDA dependencies and the MSVC runtime must already be available to Windows. The package does not install dependencies or change PATH. It does not contain GGUF weights.

The first reproduction step verifies all four model shards against `model_hashes.json`. Supply the first shard through `--model`. The runner creates a new directory and fails if it already exists. Do not run two inference processes at once on the test machine; they compete for CPU and GPU resources. Do not rebuild DLLs while a harness has them loaded.

## Source preparation

The base is public commit `6b46b979bea3d7b08e21a9f540d61503c7a84ac9` in `https://github.com/noonr48/llama.cpp`. Prepare a separate checkout of that commit. Apply these patches in order from the checkout root, using the actual absolute archive path:

```powershell
git clone https://github.com/noonr48/llama.cpp qwen-recurrence-source
Set-Location qwen-recurrence-source
git checkout --detach 6b46b979bea3d7b08e21a9f540d61503c7a84ac9
git apply --check C:/path/to/release27/baseline_prerequisites.patch
git apply C:/path/to/release27/baseline_prerequisites.patch
git apply --check C:/path/to/release27/recurrence_release27.patch
git apply C:/path/to/release27/recurrence_release27.patch
```

The prerequisite patch preserves the two pre-existing build fixes used by both stock and recurrent runtimes. The recurrence patch changes the model graph and backend scheduler. `recurrence_release27.patch` is an LF-normalized copy of the original `recurrence_cache18.patch`; the original CRLF patch fails against an LF checkout. Both patches were successfully applied in order to clean base versions of all five affected files, and the resulting source matched the tested implementation after line-ending normalization. `patch_application27_check.json` records that verification. The frozen copies of `qwen4exp.cpp`, `models.h`, and `ggml-backend.cpp` are in `confirmation27_bin`; their hashes are in `confirmation27_runtime_manifest.json`.

## Build configuration

The tested runtime was incrementally built from an existing CMake/Ninja build directory. A completely clean rebuild was not performed during this experiment. `build_model.ps1` and `build_schedule.ps1` record the successful commands, but contain this machine's absolute paths. They require a configured repository build and its common-library import libraries. Merely running either script in an empty checkout is insufficient.

The relevant installed tools were MSVC 14.37.32822, Windows SDK 10.0.22621.0, CUDA 13.2, CMake, and Ninja. CMake's compiler paths were:

```text
CMAKE_C_COMPILER / CMAKE_CXX_COMPILER:
C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe
CMAKE_CUDA_COMPILER:
C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe
CMAKE_MAKE_PROGRAM:
C:/Users/user/AppData/Local/Programs/Python/Python3119/Scripts/ninja.exe
```

Configuration uses Ninja, Release, shared libraries, CUDA enabled, CUDA architecture 120, native CPU compilation, CPU repacking enabled, OpenMP enabled, and BLAS disabled. Explicit GGML_AVX, GGML_AVX2, GGML_AVX512, GGML_AVX512_BF16, GGML_AVX512_VBMI, GGML_AVX512_VNNI, and GGML_AVX_VNNI cache switches were OFF. These switches alone do not establish the final CPU instruction set when native compilation is enabled. CUDA graphs were compiled in, then disabled at inference through the documented environment variable.

A configure/build recipe matching those settings follows. It is provided for source reproduction; the complete clean-build recipe has not been executed. It assumes the listed dependencies already exist. Adjust the path literals to installed tools rather than changing PATH. Run from the patched source checkout:

```powershell
$releaseRoot = 'C:/path/to/release27'
$toolRoot = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.37.32822'
$sdkRoot = 'C:/Program Files (x86)/Windows Kits/10'
$cudaCompiler = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe'
$ninjaExe = 'C:/Users/user/AppData/Local/Programs/Python/Python3119/Scripts/ninja.exe'
$cmakeExe = (Get-Command cmake -CommandType Application -ErrorAction Stop).Source
$compilerExe = "$toolRoot/bin/Hostx64/x64/cl.exe"
$buildOldInclude = $env:INCLUDE
$buildOldLib = $env:LIB
try {
    $env:INCLUDE = "$toolRoot/include;$sdkRoot/Include/10.0.22621.0/ucrt;$sdkRoot/Include/10.0.22621.0/shared;$sdkRoot/Include/10.0.22621.0/um"
    $env:LIB = "$toolRoot/lib/x64;$sdkRoot/Lib/10.0.22621.0/ucrt/x64;$sdkRoot/Lib/10.0.22621.0/um/x64"
    $configureArgs = @(
        '-S', '.', '-B', 'build', '-G', 'Ninja',
        "-DCMAKE_MAKE_PROGRAM=$ninjaExe",
        "-DCMAKE_C_COMPILER=$compilerExe", "-DCMAKE_CXX_COMPILER=$compilerExe",
        "-DCMAKE_CUDA_COMPILER=$cudaCompiler", "-DCMAKE_CUDA_HOST_COMPILER=$compilerExe",
        "-DCMAKE_LINKER=$toolRoot/bin/Hostx64/x64/link.exe",
        "-DCMAKE_RC_COMPILER=$sdkRoot/bin/10.0.22621.0/x64/rc.exe",
        "-DCMAKE_MT=$sdkRoot/bin/10.0.22621.0/x64/mt.exe",
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
        '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=120',
        '-DGGML_NATIVE=ON', '-DGGML_CPU_REPACK=ON', '-DGGML_OPENMP=ON', '-DGGML_BLAS=OFF',
        '-DGGML_AVX=OFF', '-DGGML_AVX2=OFF', '-DGGML_AVX512=OFF',
        '-DGGML_AVX512_BF16=OFF', '-DGGML_AVX512_VBMI=OFF',
        '-DGGML_AVX512_VNNI=OFF', '-DGGML_AVX_VNNI=OFF'
    )
    & $cmakeExe @configureArgs
    if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
    & $cmakeExe --build build --target llama llama-common --parallel 2
    if ($LASTEXITCODE -ne 0) { throw 'Library build failed' }
    & $compilerExe /nologo /std:c++17 /EHsc /O2 /MD /I common /I include /I ggml/include /I vendor /Fe:build/bin/latent-eval-schedule.exe /Fo:build/latent_eval_schedule.obj "$releaseRoot/latent_eval_schedule.cpp" /link build/common/llama-common.lib build/src/llama.lib build/ggml/src/ggml.lib build/ggml/src/ggml-base.lib
    if ($LASTEXITCODE -ne 0) { throw 'Harness build failed' }
} finally {
    $env:INCLUDE = $buildOldInclude
    $env:LIB = $buildOldLib
}
```

Building `llama` also builds its backend dependencies, including the modified scheduler in `ggml-base`. Keep the resulting executable adjacent to its corresponding DLLs. The original recorded harness compile/link invocation is in `build_schedule.ps1`. A fresh build can differ in compiler-generated code and library versions, so do not replace frozen runtime files when checking the historical result.

## Verification

Use the archive's primary scorer to validate the recorded result without inference:

```powershell
python confirmation27_score.py --baseline confirmation27_stock.jsonl --candidate confirmation27_norm28_31x4.jsonl --manifest confirmation27_manifest.json --check-only
```

For a newly built runtime, run `reproduce_recurrence.py` with `reproduction_smoke_tasks.jsonl` first, then use the complete held-out task file. The runner's first and last case neutralization checks must pass. Bitwise output identity across different hardware, toolchains, or kernels is not assumed; record the new manifest and rescore the entire comparison. `score_reproduction.py` evaluates new runs without asserting historical frozen provenance.
