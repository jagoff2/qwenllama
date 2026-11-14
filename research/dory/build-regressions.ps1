param(
    [Parameter(Mandatory)][string]$Build,
    [Parameter(Mandatory)][string]$Runtime,
    [string]$Output = (Join-Path $PSScriptRoot 'out'),
    [string]$VC = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822',
    [string]$SDK = 'C:\Program Files (x86)\Windows Kits\10',
    [string]$SDKVersion = '10.0.26100.0',
    [string]$CommonLibrary
)
$ErrorActionPreference = 'Stop'
$originalPath = $env:PATH
$oldInclude = $env:INCLUDE
$oldLib = $env:LIB
$source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Build = (Resolve-Path -LiteralPath $Build).Path
$Runtime = (Resolve-Path -LiteralPath $Runtime).Path
$compiler = Join-Path $VC 'bin\Hostx64\x64\cl.exe'
$ggml = Join-Path $Build 'ggml\src\ggml.lib'
$base = Join-Path $Build 'ggml\src\ggml-base.lib'
foreach ($file in @($compiler, $ggml, $base)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$null = New-Item -ItemType Directory -Path $Output -Force
$Output = (Resolve-Path -LiteralPath $Output).Path
$pushed = $false
try {
    $env:INCLUDE = "$VC\include;$SDK\Include\$SDKVersion\ucrt;$SDK\Include\$SDKVersion\shared;$SDK\Include\$SDKVersion\um"
    $env:LIB = "$VC\lib\x64;$SDK\Lib\$SDKVersion\ucrt\x64;$SDK\Lib\$SDKVersion\um\x64"
    Copy-Item -Path (Join-Path $Runtime '*.dll') -Destination $Output
    Push-Location $Output
    $pushed = $true
    $flags = @('/nologo', '/O2', '/EHsc', '/std:c++17', '/MD', '/DGGML_SHARED', '/DGGML_BACKEND_SHARED', '/DNDEBUG',
        "/I$source\vendor", "/I$source\ggml\include", "/I$source\ggml\src", "/I$source\include", "/I$source\src")
    foreach ($test in @('gdn_cache_regression', 'topk_moe_regression', 'allocation_failure_regression')) {
        & $compiler @flags "$PSScriptRoot\regressions\$test.cpp" "/Fe:$test.exe" /link $ggml $base
        if ($LASTEXITCODE) { throw "Build failed: $test" }
    }
    & $compiler @flags "$source\tests\test-moe-stream-plan.cpp" "$source\src\llama-moe-stream-plan.cpp" /Fe:test-moe-stream-plan.exe /link $base
    if ($LASTEXITCODE) { throw 'Planner test build failed' }
    if ($CommonLibrary) {
        & $compiler @flags /DLLAMA_SHARED "/I$source\common" "$PSScriptRoot\regressions\probe.cpp" /Fe:probe.exe /link $CommonLibrary "$Build\src\llama.lib" $ggml $base
        if ($LASTEXITCODE) { throw 'Probe build failed' }
        & $compiler @flags /DLLAMA_SHARED "/I$source\common" "$PSScriptRoot\model-study\latent_eval_schedule.cpp" /Fe:latent-eval-schedule.exe /link $CommonLibrary "$Build\src\llama.lib" $ggml $base
        if ($LASTEXITCODE) { throw 'Evaluation harness build failed' }
    }
    & '.\allocation_failure_regression.exe'
    if ($LASTEXITCODE) { throw 'Allocation failure regression failed' }
    & '.\test-moe-stream-plan.exe'
    if ($LASTEXITCODE) { throw 'Planner test failed' }
} finally {
    if ($pushed) { Pop-Location }
    $env:INCLUDE = $oldInclude
    $env:LIB = $oldLib
    if ($env:PATH -cne $originalPath) { throw 'PATH changed unexpectedly' }
}
