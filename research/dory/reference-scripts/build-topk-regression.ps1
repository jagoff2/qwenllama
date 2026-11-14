param(
    [string]$RuntimeSource = 'G:\qwenllamaMTP\experiments\dory_optimizations_20261113\runtime',
    [switch]$SkipRuntimeCopy
)
$ErrorActionPreference = 'Stop'
$beforePath = $env:PATH
$oldInclude = $env:INCLUDE
$oldLib = $env:LIB
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822'
$sdk = 'C:\Program Files (x86)\Windows Kits\10'
$ver = '10.0.26100.0'
$src = 'G:\qwenllamaMTP\experiments\dory_server_20261113\source'
$build = 'G:\qwenllamaMTP\experiments\dory_server_20261113\build'
$output = Join-Path $PSScriptRoot 'topk_runtime'
$pushed = $false
try {
    $env:INCLUDE = "$vc\include;$sdk\Include\$ver\ucrt;$sdk\Include\$ver\shared;$sdk\Include\$ver\um"
    $env:LIB = "$vc\lib\x64;$sdk\Lib\$ver\ucrt\x64;$sdk\Lib\$ver\um\x64"
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    if (-not $SkipRuntimeCopy) {
        Copy-Item -Path (Join-Path $RuntimeSource '*.dll') -Destination $output
    }
    Push-Location $output
    $pushed = $true
    & "$vc\bin\Hostx64\x64\cl.exe" /nologo /EHsc /std:c++17 /MD /O2 /DGGML_SHARED /DNDEBUG /I"$src\vendor" /I"$src\ggml\include" "$PSScriptRoot\topk_moe_regression.cpp" /Fe:topk-moe-regression.exe /link "$build\ggml\src\ggml.lib" "$build\ggml\src\ggml-base.lib"
    if ($LASTEXITCODE) { throw 'Top-k regression compilation failed' }
    if ($env:PATH -cne $beforePath) { throw 'PATH unexpectedly changed' }
} finally {
    if ($pushed) { Pop-Location }
    $env:INCLUDE = $oldInclude
    $env:LIB = $oldLib
}
