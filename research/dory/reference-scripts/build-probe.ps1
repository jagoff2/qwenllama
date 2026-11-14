$ErrorActionPreference = 'Stop'
$beforePath = $env:PATH
$oldInclude = $env:INCLUDE
$oldLib = $env:LIB
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822'
$sdk = 'C:\Program Files (x86)\Windows Kits\10'
$ver = '10.0.26100.0'
$src = 'G:\qwenllamaMTP\experiments\dory_server_20261113\source'
$build = 'G:\qwenllamaMTP\experiments\dory_server_20261113\build'
$imports = 'G:\qwenllamaMTP\perf-20260911\llama.cpp\build'
try {
    $env:INCLUDE = "$vc\include;$sdk\Include\$ver\ucrt;$sdk\Include\$ver\shared;$sdk\Include\$ver\um"
    $env:LIB = "$vc\lib\x64;$sdk\Lib\$ver\ucrt\x64;$sdk\Lib\$ver\um\x64"
    New-Item -ItemType Directory -Path "$PSScriptRoot\runtime" -Force | Out-Null
    if (-not (Test-Path -LiteralPath "$PSScriptRoot\runtime\llama.dll")) {
        Copy-Item -Path 'G:\qwenllamaMTP\perf-20260911\dist\fn-dory-recurrence-20261113\bin\*.dll' -Destination "$PSScriptRoot\runtime"
    }
    Push-Location "$PSScriptRoot\runtime"
    & "$vc\bin\Hostx64\x64\cl.exe" /nologo /EHsc /std:c++17 /MD /O2 /DLLAMA_SHARED /DGGML_SHARED /DNDEBUG /I"$src\common" /I"$src\vendor" /I"$src\ggml\include" /I"$src\include" "$PSScriptRoot\probe.cpp" /Fe:probe.exe /link "$imports\common\llama-common.lib" "$build\src\llama.lib" "$build\ggml\src\ggml.lib" "$build\ggml\src\ggml-base.lib"
    if ($LASTEXITCODE) { throw 'Probe compilation failed' }
    if ($env:PATH -cne $beforePath) { throw 'PATH unexpectedly changed' }
} finally {
    Pop-Location
    $env:INCLUDE = $oldInclude
    $env:LIB = $oldLib
}
