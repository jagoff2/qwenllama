$ErrorActionPreference = 'Stop'
$originalPath = $env:PATH
$oldInclude = $env:INCLUDE
$oldLib = $env:LIB
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822'
$sdk = 'C:\Program Files (x86)\Windows Kits\10'
$sdkVersion = '10.0.26100.0'
$work = 'G:\qwenllamaMTP\experiments\dory_server_20261113'
try {
    $env:INCLUDE = "$vc\include;$sdk\Include\$sdkVersion\ucrt;$sdk\Include\$sdkVersion\shared;$sdk\Include\$sdkVersion\um;$sdk\Include\$sdkVersion\winrt"
    $env:LIB = "$vc\lib\x64;$sdk\Lib\$sdkVersion\ucrt\x64;$sdk\Lib\$sdkVersion\um\x64"
    & 'C:\Users\user\AppData\Local\Programs\Python\Python3119\Scripts\ninja.exe' -C "$work\build" -j 16 llama
    if ($LASTEXITCODE) { throw "Core build failed: $LASTEXITCODE" }
    $testRoot = Join-Path $PSScriptRoot 'planner-test'
    $null = New-Item -ItemType Directory -Path $testRoot -Force
    Push-Location $testRoot
    try {
        & "$vc\bin\Hostx64\x64\cl.exe" /nologo /O2 /EHsc /std:c++17 /MD /DGGML_SHARED /DGGML_BACKEND_SHARED "/I$work\source\src" "/I$work\source\include" "/I$work\source\ggml\include" "$work\source\tests\test-moe-stream-plan.cpp" "$work\source\src\llama-moe-stream-plan.cpp" /Fe:test-moe-stream-plan.exe /link "$work\build\ggml\src\ggml-base.lib"
        if ($LASTEXITCODE) { throw "Planner test build failed: $LASTEXITCODE" }
        Copy-Item -LiteralPath "$work\build\bin\ggml-base.dll" -Destination $testRoot
        & '.\test-moe-stream-plan.exe'
        if ($LASTEXITCODE) { throw "Planner test failed: $LASTEXITCODE" }
        & "$vc\bin\Hostx64\x64\cl.exe" /nologo /O2 /EHsc /std:c++17 /MD /DGGML_SHARED /DGGML_BACKEND_SHARED "/I$work\source\ggml\include" "/I$work\source\ggml\src" "$PSScriptRoot\allocation_failure_regression.cpp" /Fe:allocation-failure-regression.exe /link "$work\build\ggml\src\ggml-base.lib"
        if ($LASTEXITCODE) { throw "Allocation regression build failed: $LASTEXITCODE" }
        & '.\allocation-failure-regression.exe'
        if ($LASTEXITCODE) { throw "Allocation regression failed: $LASTEXITCODE" }
    } finally { Pop-Location }
    if ($env:PATH -cne $originalPath) { throw 'PATH changed unexpectedly' }
} finally {
    $env:INCLUDE = $oldInclude
    $env:LIB = $oldLib
}
