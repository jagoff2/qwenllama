$ErrorActionPreference = 'Stop'
$scheduleRoot = $PSScriptRoot
$repoRoot = 'G:\qwenllamaMTP\llama.cpp'
$scheduleCompiler = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64\cl.exe'
$scheduleOldInclude = $env:INCLUDE
$scheduleOldLib = $env:LIB
Push-Location $repoRoot
try {
    $env:INCLUDE='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um'
    $env:LIB='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.37.32822\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64'
    & $scheduleCompiler /nologo /std:c++17 /EHsc /O2 /MD /I common /I include /I ggml\include /I vendor /Fe:build\bin\latent-eval-schedule.exe "/Fo:$scheduleRoot\latent_eval_schedule.obj" "$scheduleRoot\latent_eval_schedule.cpp" /link build\common\llama-common.lib build\src\llama.lib build\ggml\src\ggml.lib build\ggml\src\ggml-base.lib
    if ($LASTEXITCODE -ne 0) { throw 'Schedule harness build failed' }
} finally {
    $env:INCLUDE = $scheduleOldInclude
    $env:LIB = $scheduleOldLib
    Pop-Location
}
