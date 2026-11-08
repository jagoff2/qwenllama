@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF > "%~dp0build.log" 2>&1 || exit /b 1
cmake --build "%~dp0build" --target decode-replay test-backend-ops -j 8 >> "%~dp0build.log" 2>&1
exit /b %ERRORLEVEL%
