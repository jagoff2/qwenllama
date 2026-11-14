param(
    [Parameter(Mandatory)][string]$Server,
    [string]$ChatTemplate,
    [ValidateSet('Loop', 'Stock', 'Neutral')][string]$Mode = 'Loop',
    [ValidateRange(1,65535)][int]$Port = 8080,
    [ValidateRange(512,261888)][int]$Context = 261888,
    [ValidateRange(64,4096)][int]$MicroBatch = 2048,
    [ValidateRange(0,8192)][int]$PrefillReserveMiB = 512,
    [Parameter(Mandatory)][string]$Model,
    [string]$MMProj,
    [switch]$DisableExpertStreaming,
    [switch]$DisableFusion,
    [switch]$DisableGraphReuse,
    [switch]$DisableCudaGraphs,
    [switch]$TraceOptimizations
)
$ErrorActionPreference = 'Stop'

# Published adapter of the tested launcher. External asset paths are explicit.
Get-ChildItem Env: | Where-Object { $_.Name -match '^(LLAMA_|GGML_|QWEN4EXP_|OMP_|DECODE_)' } | ForEach-Object {
    [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
}
$env:QWEN4EXP_QSA_GATHER = '1'
$env:QWEN4EXP_QSA_QUERY_CHUNK = '256'
$env:LLAMA_ATTN_ROT_DISABLE = '1'
$env:LLAMA_ARG_SPEC_TYPE = 'none'
$env:GGML_CPU_MOE_N1 = '1'
$env:LLAMA_MOE_CACHE_PHASE_LAYOUT = '1'
$env:LLAMA_MOE_CACHE_RESERVE_MIB = '768'
$env:LLAMA_MOE_STREAM_RESERVE_MIB = [string]$PrefillReserveMiB

# Set these after the original environment cleanup. Restart the server to change mode.
$env:QWEN4EXP_LOOP_CONFIG = Join-Path $PSScriptRoot ('config\' + $Mode.ToLowerInvariant() + '.txt')
if ($DisableFusion) { $env:GGML_CUDA_DISABLE_FUSION = '1' }
if ($DisableCudaGraphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }
if ($DisableGraphReuse) { $env:LLAMA_GRAPH_REUSE_DISABLE = '1' }
if ($TraceOptimizations) { $env:GGML_CUDA_OPT_TRACE = '1' }
# Recurrence needs more prefill scratch. Bootstrap in a one-slot stream arena;
# the existing phase transition expands decode capacity independently.
$env:LLAMA_MOE_CACHE_SLOTS = '12'
$env:LLAMA_MOE_CACHE_DECODE_BOOTSTRAP_MIB = '2048'
$env:LLAMA_MOE_CACHE_DECODE_BOOTSTRAP_SLOTS = '16'
$server = (Resolve-Path -LiteralPath $Server).Path
$template = $ChatTemplate
foreach ($required in @($server, $Model, $env:QWEN4EXP_LOOP_CONFIG) + @($template, $MMProj | Where-Object { $_ })) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing required file: $required" }
}
Write-Host "Dory recurrence mode=$Mode context=$Context port=$Port"
Write-Host "Server: $server"
Write-Host "Microbatch=$MicroBatch; prefill reserve=$PrefillReserveMiB MiB"
Write-Host ('Fusion={0}; graph reuse={1}; CUDA graphs={2}' -f (-not $DisableFusion), (-not $DisableGraphReuse), (-not $DisableCudaGraphs))
$serverArgs = @(
    '-m',
    $Model,
    '--spec-type',
    'none',
    '--jinja',
    '-c',
    "$Context",
    '-fitc',
    "$Context",
    '-v',
    '-sm',
    'layer',
    '-ts',
    '1,1',
    '-lm',
    'none',
    '-lzm',
    'off',
    '-fa',
    'on',
    '-np',
    '1',
    '--no-kv-unified',
    '--samplers',
    'penalties;top_k;top_p;min_p;temperature',
    '--temp',
    '1.0',
    '--top-p',
    '0.95',
    '--top-k',
    '20',
    '--min-p',
    '0.0',
    '--presence-penalty',
    '0.0',
    '--frequency-penalty',
    '0.0',
    '--repeat-penalty',
    '1.0',
    '--reasoning',
    'on',
    '--reasoning-budget',
    '-1',
    '--reasoning-format',
    'none',
    '--metrics',
    '-b',
    '4096',
    '-ub',
    [string]$MicroBatch,
    '-ctk',
    'f16',
    '-ctv',
    'f16',
    '-cram',
    '32768',
    '--cache-idle-slots',
    '-lv',
    '4',
    '-ngl',
    'all',
    '-ncmoe',
    '46',
    '-fit',
    'off',
    '--fn-stream-experts',
    '--fn-decode-expert-cache',
    '--fn-stream-slots',
    '2',
    '--fn-stream-min-tokens',
    '1024',
    '--fn-stream-gpu-mode',
    'primary',
    '--fn-stream-budget-mib',
    '4096',
    '--no-webui',
    '--host',
    '127.0.0.1',
    '--threads',
    '8',
    '--threads-batch',
    '24'
)

if ($MMProj) { $serverArgs += @('--mmproj', $MMProj, '--no-mmproj-offload') }
if ($template) { $serverArgs += @('--chat-template-file', $template) }

if ($DisableExpertStreaming) {
    $serverArgs = @($serverArgs | Where-Object { $_ -notin @('--fn-stream-experts', '--fn-decode-expert-cache') })
}
$serverArgs += @('--port', "$Port")
& $server @serverArgs
exit $LASTEXITCODE
