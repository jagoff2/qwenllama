$ErrorActionPreference = 'Stop'

$env:QWEN4EXP_QSA_GATHER = '1'
$env:QWEN4EXP_QSA_QUERY_CHUNK = '256'
$env:LLAMA_ATTN_ROT_DISABLE = '1'
$env:LLAMA_ARG_SPEC_TYPE = 'none'

$serverArgs = @(
    '-m',
    'C:\Users\user\Downloads\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',

    '--mmproj',
    'G:\qwenllamaMTP\perf-20260911\assets\Qwen3.8-Flash-Next\mmproj-F16.gguf',

    '--no-mmproj-offload',

	'--spec-type', 'none',

    '--jinja',

    '-c',
    '261888',

    '-fitc',
    '261888',
	
	'-v',

    '-sm',
    'layer',

    # Keep the proven target placement.
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

	'--reasoning-format', 'none',


    '--chat-template-file',
    'G:\qwenllamaMTP\qwen38-codex-system-role-demoted.jinja',

    '--metrics',

    # Slightly larger than the proven 4096 baseline.
    '-b',
    '4096',

    '-ub',
    '4096',

    # Keep target KV F16.
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

    # Proven double-buffer configuration.
    '--fn-stream-slots',
    '2',

    '--fn-stream-min-tokens',
    '1024',

    # Fastest tested topology:
    # both expert arenas on the fast CUDA0/x8 link.
    '--fn-stream-gpu-mode',
    'primary',

    '--fn-stream-budget-mib',
    '4096',

    '--no-webui',

    '--host',
    '127.0.0.1'
)

& 'G:\qwenllamaMTP\perf-20260911\dist\fn-decode-cache-think-guard-20261006\bin\llama-server.exe' @serverArgs


