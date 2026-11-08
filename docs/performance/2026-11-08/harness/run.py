"""Fresh-context, fixed-token comparisons. Owns only its replay child."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / 'decode-diagnosis-20261108'
PERF = HERE.parents[1]
p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['stock', 'control', 'compact', 'batch', 'n1', 'combined', 'compact-n1', 'serial', 'serial-n1'], default='control')
p.add_argument('--threads', default='24,8')
p.add_argument('--count', type=int, default=320)
p.add_argument('--prompt', type=int, default=8192)
p.add_argument('--profile', action='store_true')
p.add_argument('--label', required=True)
p.add_argument('--prepare', action='store_true')
p.add_argument('--matrix', action='store_true')
p.add_argument('--serial-matrix', action='store_true')
p.add_argument('--ctx', type=int)
p.add_argument('--batch', type=int)
p.add_argument('--slots', type=int)
p.add_argument('--cache-slots', type=int)
p.add_argument('--budget', type=int)
p.add_argument('--admission', type=int)
p.add_argument('--admissions', help='Comma separated refill budgets, one fresh context each')
p.add_argument('--pipeline', action='store_true')
p.add_argument('--pipeline-matrix', action='store_true')
p.add_argument('--cuda-dll', type=Path)
p.add_argument('--multi-cache', action='store_true')
p.add_argument('--stream-mode', choices=['primary', 'layer'])
p.add_argument('--tensor-split')
p.add_argument('--cpu-moe', type=int)
p.add_argument('--phase', action='store_true')
p.add_argument('--phase-matrix', action='store_true')
p.add_argument('--reentry', action='store_true')
p.add_argument('--layouts', help='Comma separated primary:27 or layer:44 cases; same weights/context/batch/threads')
a = p.parse_args()
out = HERE / a.label
out.mkdir(exist_ok=False)
launcher = (HERE / 'original-launcher.ps1').read_text(encoding='utf-8-sig')
argv = re.findall(r"'([^']*)'", launcher.split('$serverArgs = @(', 1)[1].split('\n)', 1)[0])
for key, value in [('-c', a.ctx), ('-fitc', a.ctx), ('-b', a.batch), ('-ub', a.batch),
                   ('-ts', a.tensor_split),
                   ('-ncmoe', a.cpu_moe),
                   ('--fn-stream-gpu-mode', a.stream_mode),
                   ('--fn-stream-slots', a.slots), ('--fn-stream-budget-mib', a.budget)]:
    if value is not None: argv[argv.index(key)+1] = str(value)
controls = dict(re.findall(r"\$env:(\w+)\s*=\s*'([^']*)'", launcher))
active = Path(re.search(r"&\s*'([^']+llama-server\.exe)'", launcher)[1]).parent
stock = HERE / 'stock-bin'
candidate = HERE / 'candidate-bin'
if a.prepare:
    for dest in [stock, candidate]:
        dest.mkdir(exist_ok=True)
        for dll in active.glob('*.dll'): shutil.copy2(dll, dest / dll.name)
        shutil.copy2(HERE / 'build/bin/decode-replay.exe', dest / 'decode-replay.exe')
    for name in ['ggml-base.dll', 'ggml-cpu.dll', 'llama.dll', 'llama-common.dll', 'test-backend-ops.exe']:
        shutil.copy2(HERE / 'build/bin' / name, candidate / name)
exe = (stock if a.variant == 'stock' else candidate) / 'decode-replay.exe'
if a.cuda_dll: shutil.copy2(a.cuda_dll, candidate / 'ggml-cuda.dll')
token_source = DIAG / ('stages-t24-p32768/measured/response.json' if a.prompt == 32768 else 'original-t24-p8192/measured/response.json')
tokens = json.loads(token_source.read_text())['tokens'][:a.count]
assert len(tokens) == a.count
prompt = json.loads((DIAG / f'prompt-{a.prompt}.json').read_text())['tokens']
cases = [(a.variant, int(n)) for n in a.threads.split(',')]
if a.matrix:
    assert a.variant in ['control', 'n1']
    cases = [(variant, int(n)) for n in a.threads.split(',') for variant in
             (['control', 'compact', 'batch'] if a.variant == 'control' else ['n1', 'compact-n1', 'combined'])]
if a.serial_matrix:
    assert a.variant in ['control', 'n1']
    cases = [(variant, int(n)) for n in a.threads.split(',') for variant in
             (['control', 'serial'] if a.variant == 'control' else ['n1', 'serial-n1'])]
runs = [{'label': f'{a.label}-{i}-{variant}-t{n}', 'threads': n, 'variant': variant,
         'compact': variant in ['compact', 'compact-n1', 'combined'],
         'batch_d2h': variant in ['batch', 'combined'],
         'serial': variant in ['serial', 'serial-n1'],
         'output': str(out / f'result-{i}-{variant}-t{n}.json')}
        for i, (variant,n) in enumerate(cases)]
if a.admissions:
    assert len(cases) == 1
    base = runs[0]
    runs = [dict(base, label=f'{a.label}-{i}-a{mib}', admission_mib=int(mib),
                 output=str(out / f'result-{i}-a{mib}.json'))
            for i, mib in enumerate(a.admissions.split(','))]
if a.pipeline_matrix:
    runs = [dict(run, label=run['label']+f'-pipe{flag}', pipeline=bool(flag),
                 output=run['output'].replace('.json', f'-pipe{flag}.json'))
            for run in runs for flag in [0, 1]]
if a.layouts:
    assert len(runs) == 1
    base = runs[0]
    runs = []
    for i, layout in enumerate(a.layouts.split(',')):
        mode, slots = layout.split(':')
        assert mode in ['primary', 'layer']
        runs.append(dict(base, label=f'{a.label}-{i}-{mode}{slots}', stream_mode=mode, cache_slots=int(slots),
                         output=str(out / f'result-{i}-{mode}{slots}.json')))
if a.phase_matrix:
    assert len(runs) == 1
    base = runs[0]
    runs = [dict(base,label=f'{a.label}-{i}-phase{flag}',phase_layout=bool(flag),
                 output=str(out / f'result-{i}-phase{flag}.json'))
            for i,flag in enumerate([0,1,1,0])]
config = {'prompt': prompt, 'tokens': tokens, 'warm': 64, 'runs': runs,
          'reference': str(HERE / f'reference-p{a.prompt}-n{a.count}.bin'), 'flush_profiles': a.profile, 'reentry':a.reentry}
(out / 'config.json').write_text(json.dumps(config))
env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_', 'QWEN4EXP_', 'OMP_'))}
env.update(controls)
env['DECODE_REPLAY_CONFIG'] = str(out / 'config.json')
if a.variant in ['compact', 'compact-n1', 'combined']: env['LLAMA_FN_CACHE_COMPACT_DECODE'] = '1'
if a.variant in ['n1', 'compact-n1', 'combined', 'serial-n1']: env['GGML_CPU_MOE_N1'] = '1'
if a.variant in ['batch', 'combined']: env['GGML_SCHED_BATCH_D2H'] = '1'
if a.variant in ['serial', 'serial-n1']: env['GGML_SCHED_SERIAL_CACHE_PREP'] = '1'
if a.cache_slots is not None: env['LLAMA_MOE_CACHE_SLOTS'] = str(a.cache_slots)
if a.admission is not None: env['LLAMA_MOE_CACHE_ADMISSION_MIB'] = str(a.admission)
if a.pipeline: env['GGML_CUDA_CACHE_PIPELINE'] = '1'
if a.multi_cache: env['LLAMA_MOE_CACHE_MULTI_GPU'] = '1'
if a.phase: env['LLAMA_MOE_CACHE_PHASE_LAYOUT'] = '1'
if a.profile:
    env['GGML_HANDOFF_PROFILE'] = str(out / 'handoffs.csv')
    env['GGML_CPU_OP_PROFILE'] = str(out / 'cpu-ops.csv')
command = [str(exe)] + argv + ['--threads', '24', '--threads-batch', '24', '--no-warmup']
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {'argv': command, 'variant': a.variant,
    'env': {k:v for k,v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'QWEN4EXP_', 'DECODE_'))},
    'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
    'replay_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    'files': {f.name:digest(f) for f in sorted(exe.parent.glob('*')) if f.suffix in ['.exe', '.dll']}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
start = time.monotonic()
with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
    child = subprocess.Popen(command, cwd=PERF, env=env, stdout=stdout, stderr=stderr,
                             stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    (out / 'pid.json').write_text(json.dumps({'pid':child.pid, 'owned':True}))
    try:
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - start > 1800:
                    raise
                log = (out / 'stderr.log').read_text(errors='replace')
                if 'moe_cache: shared-pool cache unavailable' in log:
                    (out / 'failure.txt').write_text('Required expert cache did not initialize; replay cancelled before reporting fallback performance.\n')
                    child.terminate()
                    child.wait(timeout=30)
                    code = 125
                    break
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=30)
(out / 'exit.json').write_text(json.dumps({'code':code, 'elapsed':time.monotonic()-start}))
print(json.dumps({'label':a.label, 'exit':code, 'elapsed':time.monotonic()-start}), flush=True)
for result in sorted(out.glob('result-*.json')):
    row = json.loads(result.read_text()); row.pop('tokens')
    print(json.dumps(row), flush=True)
raise SystemExit(code)
