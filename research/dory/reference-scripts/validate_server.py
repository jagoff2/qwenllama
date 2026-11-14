"""Serial integration tests. Only newly generated diagnostics are read."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
PACKAGE = Path(r'G:\qwenllamaMTP\perf-20260911\dist\fn-dory-recurrence-graphs-20261113')
PS = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
PORT = 18083
URL = f'http://127.0.0.1:{PORT}'

def http(endpoint, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + endpoint, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def probability_values(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('prob','logprob') and isinstance(item,(int,float)):
                yield item
            else:
                yield from probability_values(item)
    elif isinstance(value,list):
        for item in value:
            yield from probability_values(item)

def run(mode, context, disabled=False, short=False, tag='v2', launcher=None, extra_args=None, benchmark=False):
    label = tag + '_' + mode.lower() + ('_no_stream' if disabled else '') + f'_c{context}'
    out = ROOT / (label + '.current-stdout.txt')
    err = ROOT / (label + '.current-stderr.txt')
    assert not out.exists() and not err.exists(), 'Use fresh labels; never read prior run logs.'
    launcher = launcher or PACKAGE/'launch-best-decode-no-draft-think-guard-dory.ps1'
    args = [PS, '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(launcher), '-Mode',mode,'-Port',str(PORT),'-Context',str(context)]
    args += ['-TraceOptimizations'] + (extra_args or [])
    if disabled:
        args += ['-DisableExpertStreaming']
    before_path = os.environ.get('PATH')
    result = {'mode':mode,'context':context,'disable_stream':disabled,'tests':[]}
    result['extra_args'] = extra_args or []
    result['cuda_sha256'] = hashlib.sha256((PACKAGE/'bin/ggml-cuda.dll').read_bytes()).hexdigest()
    start = time.perf_counter()
    with out.open('w') as stdout, err.open('w') as stderr:
        process = subprocess.Popen(args, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while time.perf_counter()-start < 360:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode}')
                try:
                    if http('/health', timeout=2).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Server startup exceeded 360 seconds')
            result['startup_seconds'] = time.perf_counter()-start
            print(json.dumps({'ready':label,'seconds':round(result['startup_seconds'],2)}), flush=True)
            props = http('/props')
            result['props'] = props
            cases = [('short_a', '<|im_start|>user\nReturn only the capital of France.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')]
            if not short:
                seed = ' Reference data: amber cedar birch violet quartz river maple.\n' * 800
                pad = http('/tokenize', {'content':seed,'add_special':False})['tokens']
                prefix = http('/tokenize', {'content':'<|im_start|>user\nRead the reference data below.\n','add_special':False})['tokens']
                suffix = http('/tokenize', {'content':'\nAnswer with just this number: 37.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n','add_special':False})['tokens']
                for length in (1023,1025,4095,4113):
                    cases.append((f'prefill_{length}', prefix+pad[:length-len(prefix)-len(suffix)]+suffix))
                cases.append(('short_b','<|im_start|>user\nReturn only the result of 17 + 26.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'))
                cases.append(('short_a_repeat',cases[0][1]))
                cases.append(('short_a_cached',cases[0][1]))
                cases.append(('decode_128','<|im_start|>user\nWrite the integers from 1 through 50 in ascending order, separated by commas. No introduction.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'))
            if benchmark:
                counting='<|im_start|>user\nWrite the integers from 1 through 100 in ascending order, separated by commas. No introduction.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                cases=[case for case in cases if case[0]=='prefill_4095']
                cases += [('bench_warmup',counting)] + [(f'bench_{i}',counting) for i in range(3)]
            for name,prompt in cases:
                body = {'prompt':prompt,'n_predict':8,'temperature':0,'seed':123,'cache_prompt':False,'n_probs':5,'post_sampling_probs':False}
                if name.endswith('_cached'):
                    body['cache_prompt'] = True
                if name == 'decode_128':
                    body['n_predict'] = 128
                if name.startswith('bench_'):
                    body['n_predict'] = 64 if name=='bench_warmup' else 256
                t0 = time.perf_counter()
                response = http('/completion', body)
                elapsed = time.perf_counter()-t0
                probabilities = list(probability_values(response))
                finite = bool(probabilities) and all(math.isfinite(p) for p in probabilities)
                test = {'name':name,'wall_seconds':elapsed,'finite_returned_probs':finite,'response':response}
                result['tests'].append(test)
                print(json.dumps({'run':label,'test':name,'seconds':round(elapsed,2),'content':response.get('content') if not benchmark else response.get('content','')[:65],'evaluated':response.get('tokens_evaluated'),'finite':finite,'timings':response.get('timings')}),flush=True)
                assert finite
                assert response.get('content','').strip(), 'Empty generation'
            # Chat route must retain the copied template and reasoning configuration.
            chat = http('/v1/chat/completions', {'model':'local','messages':[{'role':'user','content':'What is 6 times 7?'}], 'temperature':0,'max_tokens':32,'seed':123})
            result['chat_response'] = chat
            result['completed'] = True
        except BaseException as e:
            result['error'] = repr(e)
            print('\n'.join(line for line in err.read_text(errors='replace').splitlines() if any(s in line for s in (' E ', ' W ', 'moe_cache:', 'moe_phase:', 'GGML_ASSERT')))[-8000:],flush=True)
            raise
        finally:
            subprocess.run([r'C:\Windows\System32\taskkill.exe','/PID',str(process.pid),'/T','/F'],capture_output=True)
            process.wait(timeout=30)
            assert os.environ.get('PATH') == before_path
            diagnostic = err.read_text(errors='replace')
            result['integration_diagnostics'] = [line for line in diagnostic.splitlines() if any(s in line for s in ('recurrent layer','MoE source classification','decode_cache:', 'moe_cache:', 'moe_stream:', 'loop_start','Qwen loop','cuda_opt_trace','graphs reused'))][-100:]
            (ROOT / (label + '.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=['Loop','Stock','Neutral'])
    p.add_argument('--context',type=int,default=8192)
    p.add_argument('--disable-stream',action='store_true')
    p.add_argument('--short',action='store_true')
    p.add_argument('--tag',default='v2')
    p.add_argument('--launcher',type=Path)
    p.add_argument('--no-fusion',action='store_true')
    p.add_argument('--no-reuse',action='store_true')
    p.add_argument('--no-capture',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    a=p.parse_args()
    run(a.mode,a.context,a.disable_stream,a.short,a.tag,a.launcher, [s for enabled,s in [(a.no_fusion,'-DisableFusion'),(a.no_reuse,'-DisableGraphReuse'),(a.no_capture,'-DisableCudaGraphs')] if enabled],a.benchmark)
