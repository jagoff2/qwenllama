"""One short native completion per installed build, sequential and isolated."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--only', choices=['stock', 'fast'])
parser.add_argument('--suffix', default='')
parser.add_argument('--chat', action='store_true')
args = parser.parse_args()
PERF = HERE.parents[1]
prompt = json.loads((HERE.parent / 'decode-diagnosis-20261108/prompt-8192.json').read_text())['tokens'][:1024]
fast_manifest = json.loads((HERE.parent / 'decode-optimization-20261108/candidate-manifest.json').read_text())
cases = [
    ('stock', Path(r'C:\Users\user\qwenllama\llama-server.exe'), {}),
    ('fast', PERF / 'dist/fn-fast-prefill-local-decode-20261108/bin/llama-server.exe', fast_manifest['env']),
]
common = ['-m', r'C:\Users\user\Downloads\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
          '-c', '8192', '-fitc', '8192', '-b', '4096', '-ub', '4096', '-np', '1',
          '-ngl', 'all', '-ncmoe', '46', '-sm', 'layer', '-ts', '1,1', '-fa', 'on',
          '-ctk', 'f16', '-ctv', 'f16', '-fit', 'off', '-lm', 'none', '--no-kv-unified',
          '--threads', '8', '--threads-batch', '24', '--no-warmup', '--ctx-checkpoints', '0',
          '--no-webui', '--host', '127.0.0.1', '--spec-type', 'none', '-lv', '4']
stream_args = ['--fn-stream-experts', '--fn-decode-expert-cache', '--fn-stream-slots', '2',
               '--fn-stream-min-tokens', '1024', '--fn-stream-gpu-mode', 'primary',
               '--fn-stream-budget-mib', '4096']
request_body = {'prompt':prompt, 'n_predict':192, 'temperature':0, 'top_k':1, 'top_p':1,
                'min_p':0, 'repeat_penalty':1, 'seed':20261108, 'ignore_eos':True,
                'cache_prompt':False, 'stream':True, 'timings_per_token':True, 'return_tokens':True}
if args.chat:
    paragraph = ('The machine runs one large mixture of experts language model using system RAM and two graphics cards. '
                 'Most expert weights live in system RAM because they do not all fit in graphics memory. '
                 'Every generated token selects a small subset of experts in every layer. '
                 'The processor evaluates experts that are not resident on a graphics card. '
                 'An expert cache can reduce processor work but requires copies, synchronization, and careful buffer lifetimes. '
                 'Prompt processing can batch many tokens together, while single token generation has little work per operation. ')
    content = '\n\n'.join(f'Observation {i+1}: '+paragraph for i in range(8))
    content += ('\n\nExplain the performance of this system in a detailed technical tutorial of at least 1500 words. '
                'Cover memory bandwidth, arithmetic intensity, cache admission, synchronization, thread counts, '
                'prefill versus decode, measurement methodology, and numerical correctness. '
                'Use concrete examples and equations. Start your explanation immediately.')
    request_body = {'messages':[{'role':'user','content':content}], 'max_tokens':192,
                    'temperature':0, 'top_k':1, 'top_p':1, 'min_p':0, 'repeat_penalty':1,
                    'seed':20261108, 'ignore_eos':False, 'cache_prompt':False, 'stream':True,
                    'timings_per_token':True, 'stream_options':{'include_usage':True}}
(HERE / ('request-chat.json' if args.chat else 'request.json')).write_text(json.dumps(request_body))
results = []
for name, exe, controls in cases:
    if args.only and name != args.only:
        results.append(json.loads((HERE / name / 'result.json').read_text(encoding='utf-8')))
        continue
    out = HERE / (name + args.suffix)
    out.mkdir(exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    argv = [str(exe)] + common + (stream_args if name == 'fast' else []) + ['--port', str(port)]
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('LLAMA_', 'GGML_', 'QWEN4EXP_', 'OMP_', 'DECODE_'))}
    env.update(controls)
    version = subprocess.check_output([str(exe), '--version'], stderr=subprocess.STDOUT, text=True, env=env)
    (out / 'manifest.json').write_text(json.dumps({'argv':argv, 'env':controls, 'version':version,
        'prompt_sha256':hashlib.sha256(json.dumps(request_body).encode()).hexdigest()}, indent=2))
    started = time.perf_counter()
    with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
        child = subprocess.Popen(argv, cwd=PERF, env=env, stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        (out / 'process.json').write_text(json.dumps({'pid':child.pid, 'owned':True}))
        try:
            base = f'http://127.0.0.1:{port}'
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'{name} exited {child.returncode}: '+(out/'stderr.log').read_text(errors='replace')[-2500:])
                if time.perf_counter()-started > 180: raise TimeoutError('startup')
                try:
                    with urllib.request.urlopen(base+'/health', timeout=1) as response:
                        if response.status == 200: break
                except Exception:
                    time.sleep(.5)
            startup_s = time.perf_counter()-started
            print(name, 'loaded in', round(startup_s,2), 'seconds', flush=True)
            request = urllib.request.Request(base+('/v1/chat/completions' if args.chat else '/completion'), json.dumps(request_body).encode(), {'Content-Type':'application/json'})
            begin = time.perf_counter()
            points, events, tokens, content = [], [], [], ''
            final = None
            with urllib.request.urlopen(request, timeout=180) as response:
                for line in response:
                    if not line.startswith(b'data:'): continue
                    data = line[5:].strip()
                    if data == b'[DONE]': break
                    event = json.loads(data)
                    elapsed = time.perf_counter()-begin
                    events.append({'seconds':elapsed, 'data':event})
                    if event.get('error'): raise RuntimeError(event['error'])
                    content += event.get('content','')
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        content += delta.get('reasoning_content') or ''
                        content += delta.get('content') or ''
                    tokens.extend(event.get('tokens',[]))
                    count = event.get('timings',{}).get('predicted_n',0)
                    if count > (points[-1]['token'] if points else 0):
                        points.append({'token':count,'seconds':elapsed})
                    if event.get('stop'): final = event
                    if args.chat and event.get('timings'): final = event
            request_s = time.perf_counter()-begin
            assert final and final['timings']['predicted_n'] == 192
            if not args.chat: assert final['timings']['prompt_n'] == 1024, final['timings']
            assert len(points) >= 128, 'Missing per-token stream timing'
            tail = [point for point in points if point['token'] >= 64]
            steady = (tail[-1]['token']-tail[0]['token'])/(tail[-1]['seconds']-tail[0]['seconds'])
            result = {'case':name, 'version':version.strip(), 'startup_s':startup_s, 'request_s':request_s,
                      'timings':final['timings'], 'first_token_s':points[0]['seconds'],
                      'first_decode_gap_s':points[1]['seconds']-points[0]['seconds'],
                      'steady_from_token':tail[0]['token'], 'steady_to_token':tail[-1]['token'],
                      'steady_tps':steady, 'tokens':tokens, 'content':content}
            (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
            (out/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events),encoding='utf-8')
            if name == 'fast':
                log = (out/'stderr.log').read_text(errors='replace')
                assert 'moe_phase: released prefill compute and expert pools' in log
                assert '2 devices, 46 layers, 44 slots/layer' in log
                assert 'generation uses baseline experts' not in log
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ['tokens','content']},indent=2),flush=True)
        except Exception as error:
            (out/'failure.json').write_text(json.dumps({'error':str(error)},indent=2))
            raise
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
summary = {'scope':'one quick run per build; identical prompt / 192 generated; context 8192; threads 8 / batch 24',
           'results':[{k:v for k,v in r.items() if k not in ['tokens','content']} for r in results],
           'same_output':results[0]['content']==results[1]['content'],
           'steady_gain_pct':100*(results[1]['steady_tps']/results[0]['steady_tps']-1)}
assert results[0]['timings']['prompt_n'] == results[1]['timings']['prompt_n']
(HERE/('summary-chat.json' if args.chat else 'summary.json')).write_text(json.dumps(summary,indent=2))
