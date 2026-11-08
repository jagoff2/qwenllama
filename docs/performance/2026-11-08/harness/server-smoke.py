"""Verify packaged phase transitions through real HTTP chat requests."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.request

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--label', default='server-smoke-phase')
parser.add_argument('--predict', type=int, default=192)
parser.add_argument('--baseline', action='store_true', help='Use original pools with the same 8-thread/N=1 build')
args = parser.parse_args()
if args.predict < 1: parser.error('--predict must be positive')
out = here / args.label
out.mkdir(exist_ok=False)
manifest = json.loads((here / 'candidate-manifest.json').read_text())
command = manifest['argv'][:]
assert command[1:-4] == manifest['original_argv'], 'Original prefill arguments changed'
assert command[-4:] == ['--threads', '8', '--threads-batch', '24']
assert manifest['env']['GGML_CPU_MOE_N1'] == '1'
assert manifest['env']['LLAMA_MOE_CACHE_PHASE_LAYOUT'] == '1'
assert 'LLAMA_MOE_CACHE_SLOTS' not in manifest['env']
assert 'LLAMA_MOE_CACHE_MULTI_GPU' not in manifest['env']
for name, expected in manifest['files'].items():
    path = Path(command[0]).parent / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Package changed: {name}'
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
command += ['--port', str(port)]
prefixes = tuple(manifest['clear_env_prefixes'])
env = {k:v for k,v in os.environ.items() if not k.upper().startswith(prefixes)}
for key in manifest.get('clear_env', []): env.pop(key, None)
env.update(manifest['env'])
if args.baseline:
    env['LLAMA_MOE_CACHE_PHASE_LAYOUT'] = '0'
base = f'http://127.0.0.1:{port}'

def post(path, payload, label):
    (out / f'{label}-request.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    request = urllib.request.Request(base + path, json.dumps(payload).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    (out / f'{label}-response.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result

def stream_chat(payload, label):
    payload = dict(payload, stream=True, stream_options={'include_usage':True}, timings_per_token=True)
    (out / f'{label}-request.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type':'application/json'})
    started = time.monotonic()
    events, arrivals = [], []
    message, usage, timings, finish = {'role':'assistant', 'content':''}, None, None, None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b'data:'): continue
            data = raw[5:].strip()
            if data == b'[DONE]': break
            event = json.loads(data)
            arrived = time.monotonic() - started
            events.append({'arrival_seconds':arrived, 'data':event})
            if event.get('error'): raise RuntimeError(event['error'])
            if event.get('usage'): usage = event['usage']
            if event.get('timings'):
                timings = event['timings']
                count = timings.get('predicted_n', 0)
                if count > (arrivals[-1]['token'] if arrivals else 0):
                    arrivals.append({'token':count, 'seconds':arrived})
            for choice in event.get('choices', []):
                if choice.get('finish_reason'): finish = choice['finish_reason']
                for key, value in choice.get('delta', {}).items():
                    if key in ('content', 'reasoning_content') and value:
                        message[key] = message.get(key, '') + value
    elapsed = time.monotonic() - started
    (out / f'{label}-events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in events), encoding='utf-8')
    assert usage and timings and len(arrivals) >= 2, 'Missing streamed usage or per-token timing counters'
    intervals = [{'from_token':a['token'], 'to_token':b['token'], 'tokens':b['token']-a['token'],
                  'seconds':b['seconds']-a['seconds']} for a,b in zip(arrivals, arrivals[1:])]
    tail = [point for point in arrivals if point['token'] >= max(64, arrivals[-1]['token']-128)]
    steady = None
    if len(tail) >= 2 and tail[-1]['token']-tail[0]['token'] >= 64:
        steady = {'start_token':tail[0]['token'], 'end_token':tail[-1]['token'],
                  'tokens':tail[-1]['token']-tail[0]['token'], 'seconds':tail[-1]['seconds']-tail[0]['seconds']}
        steady['tokens_per_second'] = steady['tokens']/steady['seconds']
    measured = {'request_seconds':elapsed, 'first_observed_token_seconds':arrivals[0]['seconds'],
                'first_observed_token_index':arrivals[0]['token'],
                'completion_tokens_per_request_second':usage['completion_tokens']/elapsed,
                'server_predicted_per_second':timings['predicted_per_second'],
                'first_decode_interval':intervals[0], 'max_arrival_gap_seconds':max(i['seconds'] for i in intervals),
                'steady_after_at_least_64_tokens':steady, 'token_arrival_intervals':intervals,
                'measurement':'Client SSE arrival times indexed by server predicted_n; chunks may contain multiple tokens.'}
    result = {'object':'chat.completion', 'choices':[{'message':message, 'finish_reason':finish}],
              'usage':usage, 'timings':timings, 'client_stream_measurements':measured}
    (out / f'{label}-response.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result

def check_generation(result):
    assert result['object'] == 'chat.completion'
    assert result['usage']['completion_tokens'] == args.predict
    assert result['choices'][0]['finish_reason'] == 'length'
    message = result['choices'][0]['message']
    assert message.get('content') or message.get('reasoning_content'), 'Empty assistant response'
    return message

def check_phase_log(log, reentry=False):
    failures = ['shared-pool cache unavailable', 'generation uses baseline experts',
                'restoring original arena layout', 'cannot restore', 'CUDA error',
                'decode reservation failed', 'streamed full-memory compute reservation failed']
    assert not any(text in log for text in failures), 'Backend fallback/failure in server log'
    if args.baseline:
        assert 'moe_phase:' not in log
        reports = [line for line in log.splitlines() if 'llama_moe_stream_report: ubatch' in line]
        assert reports and all('errors 0 | devices 1' in line for line in reports)
        if not reentry:
            assert '1 devices, 46 layers, 27 slots/layer' in log
            assert 'n_ctx = 261888, n_tokens = 4096' in log
        return {'prefill_microbatches':len(reports), 'prefill_devices':1, 'decode_devices':1,
                'decode_slots_per_layer':27,
                'prefill_batch_tokens':[int(re.search(r'ubatch (\d+) tokens', line)[1]) for line in reports]}
    switched = log.index('moe_phase: released prefill compute and expert pools')
    before, after = log[:switched], log[switched:]
    if reentry:
        assert 'moe_phase: released decode compute and expert pools' in before
    assert 'all arenas on device 0' in before
    assert 'device 0 armed with 2 expert arena slot(s)' in before
    assert 'device 1 armed with' not in before
    assert '1 devices, 46 layers, 27 slots/layer' in before
    assert 'n_ctx = 261888, n_tokens = 4096' in before
    reports = [line for line in before.splitlines() if 'llama_moe_stream_report: ubatch' in line]
    batches = [int(re.search(r'ubatch (\d+) tokens', line)[1]) for line in reports]
    # Server checkpoints can split a 4096-capacity batch at an earlier boundary.
    assert batches and all(1024 <= count <= 4096 for count in batches), 'Invalid streamed prefill batches'
    assert reports and all('errors 0 | devices 1' in line for line in reports)
    for device, layers in [(0, 25), (1, 21)]:
        assert f'device {device} armed with 2 expert arena slot(s)' in after
        assert f'moe_cache: device {device}, {layers} layers, 44 slots/layer' in after
    assert '2 devices, 46 layers, 44 slots/layer' in after
    assert 'n_ctx=261888, rows=49' in after
    assert 'prompt seed ready' in after
    return {'prefill_microbatches': len(reports), 'prefill_batch_tokens': batches, 'prefill_devices': 1,
            'prefill_arenas': 2, 'decode_devices': 2, 'decode_slots_per_layer': 44}

start = time.monotonic()
with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
    child = subprocess.Popen(command, cwd=here.parents[1], env=env, stdout=stdout, stderr=stderr,
                             stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    recorded_env = {key:env[key] for key in manifest['env']}
    (out / 'process.json').write_text(json.dumps({'pid':child.pid, 'owned':True, 'argv':command, 'env':recorded_env}))
    try:
        while True:
            if child.poll() is not None: raise RuntimeError(f'server exited {child.returncode}')
            if time.monotonic() - start > 300: raise TimeoutError('server startup')
            try:
                with urllib.request.urlopen(base + '/health', timeout=2) as response:
                    if response.status == 200: break
            except Exception:
                time.sleep(1)
        tokens = json.loads((here.parent / 'decode-diagnosis-20261108/prompt-8192.json').read_text())['tokens']
        content = post('/detokenize', {'tokens':tokens}, 'fixture')['content']
        messages = [{'role':'user', 'content':content}]
        common = {'max_tokens':args.predict, 'temperature':0, 'seed':20261108,
                  'ignore_eos':True, 'cache_prompt':True, 'id_slot':0}
        initial = stream_chat(dict(common, messages=messages), 'single')
        assistant = check_generation(initial)
        first_log = (out / 'stderr.log').read_text(errors='replace')
        first_checks = check_phase_log(first_log)
        # A large added turn must restore the original prefill pools, then decode pools.
        messages += [assistant, {'role':'user', 'content':'Continue the discussion using this additional material:\n' + content}]
        continued = stream_chat(dict(common, messages=messages), 'continued')
        check_generation(continued)
        assert continued['usage']['prompt_tokens_details']['cached_tokens'] > 0, 'Conversation prefix was not reused'
        last_log = (out / 'stderr.log').read_text(errors='replace')
        continued_checks = check_phase_log(last_log[len(first_log):], reentry=True)
        validation = {'ok':True, 'baseline':args.baseline, 'elapsed':time.monotonic()-start,
                      'single':{'usage':initial['usage'], 'timings':initial['timings'],
                                'stream_measurements':initial['client_stream_measurements'], 'checks':first_checks},
                      'continued':{'usage':continued['usage'], 'timings':continued['timings'],
                                   'stream_measurements':continued['client_stream_measurements'], 'checks':continued_checks}}
        (out / 'validation.json').write_text(json.dumps(validation, indent=2))
        print(json.dumps(validation, indent=2))
    except Exception as error:
        (out / 'validation.json').write_text(json.dumps({'ok':False, 'error':str(error), 'elapsed':time.monotonic()-start}, indent=2))
        raise
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
