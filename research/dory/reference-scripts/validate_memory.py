import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PS = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
PORT = 18084

def http(endpoint, body=None, timeout=1200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}' + endpoint, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def probabilities(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('prob', 'logprob') and isinstance(child, (float, int)):
                yield child
            else:
                yield from probabilities(child)
    elif isinstance(value, list):
        for child in value:
            yield from probabilities(child)

def run(args):
    folder = ROOT / args.label
    folder.mkdir(exist_ok=False)
    package = args.launcher.parent
    command = [PS, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(args.launcher),
               '-Port', str(PORT), '-TraceOptimizations', '-Mode', args.mode]
    if args.microbatch is not None:
        command += ['-MicroBatch', str(args.microbatch)]
    if args.reserve is not None:
        command += ['-PrefillReserveMiB', str(args.reserve)]
    original_path = os.environ.get('PATH')
    result = {'command': command, 'tests': [], 'gpu_samples': [], 'hashes': {
        name: hashlib.sha256((package / 'bin' / name).read_bytes()).hexdigest()
        for name in ('llama.dll', 'ggml-base.dll', 'ggml-cuda.dll')}}
    done = threading.Event()
    started = time.monotonic()
    def monitor():
        while not done.is_set():
            sample = subprocess.run([r'C:\Windows\System32\nvidia-smi.exe',
                '--query-gpu=index,memory.used,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            result['gpu_samples'].append({'seconds': time.monotonic()-started, 'memory': sample.stdout.strip()})
            done.wait(1)
    with (folder/'stdout.txt').open('w') as out, (folder/'stderr.txt').open('w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        try:
            while time.monotonic()-started < 360:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode}')
                try:
                    if http('/health', timeout=2).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('startup')
            print(json.dumps({'ready': args.label, 'seconds': time.monotonic()-started}), flush=True)
            pad = http('/tokenize', {'content': ' amber cedar birch violet quartz river maple.\n' * 6000,
                                    'add_special': False})['tokens']
            prefix = http('/tokenize', {'content': '<|im_start|>user\nReference data follows.\n', 'add_special': False})['tokens']
            suffix = http('/tokenize', {'content': '\nReturn only this number: 37.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', 'add_special': False})['tokens']
            for length in args.lengths:
                prompt = prefix + pad[:length-len(prefix)-len(suffix)] + suffix
                assert len(prompt) == length
                begin = time.monotonic()
                response = http('/completion', {'prompt': prompt, 'n_predict': args.predict,
                    'temperature': 0, 'seed': 123, 'cache_prompt': False, 'n_probs': 5})
                probs = list(probabilities(response))
                entry = {'length': length, 'seconds': time.monotonic()-begin,
                         'finite': bool(probs) and all(math.isfinite(v) for v in probs), 'response': response}
                result['tests'].append(entry)
                print(json.dumps({k: v for k, v in entry.items() if k != 'response'} | {
                    'content': response.get('content'), 'timings': response.get('timings')}), flush=True)
                assert entry['finite'] and response.get('content', '').strip()
                assert response.get('tokens_evaluated') == length
            if args.benchmark:
                counting = '<|im_start|>user\nWrite the integers from 1 through 100 in ascending order, separated by commas. No introduction.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                result['benchmark'] = []
                for repeat in range(4):
                    response = http('/completion', {'prompt': counting, 'n_predict': 64 if repeat == 0 else 256,
                        'temperature': 0, 'seed': 123, 'cache_prompt': False, 'n_probs': 5})
                    probs = list(probabilities(response))
                    assert probs and all(math.isfinite(v) for v in probs)
                    result['benchmark'].append(response)
                    print(json.dumps({'benchmark': repeat, 'timings': response['timings']}), flush=True)
            result['passed'] = True
        except Exception as error:
            result['error'] = repr(error)
            print(json.dumps({'error': repr(error)}), flush=True)
        finally:
            subprocess.run([r'C:\Windows\System32\taskkill.exe', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            process.wait(timeout=30)
            done.set()
            worker.join(timeout=10)
            assert os.environ.get('PATH') == original_path
            diagnostic = (folder/'stderr.txt').read_text(errors='replace')
            result['streamed_ubatches'] = diagnostic.count('llama_moe_stream_report: ubatch ')
            result['prefill_to_decode'] = diagnostic.count('released prefill compute and expert pools')
            result['decode_to_prefill'] = diagnostic.count('released decode compute and expert pools')
            result['cuda_graph_launch_lower_bound'] = max([int(n) for n in re.findall(r'graph_launch=(\d+)', diagnostic)] or [0])
            result['oom'] = 'cudaMalloc failed: out of memory' in diagnostic
            if args.require_stream and (not result['streamed_ubatches'] or not result['prefill_to_decode'] or result['oom']):
                result['passed'] = False
                result['error'] = 'Streaming or phase transitions missing, or OOM occurred'
            result['diagnostics'] = [line for line in diagnostic.splitlines() if any(s in line for s in (
                'compute buffer', 'cudaMalloc failed', 'failed to allocate', 'moe_phase:',
                'moe_cache:', 'cuda_opt_trace', 'progress =', 'prefill scratch'))]
            (folder/'result.json').write_text(json.dumps(result, indent=2))
            print('\n'.join(result['diagnostics'][-18:]), flush=True)
    return 0 if result.get('passed') else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--mode', default='Loop', choices=['Loop','Stock','Neutral'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[23486])
    parser.add_argument('--predict', type=int, default=16)
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--reserve', type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--require-stream', action='store_true')
    raise SystemExit(run(parser.parse_args()))
