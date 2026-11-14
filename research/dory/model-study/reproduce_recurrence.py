"""Portable reproduction of the frozen stock versus four-pass comparison.

Requires Python 3.10+, the patched latent-eval-schedule executable and its DLLs,
the user's GGUF shards, and the supplied task JSONL. Never edits model weights
or PATH. Each request receives a new context inside the C++ harness.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

STOCK = [-1, -1, 1, 1, 0, 0, 0]
CANDIDATE = [28, 31, 4, 1, 0, 0, 2, 0, 0, 0]
GRAMMAR = 'root ::= [1-9] [0-9]* "\\n"?'
ARMS = [('stock', STOCK), ('norm28_31x4', CANDIDATE)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model, manifest_path):
    records = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if len(records) != 4 or model.name != records[0]['name']:
        raise ValueError('Expected the first shard of the supplied four-shard model manifest')
    match = re.fullmatch(r'(.+)-00001-of-00004\.gguf', model.name)
    if not match:
        raise ValueError('Invalid first model shard filename')
    stem = match.group(1)
    verified = []
    for index, record in enumerate(records, 1):
        name = record['name']
        if (Path(name).name != name or '/' in name or '\\' in name or
                name != f'{stem}-{index:05d}-of-00004.gguf'):
            raise ValueError(f'Invalid model shard filename: {name}')
        path = model.parent / name
        if path.stat().st_size != record['bytes']:
            raise ValueError(f'Model shard size mismatch: {name}')
        actual = sha(path)
        if actual != record['sha256']:
            raise ValueError(f'Model shard SHA256 mismatch: {name}')
        verified.append({'name': name, 'bytes': path.stat().st_size, 'sha256': actual})
        print(json.dumps({'verified_model_shard': name, 'sha256': actual}), flush=True)
    return verified


def prompt_for(task):
    return ('<|im_start|>user\n' + task['prompt'] + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='First GGUF shard')
    parser.add_argument('--model-manifest', type=Path,
                        default=Path(__file__).resolve().parent / 'model_hashes.json',
                        help='All four shard sizes and SHA256 hashes, checked before inference')
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='A new directory')
    args = parser.parse_args()
    binary, model, task_file = [p.resolve(strict=True) for p in [args.binary, args.model, args.tasks]]
    model_manifest = args.model_manifest.resolve(strict=True)
    tasks = [json.loads(line) for line in task_file.read_text(encoding='utf-8').splitlines()]
    assert tasks and len({t['id'] for t in tasks}) == len(tasks)
    for task in tasks:
        a, b = task['parameters']['a'], task['parameters']['b']
        assert task['answer'] == str(a * b)
        assert task['prompt'] == f'Compute {a} \u00d7 {b}. Return only the exact integer product, with no spaces, commas, or explanation.'
        assert not task.get('assistant_prefix')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    verified_shards = verify_model(model, model_manifest)
    output.mkdir(parents=True, exist_ok=False)
    # These are the only environment variables assigned. PATH is inherited.
    os.environ['QWEN4EXP_LOOP_CONFIG'] = str(output / 'loop_config.txt')
    os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    os.environ['GGML_CUDA_DISABLE_FUSION'] = '1'
    os.environ['LLAMA_GRAPH_REUSE_DISABLE'] = '1'
    command = [str(binary), '-m', str(model), '-c', '1024', '-b', '512', '-ub', '512',
               '-ngl', '99', '--cpu-moe', '--fit', 'off', '-t', '12', '-tb', '12',
               '-fa', 'on', '--no-op-offload', '-lv', '1']
    manifest = {'task_file': str(task_file), 'tasks_sha256': sha(task_file), 'count': len(tasks),
                'binary_sha256': sha(binary), 'runner_sha256': sha(__file__),
                'runtime_files': {p.name: sha(p) for p in binary.parent.glob('*.dll')},
                'model_first_shard': str(model), 'model_first_shard_bytes': model.stat().st_size,
                'model_manifest_sha256': sha(model_manifest), 'verified_model_shards': verified_shards,
                'command': command, 'python': sys.version, 'settings': ARMS,
                'cuda_fusion': False, 'cuda_graphs': False, 'graph_reuse': False,
                'fresh_context': True, 'exact_generation': True, 'teacher_forcing': False,
                'arm_order_seed': 'paired-generation-v1',
                'note': 'Reproduction output; independent confirmation was frozen separately.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    diagnostics = (output / 'runtime.stderr.txt').open('x', encoding='utf-8')
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=diagnostics, text=True, encoding='utf-8',
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    streams = {}
    try:
        def read():
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'Harness exited with code {proc.poll()}')
                if line.startswith('{'):
                    result = json.loads(line)
                    if 'error' in result:
                        raise RuntimeError(result['error'])
                    return result
        assert read().get('ready') is True
        def request(task, config, capture=False):
            message = {'prompt': prompt_for(task), 'config': config, 'loop_schedule': [],
                       'grammar': GRAMMAR, 'max_new_tokens': 32, 'capture_state': capture}
            proc.stdin.write(json.dumps(message) + '\n')
            proc.stdin.flush()
            return read()
        neutral = CANDIDATE + [1]
        neutral[4] = 1
        with (output / 'controls.jsonl').open('x') as f:
            for task in (tasks[0], tasks[-1]):
                stock = request(task, STOCK, capture=True)
                control = request(task, neutral, capture=True)
                passed = all(stock[k] == control[k] for k in [
                    'logits_fingerprint_fnv1a64', 'generated_ids', 'state_bytes', 'state_fingerprint_fnv1a64'])
                f.write(json.dumps({'id': task['id'], 'passed': passed,
                                    'stock': stock, 'neutralized_loop': control}) + '\n')
                f.flush()
                assert passed, 'Neutralized recurrence failed identity control'
        for name, config in ARMS:
            request({'prompt': 'Compute 123 times 456. Return only the exact integer product.'}, config)
            streams[name] = (output / (name + '.jsonl')).open('x', encoding='utf-8')
        rng, correct = random.Random('paired-generation-v1'), {name: 0 for name, _ in ARMS}
        for i, task in enumerate(tasks):
            order = list(ARMS)
            rng.shuffle(order)
            for name, config in order:
                result = request(task, config)
                answer = result['generated_text'].strip()
                valid = bool(re.fullmatch(r'[1-9][0-9]*', answer))
                ok = valid and answer == task['answer']
                correct[name] += ok
                row = {'variant': name, 'config': config, 'task': task, 'result': result,
                       'answer': answer, 'correct': ok, 'valid': valid, 'schedule': [],
                       'grammar': GRAMMAR, 'max_new_tokens': 32,
                       'prompt_sha256': hashlib.sha256(prompt_for(task).encode()).hexdigest(),
                       'arm_order': [n for n, _ in order]}
                streams[name].write(json.dumps(row) + '\n')
                streams[name].flush()
            print(json.dumps({'n': i + 1, 'total': len(tasks), 'correct': correct}), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
        proc.stdin.close()
        proc.wait(timeout=60)
        diagnostics.close()


if __name__ == '__main__':
    main()
