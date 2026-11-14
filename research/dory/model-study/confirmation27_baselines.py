"""Prespecified stock-only controls on every frozen confirmation case."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from direct import DirectModel, STOCK
from integer_tasks import prompt_for as closed_prompt
from integer_eval import GRAMMAR
from thinking_sweep import prompt_for as thought_prompt
from confirmation27_preflight import verify_runtime, sha

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['original', 'fused', 'thought', 'thought_best'], required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--control-design', required=True)
    args = parser.parse_args()
    design = verify_runtime()
    control_design = json.loads(Path(args.control_design).read_text())
    assert control_design['primary_design_sha256'] == sha(ROOT / 'confirmation27_design.json')
    for filename, expected in control_design['runtime_sha256'].items():
        assert sha(ROOT / filename) == expected, filename
    tasks = [json.loads(s) for s in (ROOT / 'confirmation27_heldout192.jsonl').read_text().splitlines()]
    assert len(tasks) == design['count'] == 192
    assert len({t['id'] for t in tasks}) == 192
    is_thought = args.kind in ('thought', 'thought_best')
    budget_key = 'best_short_thought_budget' if args.kind == 'thought_best' else 'selected_thought_budget'
    budget = control_design[budget_key] if is_thought else None
    binary = (ROOT / 'confirmation27_bin' / 'latent-eval-thinking.exe' if is_thought
              else ROOT / 'stock_bin' / 'latent-eval-v3.exe')
    prompt_fn = thought_prompt if is_thought else closed_prompt
    if args.kind == 'fused':
        os.environ.pop('GGML_CUDA_DISABLE_FUSION', None)
    else:
        os.environ['GGML_CUDA_DISABLE_FUSION'] = '1'
    manifest = {'kind': args.kind, 'thought_budget': budget, 'count': 192,
                'tasks_sha256': design['tasks_sha256'], 'binary_sha256': sha(binary),
                'llama_sha256': sha(binary.parent / 'llama.dll'),
                'runner_sha256': sha(__file__), 'control_design_sha256': sha(args.control_design),
                'fusion': args.kind == 'fused', 'graphs': False, 'graph_reuse': False,
                'fresh_context': True, 'warmup': 'one unscored easy product',
                'timing_comparison': 'separate run; not randomized interleaved with primary arms',
                'no_candidate_changes': True}
    with (ROOT / (args.prefix + '_manifest.json')).open('x') as f:
        json.dump(manifest, f, indent=2)
    model = DirectModel(args.prefix, binary=binary)
    try:
        def request(task):
            req = {'prompt': prompt_fn(task), 'config': STOCK,
                   'grammar': GRAMMAR, 'max_new_tokens': 32}
            if budget is not None:
                req['thinking_tokens'] = budget
            return req
        model.request(request({'prompt': 'Compute 123 times 456. Return only the exact integer product.'}))
        reference = None
        if args.kind == 'original':
            source = [json.loads(s) for s in (ROOT / 'confirmation27_stock.jsonl').read_text().splitlines()]
            reference = {r['task']['id']: r for r in source}
            assert len(source) == len(reference) == 192
            assert set(reference) == {t['id'] for t in tasks}
        correct = 0
        with (ROOT / (args.prefix + '_stock.jsonl')).open('x') as f:
            for i, task in enumerate(tasks):
                req = request(task)
                result = model.request(req)
                answer = result['generated_text'].strip()
                valid = bool(re.fullmatch(r'[1-9][0-9]*', answer))
                ok = valid and answer == task['answer']
                correct += ok
                if reference is not None:
                    expected = reference[task['id']]['result']
                    for key in ('generated_ids', 'logits_fingerprint_fnv1a64'):
                        assert result[key] == expected[key], (task['id'], key)
                row = {'variant': args.kind, 'config': STOCK, 'task': task, 'result': result,
                       'answer': answer, 'correct': ok, 'valid': valid,
                       'grammar': GRAMMAR, 'max_new_tokens': 32, 'thought_budget': budget,
                       'prompt_sha256': hashlib.sha256(req['prompt'].encode()).hexdigest()}
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(json.dumps({'control': args.kind, 'n': i + 1, 'total': 192,
                                  'correct': correct}), flush=True)
        if reference is not None:
            print(json.dumps({'original_generation_parity': 'exact', 'n': 192}), flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()
