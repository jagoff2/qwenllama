"""Strict external-control analysis after all frozen confirmation controls finish."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from confirmation27_preflight import sha, verify_runtime

ROOT = Path(__file__).resolve().parent
GRAMMAR = 'root ::= [1-9] [0-9]* "\\n"?'


def closed_prompt(task):
    return ('<|im_start|>user\n' + task['prompt'] + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def thought_prompt(task):
    return ('<|im_start|>user\n' + task['prompt'] + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n')


def load_rows(name):
    return [json.loads(s) for s in (ROOT / name).read_text().splitlines()]


def main():
    verify_runtime()
    design = json.loads((ROOT / 'confirmation27_control_design.json').read_text())
    for filename, expected in design['runtime_sha256'].items():
        assert sha(ROOT / filename) == expected, filename
    tasks = load_rows('confirmation27_heldout192.jsonl')
    expected = {t['id']: t for t in tasks}
    assert len(expected) == len(tasks) == 192
    primary = {r['task']['id']: r for r in load_rows('confirmation27_stock.jsonl')}
    candidate = {r['task']['id']: r for r in load_rows('confirmation27_norm28_31x4.jsonl')}
    assert set(primary) == set(candidate) == set(expected)
    controls = []
    for kind, prefix in [('original', 'confirmation27original'), ('fused', 'confirmation27fused'),
                         ('thought', 'confirmation27thought'), ('thought_best', 'confirmation27thoughtbest')]:
        manifest = json.loads((ROOT / (prefix + '_manifest.json')).read_text())
        assert manifest['control_design_sha256'] == sha(ROOT / 'confirmation27_control_design.json')
        assert manifest['kind'] == kind and manifest['count'] == 192
        data = load_rows(prefix + '_stock.jsonl')
        by_id = {r['task']['id']: r for r in data}
        assert len(data) == len(by_id) == 192 and set(by_id) == set(expected)
        is_thought = kind in ('thought', 'thought_best')
        budget = (design['best_short_thought_budget'] if kind == 'thought_best'
                  else design['selected_thought_budget']) if is_thought else None
        prompt_fn = thought_prompt if is_thought else closed_prompt
        differences, times, loop_times = [], [], []
        for task in tasks:
            row = by_id[task['id']]
            result = row['result']
            assert row['task'] == task
            assert row['variant'] == kind and row['config'] == [-1, -1, 1, 1, 0, 0, 0]
            assert row['grammar'] == GRAMMAR and row['max_new_tokens'] == 32
            assert row['thought_budget'] == budget
            assert row['prompt_sha256'] == hashlib.sha256(prompt_fn(task).encode()).hexdigest()
            assert result['stopped_eog'] is True
            assert len(result['generated_ids']) < 32
            assert math.isfinite(result['seconds']) and result['seconds'] > 0
            if is_thought:
                assert result['thinking_budget'] == budget and len(result['thinking_ids']) <= budget
            answer = result['generated_text'].strip()
            valid = bool(re.fullmatch(r'[1-9][0-9]*', answer))
            correct = valid and answer == str(task['parameters']['a'] * task['parameters']['b'])
            assert row['answer'] == answer and row['valid'] == valid and row['correct'] == correct
            loop_row = candidate[task['id']]
            loop_correct = loop_row['result']['generated_text'].strip() == task['answer']
            assert loop_row['correct'] == loop_correct
            differences.append(int(loop_correct) - int(correct))
            times.append(result['seconds'])
            loop_times.append(loop_row['result']['seconds'])
            if kind == 'original':
                for key in ('generated_ids', 'logits_fingerprint_fnv1a64'):
                    assert result[key] == primary[task['id']]['result'][key]
        d = np.asarray(differences)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        rng = np.random.default_rng(123456)
        draws = d[rng.integers(0, 192, size=(10000, 192))].mean(axis=1) * 100
        controls.append({'kind': kind, 'n': 192, 'thought_budget': budget,
                         'stock_correct': sum(r['correct'] for r in data),
                         'loop_correct': sum(r['correct'] for r in candidate.values()),
                         'wins': wins, 'losses': losses, 'delta_pp': float(d.mean()) * 100,
                         'mcnemar_two_sided_p': float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.0,
                         'paired_bootstrap_95ci_pp': np.quantile(draws, [.025, .975]).tolist(),
                         'stock_mean_seconds': statistics.mean(times),
                         'loop_mean_seconds': statistics.mean(loop_times),
                         'ratio_loop_to_control': statistics.mean(loop_times) / statistics.mean(times),
                         'timings_interleaved': False, 'affects_primary_success': False,
                         'prompt_differs_from_primary': is_thought,
                         'output_mean_tokens': statistics.mean(len(r['result']['generated_ids']) for r in data),
                         'thought_mean_tokens': statistics.mean(len(r['result'].get('thinking_ids', [])) for r in data),
                         'natural_thought_close': sum(r['result'].get('thinking_natural_close', False) for r in data),
                         'source_sha256': sha(ROOT / (prefix + '_stock.jsonl')),
                         'manifest_sha256': sha(ROOT / (prefix + '_manifest.json'))})
    report = {'scope': 'Prespecified secondary held-out controls; primary result unchanged',
              'original_runtime_exact_parity_cases': 192,
              'control_design_sha256': sha(ROOT / 'confirmation27_control_design.json'),
              'scorer_sha256': sha(__file__), 'controls': controls,
              'limitations': ['Thought conditions use an open thought prefix and separately decoded closing tag.',
                             'Short thought truncation can reduce answer quality and output length.',
                             'Separately run latency ratios are descriptive, not randomized paired timing.']}
    with (ROOT / 'confirmation27_secondary_score.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
