"""Strict offline scorer for the frozen 192-case confirmation.

Requires NumPy for the frozen 10,000-draw bootstrap. Never starts inference or
imports experiment runners. Use --check-only to validate without writing a file.
Development validation requires --dev and an explicit --expected-count.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
STOCK = [-1, -1, 1, 1, 0, 0, 0]
GRAMMAR = 'root ::= [1-9] [0-9]* "\\n"?'
BUDGET = 32
ORDER_DESCRIPTION = 'independently shuffled per task with fixed seed paired-generation-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON number')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def parse(text):
    value = json.loads(text, object_pairs_hook=object_without_duplicates)
    finite_tree(value)
    return value


def load(path):
    return parse(Path(path).read_text(encoding='utf-8'))


def rows(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    require(lines and all(line.strip() for line in lines), f'empty file/row: {path}')
    return [parse(line) for line in lines]


def school_product(a, b):
    digits = [0] * (len(str(a)) + len(str(b)) + 1)
    for i, x in enumerate(map(int, str(a)[::-1])):
        for j, y in enumerate(map(int, str(b)[::-1])):
            digits[i + j] += sum(x for _ in range(y))
    for i in range(len(digits) - 1):
        digits[i + 1] += digits[i] // 10
        digits[i] %= 10
    return int(''.join(map(str, digits[::-1])))


def generated_task(split, index):
    seed = {'dev': 41017, 'heldout': 99017}[split]
    rng = random.Random(': Follow the experimental seed :'.join(
        map(str, (seed, 'integer_product', 5, index))))
    values = []
    while len(values) < 2:
        value = rng.randrange(10000, 100000)
        if value % 10 in (1, 3, 7, 9):
            values.append(value)
    a, b = values
    answer = str(school_product(a, b))
    require(answer == str(a * b), 'independent arithmetic oracle mismatch')
    return {'id': f'{split}-integer_product-n5-{index:04}', 'split': split,
            'family': 'integer_product', 'difficulty': '5', 'index': index,
            'parameters': {'a': a, 'b': b, 'digits': 5},
            'prompt': f'Compute {a} \u00d7 {b}. Return only the exact integer product, with no spaces, commas, or explanation.',
            'answer': answer, 'answer_value': answer}


def prompt(task):
    return ('<|im_start|>user\n' + task['prompt'] + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def validate_arm(data, tasks, setting, orders=None):
    require(len(data) == len(tasks), f"{setting['name']}: wrong row count")
    ids = [row['task']['id'] for row in data]
    expected_ids = [task['id'] for task in tasks]
    require(len(set(ids)) == len(ids), f"{setting['name']}: duplicate ID")
    require(set(ids) == set(expected_ids), f"{setting['name']}: wrong ID set")
    if orders is not None:
        require(ids == expected_ids, 'primary output order differs from frozen task order')
    indexed = {row['task']['id']: row for row in data}
    checked = []
    for i, task in enumerate(tasks):
        row = indexed[task['id']]
        finite_tree(row)
        require(row['task'] == task, f"task mismatch: {task['id']}")
        require(row['variant'] == setting['name'], 'wrong variant')
        require(row['config'] == setting['config'], 'wrong recurrence config')
        require(row.get('schedule', []) == [], 'unexpected prediction schedule')
        require(row.get('thought_budget') is None, 'unexpected thought budget')
        require(row['grammar'] == GRAMMAR and type(row['max_new_tokens']) is int
                and row['max_new_tokens'] == BUDGET, 'grammar/budget mismatch')
        require(row['prompt_sha256'] == hashlib.sha256(prompt(task).encode()).hexdigest(),
                'prompt hash mismatch')
        if orders is not None:
            require(row['arm_order'] == orders[i], 'interleaved arm order mismatch')
            require(type(row.get('valid')) is bool, 'missing primary validity flag')
        result = row['result']
        require(not any(key in result for key in ('error', 'teacher_forced', 'target_tokens',
                'target_log_probability', 'thinking_ids', 'thinking_text', 'loop_schedule')),
                'unexpected inference mode/result')
        text, token_ids = result['generated_text'], result['generated_ids']
        require(isinstance(text, str) and isinstance(token_ids, list), 'invalid generation fields')
        require(all(type(token) is int and token >= 0 for token in token_ids), 'invalid generated token IDs')
        require(0 <= len(token_ids) <= BUDGET, 'invalid generated token count')
        require(type(result['stopped_eog']) is bool, 'missing EOG completion flag')
        completed = result['stopped_eog']
        require(completed, f"missing EOG completion: {task['id']}; no confirmation score produced")
        require(len(token_ids) < BUDGET, 'inconsistent EOG/budget metadata')
        require(type(result['tokens']) is int and 0 < result['tokens'] <= 512,
                'invalid prompt token count')
        require(type(result['seconds']) in (int, float) and result['seconds'] > 0,
                'invalid latency')
        require(set(result['logits']) == set('ABCD') and
                all(type(x) in (int, float) for x in result['logits'].values()),
                'missing finite first-logit diagnostics')
        fingerprint = result['logits_fingerprint_fnv1a64']
        require(isinstance(fingerprint, str) and re.fullmatch(r'[0-9]+', fingerprint)
                and int(fingerprint) < 2 ** 64, 'invalid first-logit fingerprint')
        answer = text.strip()
        valid = bool(re.fullmatch(r'[1-9][0-9]*', answer))
        correct = valid and answer == task['answer']
        require(row['answer'] == answer, 'stored answer differs from generated text')
        require(type(row['correct']) is bool and row['correct'] == correct,
                'stored correctness differs from complete-generation oracle score')
        if 'valid' in row:
            require(type(row['valid']) is bool and row['valid'] == valid, 'stored validity mismatch')
        checked.append({'correct': correct, 'valid': valid, 'completed': completed,
                        'seconds': result['seconds'], 'tokens': result['tokens']})
    return checked


def compare(baseline, candidate, interleaved):
    require(len(baseline) == len(candidate) and baseline, 'invalid comparison size')
    d = np.array([int(b['correct']) - int(a['correct'])
                  for a, b in zip(baseline, candidate)], dtype=np.int8)
    wins, losses, n = int((d > 0).sum()), int((d < 0).sum()), len(d)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k)
            for k in range(min(wins, losses) + 1)) / (2 ** discordant))
    rng = np.random.default_rng(123456)
    bootstrap = d[rng.integers(0, n, size=(10000, n))].mean(axis=1) * 100
    def arm_summary(arm):
        return {'correct': sum(x['correct'] for x in arm),
                'accuracy_percent': 100 * sum(x['correct'] for x in arm) / n,
                'valid': sum(x['valid'] for x in arm),
                'eog_completed': sum(x['completed'] for x in arm),
                'budget_exhausted': sum(not x['completed'] for x in arm),
                'mean_seconds': statistics.mean(x['seconds'] for x in arm),
                'median_seconds': statistics.median(x['seconds'] for x in arm)}
    a, b = arm_summary(baseline), arm_summary(candidate)
    return {'n': n, 'baseline': a, 'candidate': b, 'wins': wins, 'losses': losses,
            'both_correct': sum(a['correct'] and b['correct'] for a, b in zip(baseline, candidate)),
            'delta_pp': 100 * float(d.mean()), 'mcnemar_two_sided_p': p,
            'paired_bootstrap_95ci_pp': np.quantile(bootstrap, [.025, .975]).tolist(),
            'bootstrap_draws': 10000, 'bootstrap_seed': 123456,
            'timing': {'interleaved': interleaved,
                       'ratio_of_means': b['mean_seconds'] / a['mean_seconds'],
                       'mean_paired_delta_seconds': statistics.mean(
                           y['seconds'] - x['seconds'] for x, y in zip(baseline, candidate)),
                       'interpretation': 'randomized within-task order' if interleaved else
                           'descriptive separate runs; no interleaved timing claim',
                       'scope': 'context creation plus inference; excludes model load and teardown'}}


def score(args):
    design, tasks, settings = load(args.design), rows(args.tasks), load(args.settings)
    require(len(settings) == 2 and settings[0] == {'name': 'stock', 'config': STOCK},
            'exactly stock and one primary candidate are required')
    require(settings[1]['name'] != 'stock' and set(settings[1]) == {'name', 'config'},
            'only a static primary candidate is supported')
    require(isinstance(settings[1]['config'], list) and 7 <= len(settings[1]['config']) <= 11,
            'invalid candidate config')
    expected_count = args.expected_count if args.dev else 192
    require((args.dev and type(expected_count) is int and expected_count > 0) or
            (not args.dev and args.expected_count is None), '--expected-count is required only in --dev mode')
    require(len(tasks) == expected_count == design['count'], 'wrong frozen task count')
    split = 'dev' if args.dev else 'heldout'
    require(design['split'] == split, 'wrong design split')
    indices = [task['index'] for task in tasks]
    require(all(type(index) is int and index >= 0 for index in indices), 'invalid task index')
    if args.dev:
        require('indices' in design and indices == list(range(design['indices'][0], design['indices'][1] + 1)),
                'dev design must declare the exact index interval')
    else:
        require(indices == list(range(192)), 'wrong held-out index sequence')
    require(tasks == [generated_task(split, index) for index in indices],
            'tasks differ from independent deterministic generator/oracle')
    require(len({tuple(sorted((t['parameters']['a'], t['parameters']['b']))) for t in tasks})
            == expected_count, 'duplicate operand pair')
    manifest = load(args.manifest)
    for field, path in [('tasks_sha256', args.tasks), ('settings_sha256', args.settings)]:
        require(manifest[field] == sha(path), f'run manifest mismatch: {field}')
        frozen_key = 'file_sha256' if args.dev and field == 'tasks_sha256' else field
        if frozen_key in design:
            require(design[frozen_key] == sha(path), f'frozen design mismatch: {field}')
        else:
            require(args.dev, f'missing frozen hash: {field}')
    require(manifest['count'] == expected_count and manifest['settings'] == settings
            and manifest['split'] == [split], 'run manifest population/settings mismatch')
    for key, value in {'exact_generation': True, 'teacher_forcing': False, 'fresh_context': True,
                       'cuda_fusion': False, 'cuda_graphs': False, 'graph_reuse': False}.items():
        require(manifest.get(key) is value, f'run protocol mismatch: {key}')
    require(manifest['arm_order'] == ORDER_DESCRIPTION, 'wrong randomization protocol')
    frozen = {'runtime_verified': False}
    if not args.dev:
        require(design['success_rule'] == {'minimum_absolute_gain_pp': 20,
                'maximum_exact_two_sided_mcnemar_p': .01}, 'unexpected frozen success rule')
        for key in ('no_optional_stopping', 'no_parameter_changes_after_inference', 'answer_length_not_provided'):
            require(design.get(key) is True, f'frozen protocol flag missing: {key}')
        require(design.get('model_answer_filtering') is False, 'answer filtering is forbidden')
        require(manifest.get('heldout_design_sha256') == sha(args.design), 'run used a different design')
        for key in ('binary_sha256', 'llama_sha256', 'runner_sha256'):
            require(manifest[key] == design[key], f'run artifact differs from frozen design: {key}')
        require(sha(args.runtime_manifest) == design['runtime_manifest_sha256'], 'runtime manifest hash mismatch')
        runtime = load(args.runtime_manifest)
        for name, digest in runtime.items():
            require(Path(name).name == name, 'runtime manifest path must be a basename')
            require(sha(args.runtime_dir / name) == digest, f'frozen runtime drift: {name}')
        for name, field in [('latent-eval-schedule.exe', 'binary_sha256'), ('llama.dll', 'llama_sha256'),
                            ('paired_generation.py', 'runner_sha256'), ('model_hashes.json', 'model_manifest_sha256')]:
            require(sha(args.runtime_dir / name) == design[field], f'frozen artifact mismatch: {name}')
        helpers = load(args.helper_manifest)
        require(helpers['design_sha256'] == sha(args.design) and
                helpers['runtime_manifest_sha256'] == sha(args.runtime_manifest),
                'supplemental helper manifest binds a different design/runtime')
        for name, digest in helpers['files'].items():
            require(Path(name).name == name, 'helper manifest path must be a basename')
            require(sha(args.helper_dir / name) == digest, f'frozen helper drift: {name}')
            require(sha(args.design.parent / name) == digest, f'live helper drift: {name}')
        frozen = {'runtime_verified': True, 'runtime_files_verified': len(runtime),
                  'runtime_manifest_sha256': sha(args.runtime_manifest),
                  'helper_files_verified': len(helpers['files']),
                  'helper_manifest_sha256': sha(args.helper_manifest),
                  'model_manifest_verified': True, 'model_weight_files_rehashed': False}
    elif 'selected_config' in design:
        require(settings[1]['config'] == design['selected_config'], 'dev selected config mismatch')
    rng, orders = random.Random('paired-generation-v1'), []
    for _ in tasks:
        order = [s['name'] for s in settings]
        rng.shuffle(order)
        orders.append(order)
    baseline = validate_arm(rows(args.baseline), tasks, settings[0], orders)
    candidate = validate_arm(rows(args.candidate), tasks, settings[1], orders)
    require(all(a['tokens'] == b['tokens'] for a, b in zip(baseline, candidate)), 'paired prompt token counts differ')
    primary = compare(baseline, candidate, True)
    primary['baseline_name'], primary['candidate_name'] = [s['name'] for s in settings]
    primary['timing']['first_arm_counts'] = {s['name']: sum(order[0] == s['name'] for order in orders)
                                            for s in settings}
    passed = (100 * (primary['wins'] - primary['losses']) >= 20 * expected_count
              and primary['mcnemar_two_sided_p'] < .01)
    result = {'analysis': 'development validation' if args.dev else 'frozen held-out confirmation',
              'confirmation_success': None if args.dev else passed,
              'success_rule': {'delta_pp': '>=20', 'exact_two_sided_mcnemar_p': '<0.01'},
              'primary': primary, 'frozen_provenance': frozen,
              'numpy_version': np.__version__,
              'limitations': ['Snapshot hashes verify recorded artifacts, not historical execution.',
                             'Missing EOG completion is a validation failure, not a changed accuracy rule.'],
              'hashes': {name: {'path': str(path.resolve()), 'sha256': sha(path)} for name, path in
                         [('design', args.design), ('tasks', args.tasks), ('settings', args.settings),
                          ('run_manifest', args.manifest), ('baseline', args.baseline),
                          ('candidate', args.candidate), ('scorer', Path(__file__))]}}
    if args.original_fused:
        fused_rows = rows(args.original_fused)
        variants = {row['variant'] for row in fused_rows}
        require(len(variants) == 1 and variants <= {'stock', 'fused'},
                'original fused rows must use stock or fused consistently')
        fused = validate_arm(fused_rows, tasks, {'name': next(iter(variants)), 'config': STOCK})
        require(all(a['tokens'] == b['tokens'] for a, b in zip(fused, baseline)), 'fused prompt token counts differ')
        result['secondary_original_fused'] = compare(fused, candidate, False)
        result['secondary_original_fused']['baseline_name'] = 'original_fused'
        result['secondary_original_fused']['candidate_name'] = settings[1]['name']
        result['secondary_original_fused']['affects_primary_success'] = False
        result['secondary_original_fused']['runtime_provenance_verified'] = False
        result['hashes']['original_fused'] = {'path': str(args.original_fused.resolve()),
                                              'sha256': sha(args.original_fused)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=ROOT / 'confirmation27_design.json')
    parser.add_argument('--tasks', type=Path, default=ROOT / 'confirmation27_heldout192.jsonl')
    parser.add_argument('--settings', type=Path, default=ROOT / 'confirmation27_settings.json')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, default=ROOT / 'confirmation27_bin')
    parser.add_argument('--runtime-manifest', type=Path, default=ROOT / 'confirmation27_runtime_manifest.json')
    parser.add_argument('--helper-dir', type=Path, default=ROOT / 'confirmation27_helpers')
    parser.add_argument('--helper-manifest', type=Path, default=ROOT / 'confirmation27_helper_manifest.json')
    parser.add_argument('--original-fused', type=Path)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--expected-count', type=int)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--output', type=Path)
    output.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        result = score(args)
        text = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text + '\n')
            print(f'Wrote verified score: {args.output}')
        else:
            print(text)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'Confirmation validation failed: {error}\n')


if __name__ == '__main__':
    main()
