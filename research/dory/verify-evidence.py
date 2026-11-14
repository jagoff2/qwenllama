"""Check the published numerical claims without a model or GPU."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

g = read(E / 'graphs-summary.json')
m = read(E / 'memory-summary.json')
p = read(E / 'provenance.json')
b = g['benchmark']
assert b['repetitions'] == 3 and b['tokens_each'] == [256] * 3
baseline = statistics.median(b['baseline_tokens_per_second'])
optimized = statistics.median(b['optimized_tokens_per_second'])
assert baseline == b['baseline_median'] and optimized == b['optimized_median']
assert abs(optimized / baseline - b['speedup_ratio']) < 1e-12
assert all(b['same_generated_tokens'])
assert g['full_vocab']['final_exact'] and g['full_vocab']['final_max_abs'] == 0
assert all(all(v.values()) for v in g['reuse_capture_equivalence'].values())
assert all(g['server']['stock_neutral_tokens'].values())
assert m['passed'] and not m['oom'] and m['planner_failures'] == 0
assert m['decode_median_tokens_per_second'] == statistics.median(m['decode_tokens_per_second'])
assert m['prompt_lengths'] == [23486, 33, 32768, 23486]
assert read(E / 'allocation-failure.json')['fixed_exit_code'] == 0

final = read(E / 'full-vocab/final_all/results.json')
assert final['stock']['hashes'] == final['neutral']['hashes'] == final['wide_neutral']['hashes']
reference = read(E / 'full-vocab/baseline/results.json')
for mode in ('reuse_only', 'capture_only', 'reuse_capture'):
    result = read(E / 'full-vocab' / mode / 'results.json')
    for config in reference:
        assert result[config]['hashes'] == reference[config]['hashes'], (mode, config)

base_run = read(E / 'server/benchmark_baseline_loop_c261888.json')
opt_run = read(E / 'server/benchmark_optimized_loop_c261888.json')
for a, z in zip(base_run['tests'], opt_run['tests']):
    if a['name'] in ('bench_0', 'bench_1', 'bench_2'):
        assert a['generated_token_ids'] == z['generated_token_ids']

for name, expected in p.get('final_source_lf_sha256', {}).items():
    value = (ROOT.parents[1] / name).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(value).hexdigest() == expected, name

manifest = E / 'artifact-sha256.json'
if manifest.exists():
    for name, expected in read(manifest).items():
        path = ROOT / name
        value = path.read_bytes()
        if path.suffix != '.png':
            value = value.replace(b'\r\n', b'\n')
        assert hashlib.sha256(value).hexdigest() == expected, name

print(json.dumps({'passed': True, 'decode_speedup': optimized / baseline,
                  'memory_decode_median': m['decode_median_tokens_per_second'],
                  'longest_tested_prompt': max(m['prompt_lengths'])}, indent=2))
