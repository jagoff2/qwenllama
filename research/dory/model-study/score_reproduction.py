"""Rescore complete portable reproduction outputs, without historical claims."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from reproduce_recurrence import STOCK, CANDIDATE, GRAMMAR, prompt_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--stock', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    read = lambda path: [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    tasks = read(a.tasks)
    expected = {t['id']: t for t in tasks}
    assert len(expected) == len(tasks) > 0
    arms, timings = [], []
    for path, config in [(a.stock, STOCK), (a.candidate, CANDIDATE)]:
        data = read(path)
        indexed = {r['task']['id']: r for r in data}
        assert len(data) == len(indexed) == len(tasks) and set(indexed) == set(expected)
        correct, seconds = [], []
        for task in tasks:
            r = indexed[task['id']]
            assert r['task'] == task and r['config'] == config
            assert r['prompt_sha256'] == hashlib.sha256(prompt_for(task).encode()).hexdigest()
            assert r['grammar'] == GRAMMAR and r['max_new_tokens'] == 32
            result = r['result']
            assert result['stopped_eog'] is True
            assert math.isfinite(result['seconds']) and result['seconds'] > 0
            target = str(task['parameters']['a'] * task['parameters']['b'])
            assert task['answer'] == target
            ok = result['generated_text'].strip() == target
            assert r['correct'] == ok
            correct.append(int(ok))
            seconds.append(result['seconds'])
        arms.append(np.asarray(correct, dtype=np.int8))
        timings.append(statistics.mean(seconds))
    d = arms[1] - arms[0]
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    discordant = wins + losses
    exact_p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    rng = np.random.default_rng(123456)
    boot = d[rng.integers(0, len(tasks), size=(10000, len(tasks)))].mean(axis=1) * 100
    report = {'analysis': 'Complete reproduction rescore; does not verify historical frozen execution',
              'n': len(tasks), 'stock_correct': int(arms[0].sum()), 'candidate_correct': int(arms[1].sum()),
              'wins': wins, 'losses': losses, 'delta_pp': float(d.mean()) * 100,
              'mcnemar_two_sided_p': exact_p, 'paired_bootstrap_95ci_pp': np.quantile(boot, [.025, .975]).tolist(),
              'stock_mean_seconds': timings[0], 'candidate_mean_seconds': timings[1],
              'latency_ratio': timings[1] / timings[0],
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [a.tasks, a.stock, a.candidate]}}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
