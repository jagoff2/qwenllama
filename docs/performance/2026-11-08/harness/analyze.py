import bisect
import collections
import csv
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
summary = {}
for directory in sorted(root.iterdir()):
    if not directory.is_dir(): continue
    rows = []
    for file in sorted(directory.glob('result-*.json')):
        r = json.loads(file.read_text())
        rows.append({k:v for k,v in r.items() if k != 'tokens'})
    if not rows: continue
    summary[directory.name] = {'runs': rows, 'by_threads': {}}
    for threads in sorted({r['threads'] for r in rows}):
        values = [r['tps'] for r in rows if r['threads'] == threads]
        summary[directory.name]['by_threads'][threads] = {
            'count':len(values), 'mean_tps':statistics.mean(values), 'min_tps':min(values),
            'max_tps':max(values), 'stdev_tps':statistics.stdev(values) if len(values)>1 else None}
    if len(rows) > 1 and 'variant' in rows[0]:
        summary[directory.name]['by_case'] = {f"{r['variant']}-t{r['threads']}-{i}":r['tps'] for i,r in enumerate(rows)}
    if not (directory / 'handoffs.csv').exists(): continue
    all_runs = [json.loads(f.read_text()) for f in sorted(directory.glob('result-*.json'))]
    tokens = [t for run in all_runs for t in run['tokens'][64:]]
    starts = [t['start_us'] for t in tokens]
    ends = [t['end_us'] for t in tokens]
    labels = [run['label'] for run in all_runs for t in run['tokens'][64:]]
    count = all_runs[0]['measured_tokens']
    def inside(row):
        at = bisect.bisect_right(starts, int(row['start_us']))-1
        return labels[at] if at >= 0 and int(row['end_us']) <= ends[at] else None
    groups = collections.defaultdict(collections.Counter)
    split_us = collections.Counter()
    with (directory / 'handoffs.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            label = inside(row)
            if not label: continue
            g = groups[label + '/' + row['backend']]
            g['splits'] += 1
            g['backend_calls'] += int(row.get('backend_calls', 1))
            for key in ['source_wait_us','destination_wait_us','copy_us','async_try_us']:
                g[key] += int(row[key])
            g['prewait_us'] += int(row.get('prewait_us', 0))
            g['fences_us'] += int(row['fences_us'])-int(row['start_us'])
            g['inputs_total_us'] += int(row['copies_us'])-int(row['fences_us'])
            g['compute_or_submit_us'] += int(row['compute_us'])-int(row['copies_us'])
            g['event_record_us'] += int(row['end_us'])-int(row['compute_us'])
            split_us[label] += int(row['end_us'])-int(row['start_us'])
    profile = {'per_token':{name:{k:v/count for k,v in g.items()} for name,g in groups.items()},
               'outside_splits_us_per_token': {r['label']:(r['measured_us']-split_us[r['label']])/count for r in all_runs}}
    cpu = directory / 'cpu-ops.csv'
    if cpu.exists():
        totals = collections.defaultdict(collections.Counter)
        with cpu.open(newline='') as file:
            for r in csv.DictReader(file):
                label = inside(r)
                if not label: continue
                totals[label]['all_ops_us'] += int(r['end_us'])-int(r['start_us'])
                if 'MUL_MAT_ID' in r['op']:
                    totals[label]['moe_us'] += int(r['end_us'])-int(r['start_us'])
                    totals[label]['moe_ops'] += 1
                    for key in ['logical_bytes', 'routes', 'misses']:
                        if key in r: totals[label][key] += int(r[key])
        profile['cpu_per_token'] = {label:{k:v/count for k,v in vals.items()} for label,vals in totals.items()}
    summary[directory.name]['profile'] = profile
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
for name, s in summary.items():
    print(name, json.dumps(s['by_threads']))
    if 'profile' in s: print(json.dumps(s['profile']))
