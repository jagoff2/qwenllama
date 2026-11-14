"""Render the frozen primary accuracy and latency result from its JSON report."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
primary = json.loads((root / 'confirmation27_score.json').read_text())['primary']
labels = ['Stock', 'Four-pass recurrence']
colors = ['#66758a', '#126f69']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout='constrained')
metrics = [('accuracy_percent', 'Exact-answer accuracy (%)', 45),
           ('mean_seconds', 'Mean context + inference time (s)', 3.0)]
for ax, (key, title, upper) in zip(axes, metrics):
    values = [primary[arm][key] for arm in ('baseline', 'candidate')]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title(title, loc='left', fontweight='bold', pad=15)
    ax.set_ylim(0, upper)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    for index, (bar, value) in enumerate(zip(bars, values)):
        arm = ('baseline', 'candidate')[index]
        label = f'{value:.2f}%\n{primary[arm]["correct"]}/192' if key == 'accuracy_percent' else f'{value:.3f} s'
        ax.text(bar.get_x() + bar.get_width()/2, value + upper*.025, label, ha='center', va='bottom', fontsize=11)
fig.suptitle('Frozen paired evaluation: 192 held-out five-digit products', fontsize=14, fontweight='bold')
fig.savefig(root / 'primary-results.svg', metadata={'Date': None})
fig.savefig(root / 'primary-results.png', dpi=180)
plt.close(fig)
