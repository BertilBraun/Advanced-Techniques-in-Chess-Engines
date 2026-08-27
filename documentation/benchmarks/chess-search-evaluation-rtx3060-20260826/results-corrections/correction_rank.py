from __future__ import annotations

import gzip
import json
import random
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-corrections/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
kl = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
n = len(records)
TARGET_MEAN = 600.0

best = None
for step in range(4000):
    multiplier = 10.0 ** (-7.0 + step * 0.002)
    chosen = [min(range(len(budgets)), key=lambda i: row[i] + multiplier * budgets[i]) for row in kl]
    mean_visits = statistics.fmean([budgets[i] for i in chosen])
    if best is None or abs(mean_visits - TARGET_MEAN) < abs(best[0] - TARGET_MEAN):
        best = (mean_visits, statistics.fmean([kl[p][chosen[p]] for p in range(n)]), chosen)
_, oracle_kl, oracle_choice = best
oracle_budgets = sorted([budgets[i] for i in oracle_choice], reverse=True)
flat600 = statistics.fmean([row[budgets.index(600)] for row in kl])

i200, i600 = budgets.index(200), budgets.index(600)
generator = random.Random(20260827)

signals = {
    'true Lagrangian optimum (ceiling)': None,
    'true remaining error KL(600)': [row[i600] for row in kl],
    'policy_correction (already computed, free)': [r['policy_correction'] for r in records],
    'search_correction_target (max of the two)': [r['search_correction_target'] for r in records],
    'value_correction': [r['value_correction'] for r in records],
    'true benefit KL(200) - KL(600)': [row[i200] - row[i600] for row in kl],
    'random ordering (control)': [generator.random() for _ in range(n)],
}

print('RANKING POWER FOR BUDGET ALLOCATION, mean 600 visits')
print(f'   flat 600 = KL {flat600:.4f}   oracle = KL {oracle_kl:.4f}')
print()
for name, signal in signals.items():
    if signal is None:
        value = oracle_kl
    else:
        order = sorted(range(n), key=lambda i: -signal[i])
        assigned = [0] * n
        for rank, position in enumerate(order):
            assigned[position] = oracle_budgets[rank]
        value = statistics.fmean([kl[p][budgets.index(assigned[p])] for p in range(n)])
    captured = (flat600 - value) / (flat600 - oracle_kl) * 100.0
    print(f'   {name:44s} KL {value:.4f}   captures {captured:6.1f}%')

print()
print('Correlation of the free signals with true remaining error:')


def spearman(a: list[float], b: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out

    ra, rb = rank(a), rank(b)
    mean_a, mean_b = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    va = sum((x - mean_a) ** 2 for x in ra)
    vb = sum((y - mean_b) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5


truth = [row[i600] for row in kl]
for name in ('policy_correction', 'value_correction', 'search_correction_target'):
    print(f'   {name:32s} rho = {spearman([r[name] for r in records], truth):+.3f}')
