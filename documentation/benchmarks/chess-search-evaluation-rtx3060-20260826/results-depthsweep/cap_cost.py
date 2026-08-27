from __future__ import annotations

import gzip
import json
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-corrections/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
kl = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
n = len(records)
TARGET_MEAN = 600.0
flat = statistics.fmean([row[budgets.index(600)] for row in kl])


def oracle(menu: list[int]) -> float:
    indices = [budgets.index(v) for v in menu]
    best = None
    for step in range(4000):
        multiplier = 10.0 ** (-7.0 + step * 0.002)
        chosen = [min(indices, key=lambda i: row[i] + multiplier * budgets[i]) for row in kl]
        mean_visits = statistics.fmean([budgets[i] for i in chosen])
        if best is None or abs(mean_visits - TARGET_MEAN) < abs(best[0] - TARGET_MEAN):
            best = (mean_visits, statistics.fmean([kl[p][chosen[p]] for p in range(n)]))
    return best[1]


full = oracle(budgets)
print('WHAT DOES CAPPING THE BUDGET RANGE COST?   base budget 600, mean held at 600')
print(f'   flat 600 everywhere                    KL {flat:.4f}   (0% of the gain)')
rows = [
    ('floor 0.17x, ceiling 2x   (100-1200)', [b for b in budgets if 100 <= b <= 1200]),
    ('floor 0.17x, ceiling 4x   (100-2400)', [b for b in budgets if 100 <= b <= 2400]),
    ('floor 0.17x, ceiling 8x   (100-5000)', [b for b in budgets if 100 <= b <= 5000]),
    ('floor 0.33x, ceiling 2x   (200-1200)', [b for b in budgets if 200 <= b <= 1200]),
    ('floor 0.17x, no ceiling   (100-10000)', budgets),
]
for label, menu in rows:
    value = oracle(menu)
    captured = (flat - value) / (flat - full) * 100.0
    print(f'   {label:38s} KL {value:.4f}   captures {captured:5.1f}% of the uncapped gain')
