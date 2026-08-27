from __future__ import annotations

import gzip
import json
import random
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-oracle/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
kl = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
n = len(records)
TARGET_MEAN = 600.0


def oracle(menu: list[int]) -> tuple[float, float, list[int]]:
    """Lagrangian sweep restricted to a budget menu, returned at the multiplier closest to the target mean."""
    indices = [budgets.index(v) for v in menu]
    best = None
    for step in range(4000):
        multiplier = 10.0 ** (-7.0 + step * 0.002)
        chosen = []
        for row in kl:
            chosen.append(min(indices, key=lambda i: row[i] + multiplier * budgets[i]))
        mean_visits = statistics.fmean([budgets[i] for i in chosen])
        if best is None or abs(mean_visits - TARGET_MEAN) < abs(best[0] - TARGET_MEAN):
            best = (mean_visits, statistics.fmean([kl[p][chosen[p]] for p in range(n)]), chosen)
    return best


print('1. WHAT DOES RESTRICTING THE BUDGET MENU COST?  (all at a mean of ~600 visits)')
flat600 = statistics.fmean([row[budgets.index(600)] for row in kl])
print(f'   flat 600 everywhere                      KL {flat600:.4f}')
for menu in ([300, 600, 1200], [150 if 150 in budgets else 100, 600, 2400], [100, 600, 3200], budgets):
    mean_visits, value, _ = oracle(list(menu))
    label = 'full menu (14 levels)' if len(menu) == len(budgets) else f'{len(menu)}-level menu {menu}'
    print(f'   {label:40s} mean {mean_visits:6.1f}  KL {value:.4f}')
print()

_, _, oracle_choice = oracle(budgets)
oracle_budgets = sorted([budgets[i] for i in oracle_choice], reverse=True)
oracle_kl = statistics.fmean([kl[p][oracle_choice[p]] for p in range(n)])

print('2. HOW SKEWED IS THE ORACLE ALLOCATION?')
from collections import Counter

histogram = Counter(oracle_budgets)
print(f'   {dict(sorted(histogram.items()))}')
print()

print('3. WHICH SIGNAL SHOULD THE HEAD REGRESS AGAINST?')
print("   Each row takes the oracle's own multiset of budgets and hands them out in order of that signal,")
print('   so the budget distribution is identical and only the ordering differs.')
i200, i300, i600 = (budgets.index(v) for v in (200, 300, 600))
generator = random.Random(20260827)
signals = {
    'true Lagrangian optimum (the oracle)': None,
    'true benefit, KL(300) - KL(600)': [row[i300] - row[i600] for row in kl],
    'true remaining error, KL(600)': [row[i600] for row in kl],
    'true benefit, KL(200) - KL(600)': [row[i200] - row[i600] for row in kl],
    'observable top_visit_share at 200 (low first)': [-r['budgets'][i200]['top_visit_share'] for r in records],
    'observable top_two_margin at 200 (low first)': [-r['budgets'][i200]['top_two_margin'] for r in records],
    'random ordering (control)': [generator.random() for _ in range(n)],
}
rows = []
for name, signal in signals.items():
    if signal is None:
        rows.append((name, oracle_kl))
        continue
    order = sorted(range(n), key=lambda i: -signal[i])
    assigned = [0] * n
    for rank, position in enumerate(order):
        assigned[position] = oracle_budgets[rank]
    rows.append((name, statistics.fmean([kl[p][budgets.index(assigned[p])] for p in range(n)])))

for name, value in rows:
    captured = (flat600 - value) / (flat600 - oracle_kl) * 100.0
    print(f'   {name:46s} KL {value:.4f}   captures {captured:6.1f}% of the oracle gain')
