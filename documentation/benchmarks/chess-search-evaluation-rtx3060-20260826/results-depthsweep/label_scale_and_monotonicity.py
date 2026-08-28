from __future__ import annotations

import gzip
import json
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-corrections/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
kl = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
tv = [[b['total_variation'] for b in r['budgets']] for r in records]
n = len(records)
i600 = budgets.index(600)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


print('1. WHAT SCALE IS THE LABEL ACTUALLY ON?   TV(target@600, target@10000), 3000 positions')
label = [row[i600] for row in tv]
for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
    print(f'     quantile {q:4.2f}   {quantile(label, q):.4f}')
print(f'     mean {statistics.fmean(label):.4f}   share below 0.05: {sum(1 for v in label if v < 0.05) / n:.1%}')
print()

print('2. IS THE PER-POSITION CURVE ACTUALLY MONOTONE IN VISITS?')
pairs = 0
inversions = 0
positions_with_any = 0
for row in kl:
    has = False
    for a, b in zip(row, row[1:], strict=False):
        pairs += 1
        if b > a + 1e-12:
            inversions += 1
            has = True
    positions_with_any += has
print(f'     adjacent budget steps where more search moved the target FURTHER from truth: {inversions / pairs:.1%}')
print(f'     positions showing at least one such step: {positions_with_any / n:.1%}')
print()


def isotonic(row: list[float]) -> list[float]:
    """Force the curve to be non-increasing by taking the running minimum from the deep end backwards."""
    out = list(row)
    for index in range(len(out) - 2, -1, -1):
        out[index] = max(out[index], out[index + 1])
    return out


kl_monotone = [isotonic(row) for row in kl]


def oracle(curves: list[list[float]], menu: list[int], target_mean: float) -> float:
    indices = [budgets.index(v) for v in menu]
    best = None
    for step in range(4000):
        multiplier = 10.0 ** (-7.0 + step * 0.002)
        chosen = [min(indices, key=lambda i: row[i] + multiplier * budgets[i]) for row in curves]
        mean_visits = statistics.fmean([budgets[i] for i in chosen])
        if best is None or abs(mean_visits - target_mean) < abs(best[0] - target_mean):
            best = (mean_visits, statistics.fmean([curves[p][chosen[p]] for p in range(n)]))
    return best[1]


flat600 = statistics.fmean([row[i600] for row in kl])
flat400 = statistics.fmean([row[budgets.index(400)] for row in kl])
print('3. HOW MUCH OF THE ORACLE GAIN SURVIVES IF IT CANNOT EXPLOIT THOSE INVERSIONS?')
print(f'     flat 600                                          KL {flat600:.4f}')
print(f'     raw oracle, mean 600                              KL {oracle(kl, budgets, 600.0):.4f}')
print(f'     monotonised oracle, mean 600                      KL {oracle(kl_monotone, budgets, 600.0):.4f}')
print()
print(f'     flat 400                                          KL {flat400:.4f}')
print(
    f'     raw shrink-only oracle (<=600), mean 400          KL {oracle(kl, [b for b in budgets if b <= 600], 400.0):.4f}'
)
print(
    f'     monotonised shrink-only oracle (<=600), mean 400  KL {oracle(kl_monotone, [b for b in budgets if b <= 600], 400.0):.4f}'
)
