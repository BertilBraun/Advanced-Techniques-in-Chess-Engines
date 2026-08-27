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
BASE = 600
i_base = budgets.index(BASE)

best = None
for step in range(4000):
    multiplier = 10.0 ** (-7.0 + step * 0.002)
    chosen = [min(range(len(budgets)), key=lambda i: row[i] + multiplier * budgets[i]) for row in kl]
    mean_visits = statistics.fmean([budgets[i] for i in chosen])
    if best is None or abs(mean_visits - BASE) < abs(best[0] - BASE):
        best = (mean_visits, chosen)
_, oracle_choice = best
oracle_budget = [budgets[i] for i in oracle_choice]

# The candidate label: remaining error of the baseline-budget target, measured against truth.
label = [row[i_base] for row in kl]


def spearman(a: list[float], b: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0
            for index in range(position, end + 1):
                out[order[index]] = average
            position = end + 1
        return out

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5 if va and vb else 0.0


print('DOES "REMAINING ERROR AT THE BASELINE" KNOW ANYTHING ABOUT STOPPING EARLY?')
print(f'   base budget {BASE}, oracle mean {statistics.fmean(oracle_budget):.0f}')
print()
print(f'   Spearman(label, oracle budget) over all positions: {spearman(label, oracle_budget):+.3f}')
print()

below = [i for i in range(n) if oracle_budget[i] < BASE]
at = [i for i in range(n) if oracle_budget[i] == BASE]
above = [i for i in range(n) if oracle_budget[i] > BASE]
print('   How the label is distributed across what the oracle actually decides:')
for name, group in (
    ('oracle spends LESS than baseline', below),
    ('exactly baseline', at),
    ('MORE than baseline', above),
):
    values = [label[i] for i in group]
    print(
        f'     {name:34s} {len(group):5d} positions ({len(group) / n:4.0%})   '
        f'median label {statistics.median(values):.4f}   mean {statistics.fmean(values):.4f}'
    )
print()

print('   Ranking power inside each region (can it order positions it has already placed there?):')
for name, group in (('among the LESS-than-baseline group', below), ('among the MORE-than-baseline group', above)):
    if len(group) < 10:
        continue
    rho = spearman([label[i] for i in group], [float(oracle_budget[i]) for i in group])
    print(f'     {name:36s} Spearman {rho:+.3f}')
print()


# Separate the two halves of the job: only shrinking budgets, or only growing them.
def oracle_restricted(menu: list[int], target_mean: float) -> float:
    indices = [budgets.index(v) for v in menu]
    result = None
    for step in range(4000):
        multiplier = 10.0 ** (-7.0 + step * 0.002)
        chosen = [min(indices, key=lambda i: row[i] + multiplier * budgets[i]) for row in kl]
        mean_visits = statistics.fmean([budgets[i] for i in chosen])
        if result is None or abs(mean_visits - target_mean) < abs(result[0] - target_mean):
            result = (mean_visits, statistics.fmean([kl[p][chosen[p]] for p in range(n)]))
    return result[1]


flat = statistics.fmean([row[i_base] for row in kl])
full = statistics.fmean([kl[p][oracle_choice[p]] for p in range(n)])
shrink_only = oracle_restricted([b for b in budgets if b <= BASE], 400.0)
flat400 = statistics.fmean([row[budgets.index(400)] for row in kl])
print('   Splitting the job in two, to see which half carries the value:')
print(f'     flat {BASE} everywhere                          KL {flat:.4f}')
print(f'     full two-sided oracle at mean {BASE}            KL {full:.4f}')
print(f'     shrink-only oracle, budgets <= {BASE}, mean 400 KL {shrink_only:.4f}')
print(f'     flat 400 everywhere (same mean, no choosing)   KL {flat400:.4f}')
print(f'     -> shrinking well is worth {flat400 - shrink_only:.4f} KL at mean 400')
