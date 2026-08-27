from __future__ import annotations

import json
from collections import Counter

PER_POSITION = '/workspace/search-eval/output/oracle/per-position.json'
ALLOCATION = '/workspace/search-eval/output/oracle/allocation.json'

with open(PER_POSITION, encoding='utf-8') as handle:
    report = json.load(handle)
records = report['records']
budgets = [b['visits'] for b in records[0]['budgets']]
divergences = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
count = len(records)

with open(ALLOCATION, encoding='utf-8') as handle:
    allocation = json.load(handle)
for signal in allocation['signals']:
    print(
        f'signal {signal["feature"]:26s} at {signal["signal_visits"]} -> {signal["target_visits"]} visits: '
        f'spearman {signal["spearman_with_benefit"]:+.3f}  '
        f'top-decile benefit {signal["top_decile_mean_benefit"]:.4f}  '
        f'bottom-decile {signal["bottom_decile_mean_benefit"]:.4f}  '
        f'population {signal["population_mean_benefit"]:.4f}'
    )
print()


def allocate(multiplier: float) -> list[int]:
    chosen = []
    for row in divergences:
        best = min(range(len(budgets)), key=lambda i: row[i] + multiplier * budgets[i])
        chosen.append(budgets[best])
    return chosen


print('ORACLE ALLOCATION SHAPE at several mean budgets')
for multiplier in (3e-5, 1e-4, 2e-4, 5e-4, 1e-3):
    chosen = allocate(multiplier)
    mean = sum(chosen) / count
    histogram = Counter(chosen)
    at_reference = histogram[budgets[-1]] / count
    top_two = (histogram[budgets[-1]] + histogram[budgets[-2]]) / count
    share_from_reference = sum(v for v in chosen if v == budgets[-1]) / sum(chosen)
    print(
        f'  mean {mean:7.1f}  at 10000: {at_reference:5.1%}  at >=8000: {top_two:5.1%}  '
        f'budget share spent on 10000-visit positions: {share_from_reference:5.1%}'
    )
    print(f'            distribution: {dict(sorted(histogram.items()))}')
print()

# A reference-capped metric scores any position given the full reference budget as exactly zero, so
# recompute the frontier with those positions excluded from the allocator's menu.
capped = [row[:-1] for row in divergences]
capped_budgets = budgets[:-1]
print('ORACLE WITH THE REFERENCE BUDGET REMOVED FROM THE MENU (max 8000)')
for multiplier in (3e-5, 1e-4, 2e-4, 5e-4, 1e-3):
    total_visits = 0
    total_divergence = 0.0
    for row in capped:
        best = min(range(len(capped_budgets)), key=lambda i: row[i] + multiplier * capped_budgets[i])
        total_visits += capped_budgets[best]
        total_divergence += row[best]
    print(f'  mean {total_visits / count:7.1f}  oracle KL {total_divergence / count:.4f}')
