from __future__ import annotations

import json

with open('/workspace/search-eval/output/oracle/per-position.json', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
signal_index = budgets.index(200)
target_index = budgets.index(600)
count = len(records)

benefit = [
    r['budgets'][signal_index]['kullback_leibler'] - r['budgets'][target_index]['kullback_leibler'] for r in records
]
oracle_budget_share = None

for feature in ('top_visit_share', 'top_two_margin'):
    values = [r['budgets'][signal_index][feature] for r in records]
    order = sorted(range(count), key=lambda i: values[i])
    print(f'BENEFIT OF 200 -> 600 VISITS BY DECILE OF {feature} MEASURED AT 200 VISITS')
    print(f'  {"decile":>6s} {"signal range":>18s} {"mean benefit":>13s} {"vs population":>14s}')
    population = sum(benefit) / count
    for decile in range(10):
        lo = decile * count // 10
        hi = (decile + 1) * count // 10
        chunk = order[lo:hi]
        mean_benefit = sum(benefit[i] for i in chunk) / len(chunk)
        low_value, high_value = values[chunk[0]], values[chunk[-1]]
        print(
            f'  {decile + 1:6d} {low_value:8.3f}-{high_value:<8.3f} {mean_benefit:13.4f} '
            f'{mean_benefit / population:13.2f}x'
        )
    print(f'  population mean benefit {population:.4f}')
    print()

# The adaptive rule stops when the top-visit share clears a threshold; what does it give up?
values = [r['budgets'][signal_index]['top_visit_share'] for r in records]
for threshold in (0.5, 0.6, 0.7, 0.8):
    stopped = [i for i in range(count) if values[i] >= threshold]
    if not stopped:
        continue
    forgone = sum(benefit[i] for i in stopped) / len(stopped)
    print(
        f'rule "stop at 200 visits when top share >= {threshold}": stops {len(stopped) / count:5.1%} of positions, '
        f'mean forgone benefit {forgone:.4f} versus {sum(benefit) / count:.4f} if stopping at random'
    )
