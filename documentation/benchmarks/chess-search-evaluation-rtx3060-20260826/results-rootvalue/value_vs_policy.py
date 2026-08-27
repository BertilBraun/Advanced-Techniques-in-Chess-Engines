from __future__ import annotations

import gzip
import json
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-rootvalue/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
reference_index = len(budgets) - 1
n = len(records)

value_error = []
policy_error = []
for record in records:
    reference_value = record['budgets'][reference_index]['root_value']
    value_error.append([abs(b['root_value'] - reference_value) for b in record['budgets']])
    policy_error.append([b['total_variation'] for b in record['budgets']])

print(f'DOES ROOT VALUE CONVERGE FASTER THAN THE POLICY TARGET?  ({n} positions, reference {budgets[-1]} visits)')
print()
print(f'{"visits":>7s} {"|value err|":>12s} {"value % left":>13s} {"policy TV":>11s} {"policy % left":>14s}')
base_value = statistics.fmean([row[0] for row in value_error])
base_policy = statistics.fmean([row[0] for row in policy_error])
for index, visits in enumerate(budgets):
    mean_value = statistics.fmean([row[index] for row in value_error])
    mean_policy = statistics.fmean([row[index] for row in policy_error])
    print(
        f'{visits:7d} {mean_value:12.4f} {100.0 * mean_value / base_value:12.1f}% '
        f'{mean_policy:11.4f} {100.0 * mean_policy / base_policy:13.1f}%'
    )

print()
print('Both columns are normalised to their own error at the smallest budget, so the shapes are comparable.')
print('If the value column falls faster, cheap searches still yield usable value targets.')
print()

for threshold in (0.05, 0.10):
    print(f'Fraction of positions whose root value is already within {threshold:.2f} of the reference:')
    line = []
    for index, visits in enumerate(budgets[:8]):
        share = sum(1 for row in value_error if row[index] <= threshold) / n
        line.append(f'{visits}:{share:.0%}')
    print('   ' + '  '.join(line))

print()
print('Same question for the policy target, fraction already naming the reference best move:')
line = []
for index, visits in enumerate(budgets[:8]):
    share = sum(1 for r in records if r['budgets'][index]['leader_matches_reference']) / n
    line.append(f'{visits}:{share:.0%}')
print('   ' + '  '.join(line))
