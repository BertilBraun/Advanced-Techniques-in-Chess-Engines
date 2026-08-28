from __future__ import annotations

import gzip
import json
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-oracle/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
i200, i600, i1200, i2400 = (budgets.index(v) for v in (200, 600, 1200, 2400))
kl = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
n = len(records)

# "Contested" = the search moves the target a lot between a cheap and the production budget.
benefit = [row[i200] - row[i600] for row in kl]
order = sorted(range(n), key=lambda i: -benefit[i])
quartile = n // 4

print('WHICH POSITIONS SHOULD BECOME TRAINING TARGETS?')
print('Fast-search positions produce no sample at all, so the only question is which 25% get a full search.')
print()
print('Target RELIABILITY of the selected quartile, i.e. how far its 600-visit target still sits from truth:')
for name, chosen in (
    ('random 25% (today)', list(range(n))),
    ('most contested 25%', order[:quartile]),
    ('least contested 25%', order[-quartile:]),
):
    sample = chosen if len(chosen) == n else chosen
    mean600 = statistics.fmean([kl[i][i600] for i in sample])
    mean1200 = statistics.fmean([kl[i][i1200] for i in sample])
    mean2400 = statistics.fmean([kl[i][i2400] for i in sample])
    print(f'  {name:22s} KL at 600 {mean600:.4f}   at 1200 {mean1200:.4f}   at 2400 {mean2400:.4f}')
print()

contested = order[:quartile]
print('So selecting contested positions buys learning signal but costs target accuracy.')
print('How many visits does the contested quartile need to reach the accuracy a random quartile gets at 600?')
random_600 = statistics.fmean([row[i600] for row in kl])
for visits in budgets:
    index = budgets.index(visits)
    value = statistics.fmean([kl[i][index] for i in contested])
    if value <= random_600:
        print(
            f'  contested quartile reaches KL {value:.4f} at {visits} visits (random quartile at 600 = {random_600:.4f})'
        )
        break
else:
    print(
        f'  contested quartile never reaches {random_600:.4f} within the measured grid (best '
        f'{statistics.fmean([kl[i][budgets.index(10000)] for i in contested]):.4f} at 10000)'
    )
print()

print('Budget reallocation across the full searches only, holding total full-search compute fixed at 25% x 600:')
total_budget = quartile * 600
for label, ranked in (('by contestedness', order),):
    for top_share, top_visits in ((0.25, 1200), (0.5, 1200), (0.5, 2400)):
        top_count = int(quartile * top_share)
        rest_count = quartile - top_count
        if top_count * top_visits > total_budget:
            continue
        rest_visits = (total_budget - top_count * top_visits) // max(1, rest_count)
        candidates = [v for v in budgets if v <= rest_visits]
        if not candidates:
            continue
        rest_actual = max(candidates)
        chosen_top = ranked[:top_count]
        chosen_rest = ranked[top_count:quartile]
        mean = statistics.fmean(
            [kl[i][budgets.index(top_visits)] for i in chosen_top]
            + [kl[i][budgets.index(rest_actual)] for i in chosen_rest]
        )
        print(f'  top {top_share:.0%} of the quartile at {top_visits}, rest at {rest_actual}: mean KL {mean:.4f}')
print(f'  flat 600 for the whole quartile{"":18s} mean KL {statistics.fmean([kl[i][i600] for i in contested]):.4f}')
