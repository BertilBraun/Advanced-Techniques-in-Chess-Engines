from __future__ import annotations

import gzip
import json
import statistics

PATH = 'C:/Projects/AZ-search-eval/documentation/benchmarks/chess-search-evaluation-rtx3060-20260826/results-depthsweep/per-position.json.gz'
with gzip.open(PATH, 'rt', encoding='utf-8') as handle:
    records = json.load(handle)['records']

budgets = [b['visits'] for b in records[0]['budgets']]
kl = [[b['kullback_leibler'] for b in r['budgets']] for r in records]
n = len(records)


def isotonic(row: list[float]) -> list[float]:
    out = list(row)
    for index in range(len(out) - 2, -1, -1):
        out[index] = max(out[index], out[index + 1])
    return out


# Noise-free curves: a real predictor cannot exploit a lucky early stop.
curves = [isotonic(row) for row in kl]


def oracle(target_mean: float) -> tuple[float, list[int]]:
    best = None
    for step in range(4000):
        multiplier = 10.0 ** (-7.0 + step * 0.002)
        chosen = [min(range(len(budgets)), key=lambda i: row[i] + multiplier * budgets[i]) for row in curves]
        mean_visits = statistics.fmean([budgets[i] for i in chosen])
        if best is None or abs(mean_visits - target_mean) < abs(best[0] - target_mean):
            best = (mean_visits, statistics.fmean([curves[p][chosen[p]] for p in range(n)]), chosen)
    return best[1], best[2]


for baseline in (600, 1000):
    if baseline not in budgets:
        continue
    i_base = budgets.index(baseline)
    flat = statistics.fmean([row[i_base] for row in curves])
    oracle_kl, oracle_choice = oracle(float(baseline))
    oracle_budgets = sorted([budgets[i] for i in oracle_choice], reverse=True)
    truth = [row[i_base] for row in curves]

    print(f'=== BASELINE {baseline} VISITS (noise-free curves) ===')
    print(f'    flat {flat:.4f}   oracle {oracle_kl:.4f}')
    below = sum(1 for i in oracle_choice if budgets[i] < baseline) / n
    above = sum(1 for i in oracle_choice if budgets[i] > baseline) / n
    print(f'    oracle sends {below:.0%} below baseline, {above:.0%} above')
    print()
    print(f'    {"label = TV(baseline, depth)":34s} {"depth":>7s} {"x base":>7s} {"captures":>9s}')

    def capture(signal: list[float]) -> float:
        order = sorted(range(n), key=lambda i: -signal[i])
        assigned = [0] * n
        for rank, position in enumerate(order):
            assigned[position] = oracle_budgets[rank]
        value = statistics.fmean([curves[p][budgets.index(assigned[p])] for p in range(n)])
        return (flat - value) / (flat - oracle_kl) * 100.0

    candidates = [c for c in records[0]['label_candidates'] if c['baseline_visits'] == baseline]
    for candidate in candidates:
        depth = candidate['depth_visits']
        signal = [
            next(
                c['total_variation']
                for c in r['label_candidates']
                if c['baseline_visits'] == baseline and c['depth_visits'] == depth
            )
            for r in records
        ]
        print(f'    {"":34s} {depth:7d} {depth / baseline:6.1f}x {capture(signal):8.1f}%')
    print(f'    {"true remaining error vs 10000":34s} {10000:7d} {10000 / baseline:6.1f}x {capture(truth):8.1f}%')
    print()
