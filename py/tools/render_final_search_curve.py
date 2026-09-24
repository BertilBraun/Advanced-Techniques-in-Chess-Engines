"""Render the selected chess model's terminal search curve from frozen results."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/evaluation-results.csv'
FIGURE_PATH = REPOSITORY_ROOT / 'documentation/report/figures/final-search-curve.svg'
CONDITION_PAIRS = (
    ('policy_vs_1k', 'policy_vs_2k'),
    ('search100_p1_vs_10k', 'search100_p1_vs_5k'),
    ('search1k_p1_vs_50k', 'search1k_p1_vs_20k'),
    ('search10k_p4_vs_100k', 'search10k_p4_vs_50k'),
    ('search100k_p16_vs_200k', 'search100k_p16_vs_100k'),
)
AXIS_LABELS = (
    'Policy only\nfloat',
    '100\n1 parallel',
    '1,000\n1 parallel',
    '10,000\n4 parallel',
    '100,000\n16 parallel',
)


@dataclass(frozen=True)
class MatchEstimate:
    condition: str
    elo: int
    lower: int
    upper: int


@dataclass(frozen=True)
class SearchConditionPair:
    headline: MatchEstimate
    alternative: MatchEstimate


def required_field(row: dict[str, str | None], name: str) -> str:
    value = row[name]
    if value is None or not value:
        raise ValueError(f'Missing {name} in terminal result row')
    return value


def read_estimates(path: Path) -> tuple[SearchConditionPair, ...]:
    estimates: dict[str, MatchEstimate] = {}
    with path.open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            condition = required_field(row, 'condition')
            estimates[condition] = MatchEstimate(
                condition=condition,
                elo=int(required_field(row, 'model_elo')),
                lower=int(required_field(row, 'ci_low')),
                upper=int(required_field(row, 'ci_high')),
            )
    pairs: list[SearchConditionPair] = []
    for headline, alternative in CONDITION_PAIRS:
        if headline not in estimates or alternative not in estimates:
            raise ValueError(f'Missing terminal pair: {headline}, {alternative}')
        pairs.append(SearchConditionPair(headline=estimates[headline], alternative=estimates[alternative]))
    return tuple(pairs)


def configure_axes(axes: Axes) -> None:
    axes.set_xlim(-0.25, 4.35)
    axes.set_ylim(1450, 3400)
    axes.set_xticks(range(5), AXIS_LABELS)
    axes.set_yticks((1600, 2000, 2400, 2800, 3200))
    axes.set_ylabel('Benchmark Elo')
    axes.set_xlabel('Searches per move')
    axes.grid(axis='y', color='#d9e1e7', linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_color('#617384')
    axes.spines['bottom'].set_color('#617384')
    axes.tick_params(colors='#34495b', length=0, pad=8)


def render_figure(pairs: tuple[SearchConditionPair, ...], path: Path) -> None:
    plt.rcParams.update(
        {
            'font.family': 'Segoe UI',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 10,
            'svg.fonttype': 'path',
            'svg.hashsalt': 'alphazero-final-search-curve',
        }
    )
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(9.2, 5.1))
    figure.patch.set_facecolor('white')
    axes.set_facecolor('white')
    configure_axes(axes)

    headline = [pair.headline for pair in pairs]
    alternative = [pair.alternative for pair in pairs]
    positions = tuple(range(len(headline)))
    axes.plot(positions, [point.elo for point in headline], color='#2d6685', linewidth=2.4, zorder=2)
    axes.errorbar(
        positions,
        [point.elo for point in headline],
        yerr=[
            [point.elo - point.lower for point in headline],
            [point.upper - point.elo for point in headline],
        ],
        fmt='o',
        color='#2d6685',
        markerfacecolor='white',
        markeredgewidth=2,
        markersize=8,
        elinewidth=1.8,
        capsize=3,
        zorder=4,
        label='Headline rung (score nearest 0.5)',
    )
    axes.errorbar(
        [position + 0.10 for position in positions],
        [point.elo for point in alternative],
        yerr=[
            [point.elo - point.lower for point in alternative],
            [point.upper - point.elo for point in alternative],
        ],
        fmt='D',
        color='#a5b6bf',
        markerfacecolor='#a5b6bf',
        markersize=5,
        elinewidth=1,
        capsize=2,
        alpha=0.8,
        zorder=3,
        label='Other tested opponent rung',
    )
    for position, point in zip(positions, headline, strict=True):
        axes.annotate(
            f'{point.elo:,}',
            (position, point.upper),
            xytext=(0, 11),
            textcoords='offset points',
            ha='center',
            color='#25475d',
            fontsize=10,
            fontweight='semibold',
        )
    axes.set_title('Final model: playing strength across search budgets', loc='left', pad=17, color='#203444')
    axes.legend(loc='lower right', frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format='svg',
        metadata={'Creator': 'py/tools/render_final_search_curve.py', 'Date': None},
    )
    plt.close(figure)
    normalized = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n'
    with path.open('w', encoding='utf-8', newline='\n') as output:
        output.write(normalized)


def main() -> None:
    render_figure(read_estimates(SOURCE_PATH), FIGURE_PATH)


if __name__ == '__main__':
    main()
