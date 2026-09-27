"""Render the publication comparison of chess ladder strength over training time."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / 'documentation' / 'evidence' / 'final-chess-20260923' / 'ladder-elo-export.json'
PUBLICATION_DATA_PATH = (
    REPOSITORY_ROOT / 'documentation' / 'evidence' / 'final-chess-20260923' / 'ladder-elo-report-trimmed.json'
)
FIGURE_PATH = REPOSITORY_ROOT / 'documentation' / 'showcase' / 'chess-ladder-progress.svg'
REPORT_FIGURE_PATH = REPOSITORY_ROOT / 'documentation' / 'report' / 'figures' / 'chess-ladder-progress-paper.svg'


class LadderPoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    seconds: int
    raw_seconds: int | None = None
    elo: float


class LadderSeries(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source: str
    points: tuple[LadderPoint, ...]


class LadderExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    note: str
    series: dict[str, LadderSeries]


class PublicationSeries(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    label: str
    source_key: str
    source_identity: str
    cutoff_seconds: int | None
    cutoff_reason: str | None
    points: tuple[LadderPoint, ...]


class PublicationExport(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source_sha256: str
    metric: str
    time_axis: str
    series: tuple[PublicationSeries, ...]


@dataclass(frozen=True)
class SeriesSpecification:
    label: str
    source_key: str
    color: str
    cutoff_seconds: int | None = None
    cutoff_reason: str | None = None


SERIES_SPECIFICATIONS = (
    SeriesSpecification('Early baseline', 'v9::evaluation/ladder_elo', '#b8860b'),
    SeriesSpecification('Architecture revision', 'v29::evaluation/ladder_elo_64', '#7b4fa8'),
    SeriesSpecification(
        'Previous four-day baseline',
        'v34::evaluation/ladder_elo_64',
        '#2e7d32',
        cutoff_seconds=3 * 24 * 60 * 60,
        cutoff_reason='The clean evaluation interval ends at 3.0 days; later noisy points are outside report scope.',
    ),
    SeriesSpecification('Quantized successor', 'v46::evaluation/ladder_elo_64', '#c1440e'),
    SeriesSpecification(
        'Final recipe',
        'final::evaluation/ladder_elo_64',
        '#1f4e79',
        cutoff_seconds=5 * 12 * 60 * 60,
        cutoff_reason='The accepted final training result ends at 2.5 days; later experimentation is excluded.',
    ),
)


def load_source(path: Path) -> LadderExport:
    return LadderExport.model_validate_json(path.read_text(encoding='utf-8'))


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_publication_series(
    export: LadderExport,
    specification: SeriesSpecification,
) -> PublicationSeries:
    source_series = export.series[specification.source_key]
    selected_points = tuple(
        point
        for point in source_series.points
        if specification.cutoff_seconds is None or point.seconds <= specification.cutoff_seconds
    )
    if not selected_points:
        raise ValueError(f'No points selected for {specification.label}')
    if specification.cutoff_seconds is not None and selected_points[-1].seconds != specification.cutoff_seconds:
        raise ValueError(f'{specification.label} has no observation at its requested cutoff')
    return PublicationSeries(
        label=specification.label,
        source_key=specification.source_key,
        source_identity=source_series.source,
        cutoff_seconds=specification.cutoff_seconds,
        cutoff_reason=specification.cutoff_reason,
        points=selected_points,
    )


def bias_corrected_ema(values: tuple[float, ...], decay: float = 0.95) -> tuple[float, ...]:
    numerator = 0.0
    denominator = 0.0
    smoothed_values: list[float] = []
    for value in values:
        numerator = numerator * decay + value
        denominator = denominator * decay + 1.0
        smoothed_values.append(numerator / denominator)
    return tuple(smoothed_values)


def configure_axes(axes: Axes) -> None:
    axes.set_xlim(0.0, 3.05)
    axes.set_ylim(600.0, 2500.0)
    axes.set_xlabel('effective training time (days)')
    axes.set_ylabel('64-search ladder Elo')
    axes.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    axes.set_yticks((750, 1000, 1250, 1500, 1750, 2000, 2250, 2500))
    axes.grid(axis='both', color='#b0b0b0', linewidth=0.6, alpha=0.25)
    axes.set_axisbelow(True)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_color('#404040')
    axes.spines['bottom'].set_color('#404040')
    axes.tick_params(colors='#303030')


def plot_series(axes: Axes, series: PublicationSeries, color: str) -> None:
    days = tuple(point.seconds / 86_400.0 for point in series.points)
    elos = tuple(point.elo for point in series.points)
    axes.plot(days, bias_corrected_ema(elos), color=color, linewidth=2.35, label=series.label)


def render_figure(publication: PublicationExport, path: Path, *, paper: bool = False) -> None:
    plt.rcParams.update(
        {
            'figure.dpi': 110,
            'font.family': 'Segoe UI',
            'font.size': 10,
            'axes.titlesize': 13,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 10,
            'legend.frameon': False,
            'legend.fontsize': 9,
            'svg.fonttype': 'path',
            'svg.hashsalt': 'alphazero-chess-ladder-progress',
        }
    )
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(9.2, 4.82), constrained_layout=False)
    figure.patch.set_facecolor('white')
    axes.set_facecolor('white')
    configure_axes(axes)
    for specification, series in zip(SERIES_SPECIFICATIONS, publication.series, strict=True):
        plot_series(axes, series, specification.color)

    if not paper:
        axes.set_title(
            '64-search ladder Elo across training campaigns',
            loc='left',
            pad=10,
            color='#202020',
        )
    axes.legend(
        loc='lower right',
        ncol=1,
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.98 if paper else 0.93, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.04,
        metadata={'Creator': 'py/tools/render_ladder_progression.py', 'Date': None},
    )
    plt.close(figure)
    normalized_svg = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n'
    with path.open('w', encoding='utf-8', newline='\n') as svg_file:
        svg_file.write(normalized_svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-only', action='store_true', help='Render the title-free report plot only.')
    arguments = parser.parse_args()
    source = load_source(SOURCE_PATH)
    publication = PublicationExport(
        source_sha256=source_sha256(SOURCE_PATH),
        metric='64-search fixed-node Stockfish ladder Elo',
        time_axis='effective training seconds',
        series=tuple(select_publication_series(source, specification) for specification in SERIES_SPECIFICATIONS),
    )
    if arguments.paper_only:
        render_figure(publication, REPORT_FIGURE_PATH, paper=True)
    else:
        with PUBLICATION_DATA_PATH.open('w', encoding='utf-8', newline='\n') as publication_file:
            publication_file.write(publication.model_dump_json(indent=2, exclude_none=True) + '\n')
        render_figure(publication, FIGURE_PATH)


if __name__ == '__main__':
    main()
