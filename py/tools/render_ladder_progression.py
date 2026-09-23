"""Render the publication comparison of chess ladder strength over training time."""

from __future__ import annotations

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
    SeriesSpecification('Early baseline', 'v9::evaluation/ladder_elo', '#737373'),
    SeriesSpecification('Architecture revision', 'v29::evaluation/ladder_elo_64', '#cc79a7'),
    SeriesSpecification(
        'Previous four-day baseline',
        'v34::evaluation/ladder_elo_64',
        '#009e73',
        cutoff_seconds=3 * 24 * 60 * 60,
        cutoff_reason='The clean evaluation interval ends at 3.0 days; later noisy points are outside report scope.',
    ),
    SeriesSpecification('Quantized successor', 'v46::evaluation/ladder_elo_64', '#e69f00'),
    SeriesSpecification(
        'Final recipe',
        'final::evaluation/ladder_elo_64',
        '#0072b2',
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


def centered_mean(values: tuple[float, ...], radius: int = 3) -> tuple[float, ...]:
    return tuple(
        sum(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        / len(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        for index in range(len(values))
    )


def configure_axes(axes: Axes) -> None:
    axes.set_xlim(0.0, 3.08)
    axes.set_ylim(600.0, 2500.0)
    axes.set_xlabel('Effective training time (days)', fontsize=11)
    axes.set_ylabel('64-search ladder Elo', fontsize=11)
    axes.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    axes.set_yticks((750, 1000, 1250, 1500, 1750, 2000, 2250, 2500))
    axes.grid(axis='both', color='#d7dce2', linewidth=0.7, alpha=0.72)
    axes.set_axisbelow(True)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_color('#6b7280')
    axes.spines['bottom'].set_color('#6b7280')
    axes.tick_params(colors='#374151', labelsize=9)


def plot_series(axes: Axes, series: PublicationSeries, color: str) -> None:
    days = tuple(point.seconds / 86_400.0 for point in series.points)
    elos = tuple(point.elo for point in series.points)
    trend = centered_mean(elos)
    axes.plot(days, elos, color=color, linewidth=0.75, alpha=0.24, zorder=1)
    axes.scatter(days, elos, color=color, s=8, alpha=0.24, edgecolors='none', zorder=2)
    axes.plot(days, trend, color=color, linewidth=2.25, label=series.label, zorder=3)
    axes.scatter(days[-1], elos[-1], color=color, s=30, edgecolors='white', linewidths=0.8, zorder=4)


def render_figure(publication: PublicationExport, path: Path) -> None:
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'svg.fonttype': 'none',
            'svg.hashsalt': 'alphazero-chess-ladder-progress',
        }
    )
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(10.8, 6.2), constrained_layout=False)
    figure.patch.set_facecolor('white')
    axes.set_facecolor('white')
    configure_axes(axes)
    for specification, series in zip(SERIES_SPECIFICATIONS, publication.series, strict=True):
        plot_series(axes, series, specification.color)

    figure.suptitle(
        'Engineering progress across five chess training campaigns',
        x=0.09,
        y=0.965,
        ha='left',
        fontsize=17,
        fontweight='bold',
        color='#111827',
    )
    axes.set_title(
        'Raw observations and centered seven-point means on the fixed-node Stockfish ladder',
        loc='left',
        pad=12,
        fontsize=10.5,
        color='#4b5563',
    )
    legend = axes.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.96,
        facecolor='white',
        edgecolor='#d1d5db',
        fontsize=9,
        ncol=1,
    )
    legend.get_frame().set_linewidth(0.7)
    axes.text(
        0.0,
        -0.17,
        'Report cuts: final recipe at 2.5 days; previous baseline at 3.0 days. '
        'Later experimental/noisy points are excluded. The early baseline uses the legacy primary ladder tag.',
        transform=axes.transAxes,
        ha='left',
        va='top',
        fontsize=8.5,
        color='#4b5563',
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format='svg',
        metadata={'Creator': 'py/tools/render_ladder_progression.py', 'Date': None},
    )
    plt.close(figure)
    normalized_svg = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n'
    with path.open('w', encoding='utf-8', newline='\n') as svg_file:
        svg_file.write(normalized_svg)


def main() -> None:
    source = load_source(SOURCE_PATH)
    publication = PublicationExport(
        source_sha256=source_sha256(SOURCE_PATH),
        metric='64-search fixed-node Stockfish ladder Elo',
        time_axis='effective training seconds',
        series=tuple(select_publication_series(source, specification) for specification in SERIES_SPECIFICATIONS),
    )
    with PUBLICATION_DATA_PATH.open('w', encoding='utf-8', newline='\n') as publication_file:
        publication_file.write(publication.model_dump_json(indent=2, exclude_none=True) + '\n')
    render_figure(publication, FIGURE_PATH)


if __name__ == '__main__':
    main()
