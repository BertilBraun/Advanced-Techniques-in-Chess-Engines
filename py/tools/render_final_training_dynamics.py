"""Render archive-derived final-run training figures from the tracked trajectory CSV."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/training-trajectory.csv'
FIGURE_DIRECTORY = REPOSITORY_ROOT / 'documentation/report/figures'
SMALL_TO_MEDIUM_STEPS = 240_000
BLUE = '#2d6685'
TEAL = '#438d83'
ORANGE = '#c58240'
MUTED = '#647b88'
GRID = '#d9e1e7'


@dataclass(frozen=True)
class TrainingPoint:
    optimizer_steps: int
    completed_games: int
    materialized_positions: int
    replay_live_rows: int
    policy_loss: float
    wdl_loss: float
    total_loss: float
    learning_rate: float
    training_samples_per_second: float


def required_field(row: dict[str, str | None], name: str) -> str:
    value = row[name]
    if value is None or not value:
        raise ValueError(f'Missing {name} in training trajectory row.')
    return value


def read_points(path: Path) -> tuple[TrainingPoint, ...]:
    with path.open(encoding='utf-8', newline='') as source:
        points = tuple(
            TrainingPoint(
                optimizer_steps=int(required_field(row, 'optimizer_steps')),
                completed_games=int(required_field(row, 'completed_games')),
                materialized_positions=int(required_field(row, 'materialized_positions')),
                replay_live_rows=int(required_field(row, 'replay_live_rows')),
                policy_loss=float(required_field(row, 'policy_loss')),
                wdl_loss=float(required_field(row, 'wdl_loss')),
                total_loss=float(required_field(row, 'total_loss')),
                learning_rate=float(required_field(row, 'learning_rate')),
                training_samples_per_second=float(required_field(row, 'training_samples_per_second')),
            )
            for row in csv.DictReader(source)
        )
    if not points or points[-1].optimizer_steps != 408_500:
        raise ValueError('Trajectory does not end at the selected checkpoint.')
    return points


def smooth(values: list[float], radius: int = 5) -> list[float]:
    return [fmean(values[max(0, index - radius) : index + radius + 1]) for index in range(len(values))]


def configure_axes(axes: Axes, ylabel: str, *, paper: bool) -> None:
    axes.set_ylabel(ylabel)
    axes.grid(axis='y', color='#c9d3dc' if paper else GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_color(MUTED)
    axes.spines['bottom'].set_color(MUTED)
    axes.tick_params(colors='#34495b', length=0, pad=7)
    axes.axvline(
        SMALL_TO_MEDIUM_STEPS / 1000,
        color=MUTED,
        linewidth=1.3 if paper else 1,
        linestyle='--',
        alpha=0.95 if paper else 0.8,
    )


def configure_style(*, paper: bool) -> None:
    plt.rcParams.update(
        {
            'font.family': 'Segoe UI',
            'font.size': 12 if paper else 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12 if paper else 10,
            'svg.fonttype': 'path',
            'svg.hashsalt': 'alphazero-final-training-dynamics-paper' if paper else 'alphazero-final-training-dynamics',
        }
    )


def save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format='svg',
        metadata={'Creator': 'py/tools/render_final_training_dynamics.py', 'Date': None},
    )
    preview_directory = os.environ.get('REPORT_FIGURE_PREVIEW_DIRECTORY')
    if preview_directory is not None:
        preview_path = Path(preview_directory) / path.with_suffix('.png').name
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_path, dpi=160)
    plt.close(figure)
    normalized = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n'
    with path.open('w', encoding='utf-8', newline='\n') as output:
        output.write(normalized)


def render_loss_figure(points: tuple[TrainingPoint, ...], *, paper: bool) -> None:
    x = [point.optimizer_steps / 1000 for point in points]
    figure: Figure
    axes: list[Axes]
    if paper:
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharex=True)
    else:
        figure, axes = plt.subplots(2, 1, figsize=(9.3, 6.4), sharex=True, height_ratios=(2.1, 1))
    figure.patch.set_facecolor('white')
    for axis in axes:
        axis.set_facecolor('white')
    loss_axes, rate_axes = axes
    configure_axes(loss_axes, 'Training loss', paper=paper)
    configure_axes(rate_axes, 'Learning rate', paper=paper)
    for label, values, color in (
        ('Total', [point.total_loss for point in points], '#1f5875' if paper else BLUE),
        ('Policy', [point.policy_loss for point in points], '#1f7065' if paper else TEAL),
        ('WDL', [point.wdl_loss for point in points], '#a45d1e' if paper else ORANGE),
    ):
        loss_axes.plot(x, values, color=color, linewidth=0.75 if paper else 0.55, alpha=0.35 if paper else 0.16)
        loss_axes.plot(x, smooth(values), color=color, linewidth=2.3 if paper else 1.9, label=label)
        if paper:
            loss_axes.plot(x[-1], values[-1], marker='o', color=color, markersize=4)
            loss_axes.annotate(
                f'{values[-1]:.2f}',
                (x[-1], values[-1]),
                xytext=(6, 0),
                textcoords='offset points',
                va='center',
                color=color,
                fontsize=10,
            )
    if paper:
        loss_axes.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=3, fontsize=10)
    else:
        loss_axes.legend(loc='upper right', frameon=False, ncol=3, fontsize=9)
    if not paper:
        loss_axes.annotate(
            'Medium model active',
            (244, loss_axes.get_ylim()[0]),
            xytext=(6, 8),
            textcoords='offset points',
            color=MUTED,
            fontsize=9,
        )
    rate_axes.plot(
        x,
        [point.learning_rate for point in points],
        color='#1f5875' if paper else BLUE,
        linewidth=2.2 if paper else 1.8,
    )
    if paper:
        rate_axes.set_ylim(0.015, 0.105)
        rate_axes.set_yticks((0.02, 0.04, 0.06, 0.08, 0.10))
        rate_axes.plot(x[-1], points[-1].learning_rate, marker='o', color='#1f5875', markersize=4)
        rate_axes.annotate(
            f'{points[-1].learning_rate:.4f}',
            (x[-1], points[-1].learning_rate),
            xytext=(6, 4),
            textcoords='offset points',
            color='#1f5875',
            fontsize=10,
        )
    if paper:
        figure.supxlabel('Completed optimizer steps (thousands)', y=0.01, fontsize=10)
        for axis in axes:
            axis.set_xticks((0, 200, 400))
        loss_axes.set_xlabel('Training loss')
        rate_axes.set_xlabel('Learning rate')
        loss_axes.set_ylabel('')
        rate_axes.set_ylabel('')
    else:
        rate_axes.set_xlabel('Completed optimizer steps (thousands)')
    rate_axes.set_xlim(0, 510 if paper else 410)
    if not paper:
        figure.suptitle('Final lineage: training objectives and learning rate', x=0.09, ha='left', color='#203444')
    figure.subplots_adjust(
        left=0.07 if paper else 0.10,
        right=0.98,
        top=0.88 if paper else 0.91,
        bottom=0.30 if paper else 0.11,
        wspace=0.30,
        hspace=0.15 if paper else 0.12,
    )
    suffix = '-paper.svg' if paper else '.svg'
    save_figure(figure, FIGURE_DIRECTORY / f'final-training-loss-and-rate{suffix}')


def render_volume_figure(points: tuple[TrainingPoint, ...], *, paper: bool) -> None:
    if paper:
        render_volume_paper(points)
        return
    x = [point.optimizer_steps / 1000 for point in points]
    figure: Figure
    axes: list[Axes]
    figure, axes = plt.subplots(3, 1, figsize=(6.4, 5.2) if paper else (9.3, 8.2), sharex=True, height_ratios=(1, 1, 1))
    figure.patch.set_facecolor('white')
    for axis in axes:
        axis.set_facecolor('white')
    games_axes, replay_axes, trainer_axes = axes
    configure_axes(games_axes, 'Ingested games / quantum', paper=False)
    configure_axes(replay_axes, 'Positions (millions)', paper=False)
    configure_axes(trainer_axes, 'Trainer samples/s (thousands)', paper=False)
    games = [float(point.completed_games) for point in points]
    games_axes.plot(x, games, color='#1f7065' if paper else TEAL, linewidth=0.55, alpha=0.12 if paper else 0.16)
    games_axes.plot(x, smooth(games), color='#1f7065' if paper else TEAL, linewidth=2.2 if paper else 1.8)
    replay_axes.plot(
        x,
        [point.materialized_positions / 1e6 for point in points],
        color='#1f5875' if paper else BLUE,
        linewidth=2.3 if paper else 1.9,
        label='Net materialized',
    )
    replay_axes.plot(
        x,
        [point.replay_live_rows / 1e6 for point in points],
        color='#a45d1e' if paper else ORANGE,
        linewidth=2.1 if paper else 1.7,
        label='Live replay',
    )
    replay_axes.legend(loc='upper left', frameon=False, ncol=2, fontsize=11 if paper else 9)
    throughput = [point.training_samples_per_second / 1000 for point in points]
    trainer_axes.plot(x, throughput, color='#1f5875' if paper else BLUE, linewidth=0.55, alpha=0.12 if paper else 0.16)
    trainer_axes.plot(x, smooth(throughput), color='#1f5875' if paper else BLUE, linewidth=2.2 if paper else 1.8)
    trainer_axes.set_xlabel('Completed optimizer steps (thousands)')
    trainer_axes.set_xlim(0, 410)
    if not paper:
        figure.suptitle('Final lineage: games, replay, and trainer supply', x=0.09, ha='left', color='#203444')
    figure.subplots_adjust(
        left=0.18 if paper else 0.13,
        right=0.98,
        top=0.97 if paper else 0.92,
        bottom=0.11 if paper else 0.09,
        hspace=0.18 if paper else 0.15,
    )
    suffix = '-paper.svg' if paper else '.svg'
    save_figure(figure, FIGURE_DIRECTORY / f'final-training-volume-and-throughput{suffix}')


def render_volume_paper(points: tuple[TrainingPoint, ...]) -> None:
    x = [point.optimizer_steps / 1000 for point in points]
    figure: Figure = plt.figure(figsize=(8.0, 5.4))
    games_axes: Axes = figure.add_subplot(2, 2, 1)
    positions_axes: Axes = figure.add_subplot(2, 2, 2, sharex=games_axes)
    replay_axes: Axes = figure.add_subplot(2, 2, 3, sharex=games_axes)
    trainer_axes: Axes = figure.add_subplot(2, 2, 4, sharex=games_axes)
    for axis, label in (
        (games_axes, 'Games / quantum'),
        (positions_axes, 'Net positions (millions)'),
        (replay_axes, 'Live replay (millions)'),
        (trainer_axes, 'Trainer samples/s (k)'),
    ):
        configure_axes(axis, label, paper=True)
        axis.set_xlim(0, 410)
    games = [float(point.completed_games) for point in points]
    games_axes.plot(x, games, color='#1f7065', linewidth=0.75, alpha=0.35)
    games_axes.plot(x, smooth(games), color='#1f7065', linewidth=2.2)
    positions_axes.plot(x, [point.materialized_positions / 1e6 for point in points], color='#1f5875', linewidth=2.2)
    replay_axes.plot(
        x,
        [point.replay_live_rows / 1e6 for point in points],
        color='#a45d1e',
        linewidth=2.2,
        drawstyle='steps-post',
    )
    replay_axes.set_ylim(0, 18)
    replay_axes.set_yticks((0, 4, 8, 12, 16))
    throughput = [point.training_samples_per_second / 1000 for point in points]
    trainer_axes.plot(x, throughput, color='#1f5875', linewidth=0.75, alpha=0.35)
    trainer_axes.plot(x, smooth(throughput), color='#1f5875', linewidth=2.2)
    games_axes.tick_params(labelbottom=False)
    positions_axes.tick_params(labelbottom=False)
    figure.supxlabel('Completed optimizer steps (thousands)', y=0.02)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.15, wspace=0.34, hspace=0.16)
    save_figure(figure, FIGURE_DIRECTORY / 'final-training-volume-and-throughput-paper.svg')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--paper-only',
        action='store_true',
        help='Render enlarged paper figures without rewriting the archived originals.',
    )
    arguments = parser.parse_args()
    points = read_points(SOURCE_PATH)
    variants = (True,) if arguments.paper_only else (False,)
    for paper in variants:
        configure_style(paper=paper)
        render_loss_figure(points, paper=paper)
        render_volume_figure(points, paper=paper)


if __name__ == '__main__':
    main()
