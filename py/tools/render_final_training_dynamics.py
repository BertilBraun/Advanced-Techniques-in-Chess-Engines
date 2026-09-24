"""Render archive-derived final-run training figures from the tracked trajectory CSV."""

from __future__ import annotations

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


def configure_axes(axes: Axes, ylabel: str) -> None:
    axes.set_ylabel(ylabel)
    axes.grid(axis='y', color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_color(MUTED)
    axes.spines['bottom'].set_color(MUTED)
    axes.tick_params(colors='#34495b', length=0, pad=7)
    axes.axvline(SMALL_TO_MEDIUM_STEPS / 1000, color=MUTED, linewidth=1, linestyle='--', alpha=0.8)


def configure_style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'Segoe UI',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 10,
            'svg.fonttype': 'path',
            'svg.hashsalt': 'alphazero-final-training-dynamics',
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


def render_loss_figure(points: tuple[TrainingPoint, ...]) -> None:
    x = [point.optimizer_steps / 1000 for point in points]
    figure: Figure
    axes: list[Axes]
    figure, axes = plt.subplots(2, 1, figsize=(9.3, 6.4), sharex=True, height_ratios=(2.1, 1))
    figure.patch.set_facecolor('white')
    for axis in axes:
        axis.set_facecolor('white')
    loss_axes, rate_axes = axes
    configure_axes(loss_axes, 'Training loss')
    configure_axes(rate_axes, 'Learning rate')
    for label, values, color in (
        ('Total', [point.total_loss for point in points], BLUE),
        ('Policy', [point.policy_loss for point in points], TEAL),
        ('WDL', [point.wdl_loss for point in points], ORANGE),
    ):
        loss_axes.plot(x, values, color=color, linewidth=0.55, alpha=0.16)
        loss_axes.plot(x, smooth(values), color=color, linewidth=1.9, label=label)
    loss_axes.legend(loc='upper right', frameon=False, ncol=3, fontsize=9)
    loss_axes.annotate(
        'Medium model active',
        (244, loss_axes.get_ylim()[0]),
        xytext=(6, 8),
        textcoords='offset points',
        color=MUTED,
        fontsize=9,
    )
    rate_axes.plot(x, [point.learning_rate for point in points], color=BLUE, linewidth=1.8)
    rate_axes.set_xlabel('Completed optimizer steps (thousands)')
    rate_axes.set_xlim(0, 410)
    figure.suptitle('Final lineage: training objectives and learning rate', x=0.09, ha='left', color='#203444')
    figure.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.11, hspace=0.12)
    save_figure(figure, FIGURE_DIRECTORY / 'final-training-loss-and-rate.svg')


def render_volume_figure(points: tuple[TrainingPoint, ...]) -> None:
    x = [point.optimizer_steps / 1000 for point in points]
    figure: Figure
    axes: list[Axes]
    figure, axes = plt.subplots(3, 1, figsize=(9.3, 8.2), sharex=True, height_ratios=(1, 1, 1))
    figure.patch.set_facecolor('white')
    for axis in axes:
        axis.set_facecolor('white')
    games_axes, replay_axes, trainer_axes = axes
    configure_axes(games_axes, 'Ingested games / quantum')
    configure_axes(replay_axes, 'Positions (millions)')
    configure_axes(trainer_axes, 'Trainer samples/s (thousands)')
    games = [float(point.completed_games) for point in points]
    games_axes.plot(x, games, color=TEAL, linewidth=0.55, alpha=0.16)
    games_axes.plot(x, smooth(games), color=TEAL, linewidth=1.8)
    replay_axes.plot(
        x, [point.materialized_positions / 1e6 for point in points], color=BLUE, linewidth=1.9, label='Net materialized'
    )
    replay_axes.plot(
        x, [point.replay_live_rows / 1e6 for point in points], color=ORANGE, linewidth=1.7, label='Live replay'
    )
    replay_axes.legend(loc='upper left', frameon=False, ncol=2, fontsize=9)
    throughput = [point.training_samples_per_second / 1000 for point in points]
    trainer_axes.plot(x, throughput, color=BLUE, linewidth=0.55, alpha=0.16)
    trainer_axes.plot(x, smooth(throughput), color=BLUE, linewidth=1.8)
    trainer_axes.set_xlabel('Completed optimizer steps (thousands)')
    trainer_axes.set_xlim(0, 410)
    figure.suptitle('Final lineage: games, replay, and trainer supply', x=0.09, ha='left', color='#203444')
    figure.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.09, hspace=0.15)
    save_figure(figure, FIGURE_DIRECTORY / 'final-training-volume-and-throughput.svg')


def main() -> None:
    configure_style()
    points = read_points(SOURCE_PATH)
    render_loss_figure(points)
    render_volume_figure(points)


if __name__ == '__main__':
    main()
