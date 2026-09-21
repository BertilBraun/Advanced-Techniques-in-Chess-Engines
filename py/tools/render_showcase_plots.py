from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('svg')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LADDER_ELO_TAG = 'evaluation/ladder_elo'
LEVEL_ZERO_SCORE_TAG = 'evaluation/stockfish-level-0/score'
FIXED_DATASET_ACCURACY_TAG = 'evaluation/fixed-dataset/top_action_accuracy'
FIXED_DATASET_CROSS_ENTROPY_TAG = 'evaluation/fixed-dataset/policy_cross_entropy'
TRAINING_POLICY_LOSS_TAG = 'training/policy_loss'
TRAINING_VALUE_LOSS_TAG = 'training/wdl_loss'
OPTIMIZER_STEPS_TAG = 'training/optimizer_steps'

EVALUATION_FILE_PATTERN = re.compile(r'^(?P<boundary>\d+)-(?P<definition>.+?)-g(?P<generation>\d+)\.json$')

SERIES_COLORS = ('#1f4e79', '#c1440e', '#2e7d32', '#7b4fa8', '#b8860b')
REFERENCE_COLOR = '#8a8a8a'


@dataclass(frozen=True)
class ScalarSeries:
    steps: tuple[float, ...]
    values: tuple[float, ...]

    @property
    def is_empty(self) -> bool:
        return not self.steps

    @property
    def hours(self) -> tuple[float, ...]:
        return tuple(step / 3600.0 for step in self.steps)


@dataclass(frozen=True)
class RunSeries:
    label: str
    archive_name: str
    ladder_elo: ScalarSeries
    level_zero_score: ScalarSeries
    fixed_dataset_accuracy: ScalarSeries
    fixed_dataset_cross_entropy: ScalarSeries
    training_policy_loss: ScalarSeries
    training_value_loss: ScalarSeries
    optimizer_steps: ScalarSeries


@dataclass(frozen=True)
class ReferenceCurve:
    label: str
    hours: tuple[float, ...]
    level_zero_score: tuple[float, ...]
    fixed_dataset_accuracy: tuple[float, ...]
    fixed_dataset_cross_entropy: tuple[float, ...]


@dataclass(frozen=True)
class LabelledArchive:
    label: str
    path: Path


def _deduplicate_by_step(points: Iterable[tuple[float, float]]) -> ScalarSeries:
    latest_by_step: dict[float, float] = {}
    for step, value in points:
        latest_by_step[step] = value
    ordered = sorted(latest_by_step.items())
    return ScalarSeries(steps=tuple(step for step, _ in ordered), values=tuple(value for _, value in ordered))


def _find_coordinator_event_directory(archive: Path) -> Path | None:
    candidates = sorted(archive.rglob('coordinator/events.out.tfevents.*'))
    return candidates[0].parent if candidates else None


def _read_scalar(accumulator: EventAccumulator, available_tags: frozenset[str], tag: str) -> ScalarSeries:
    if tag not in available_tags:
        return ScalarSeries(steps=(), values=())
    return _deduplicate_by_step((float(event.step), float(event.value)) for event in accumulator.Scalars(tag))


def _load_from_tensorboard(label: str, archive: Path, event_directory: Path) -> RunSeries:
    accumulator = EventAccumulator(str(event_directory), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = frozenset(accumulator.Tags()['scalars'])
    return RunSeries(
        label=label,
        archive_name=archive.name,
        ladder_elo=_read_scalar(accumulator, tags, LADDER_ELO_TAG),
        level_zero_score=_read_scalar(accumulator, tags, LEVEL_ZERO_SCORE_TAG),
        fixed_dataset_accuracy=_read_scalar(accumulator, tags, FIXED_DATASET_ACCURACY_TAG),
        fixed_dataset_cross_entropy=_read_scalar(accumulator, tags, FIXED_DATASET_CROSS_ENTROPY_TAG),
        training_policy_loss=_read_scalar(accumulator, tags, TRAINING_POLICY_LOSS_TAG),
        training_value_loss=_read_scalar(accumulator, tags, TRAINING_VALUE_LOSS_TAG),
        optimizer_steps=_read_scalar(accumulator, tags, OPTIMIZER_STEPS_TAG),
    )


def _find_evaluation_directory(archive: Path) -> Path | None:
    candidates = sorted(path for path in archive.rglob('evaluations') if path.is_dir())
    return candidates[0] if candidates else None


def _load_from_evaluation_artifacts(label: str, archive: Path, evaluation_directory: Path) -> RunSeries:
    accuracy: list[tuple[float, float]] = []
    cross_entropy: list[tuple[float, float]] = []
    level_zero: list[tuple[float, float]] = []
    for result_path in sorted(evaluation_directory.glob('*.json')):
        match = EVALUATION_FILE_PATTERN.match(result_path.name)
        if match is None:
            continue
        boundary = float(match.group('boundary'))
        definition = match.group('definition')
        result = json.loads(result_path.read_text(encoding='utf-8'))
        match definition:
            case 'fixed-dataset':
                accuracy.append((boundary, float(result['top_action_accuracy'])))
                cross_entropy.append((boundary, float(result['policy_cross_entropy'])))
            case 'stockfish-level-0':
                level_zero.append((boundary, float(result['aggregate']['score'])))
            case _:
                continue
    empty = ScalarSeries(steps=(), values=())
    return RunSeries(
        label=label,
        archive_name=archive.name,
        ladder_elo=empty,
        level_zero_score=_deduplicate_by_step(level_zero),
        fixed_dataset_accuracy=_deduplicate_by_step(accuracy),
        fixed_dataset_cross_entropy=_deduplicate_by_step(cross_entropy),
        training_policy_loss=empty,
        training_value_loss=empty,
        optimizer_steps=empty,
    )


def load_run_series(archive: LabelledArchive) -> RunSeries:
    if not archive.path.is_dir():
        raise ValueError(f'Run archive {archive.path} does not exist.')
    event_directory = _find_coordinator_event_directory(archive.path)
    if event_directory is not None:
        return _load_from_tensorboard(archive.label, archive.path, event_directory)
    evaluation_directory = _find_evaluation_directory(archive.path)
    if evaluation_directory is None:
        raise ValueError(
            f'Run archive {archive.path} has neither a coordinator TensorBoard directory nor evaluation artifacts.'
        )
    return _load_from_evaluation_artifacts(archive.label, archive.path, evaluation_directory)


def load_reference_curve(path: Path, label: str) -> ReferenceCurve:
    hours: list[float] = []
    level_zero: list[float] = []
    accuracy: list[float] = []
    cross_entropy: list[float] = []
    with path.open(encoding='utf-8', newline='') as handle:
        for row in DictReader(handle):
            hours.append(float(row['h']))
            level_zero.append(float(row['sf0']))
            accuracy.append(float(row['acc']))
            cross_entropy.append(float(row['ce']))
    if not hours:
        raise ValueError(f'Reference curve {path} contains no rows.')
    return ReferenceCurve(
        label=label,
        hours=tuple(hours),
        level_zero_score=tuple(level_zero),
        fixed_dataset_accuracy=tuple(accuracy),
        fixed_dataset_cross_entropy=tuple(cross_entropy),
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            'figure.dpi': 110,
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 10,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.25,
            'grid.linewidth': 0.6,
            'legend.frameon': False,
            'legend.fontsize': 9,
            'lines.linewidth': 1.8,
            'lines.markersize': 4,
            'savefig.bbox': 'tight',
            'svg.fonttype': 'path',
            'svg.hashsalt': 'alphazero-showcase',
        }
    )


def _plot_series(axes: Axes, x_values: Sequence[float], y_values: Sequence[float], label: str, color: str) -> None:
    axes.plot(x_values, y_values, marker='o', color=color, label=label)


def _plot_reference(axes: Axes, hours: Sequence[float], values: Sequence[float], label: str, limit: float) -> None:
    within_window = [index for index, hour in enumerate(hours) if hour <= limit]
    if not within_window:
        return
    axes.plot(
        [hours[index] for index in within_window],
        [values[index] for index in within_window],
        linestyle='--',
        marker='s',
        color=REFERENCE_COLOR,
        label=label,
    )


def _annotate_provenance(figure: Figure, runs: Sequence[RunSeries], reference: ReferenceCurve | None) -> None:
    sources = [f'{run.label}: {run.archive_name}' for run in runs]
    if reference is not None:
        sources.append(f'reference: {reference.label}')
    figure.text(0.0, -0.04, '\n'.join(sources), fontsize=7, color='#555555', ha='left', va='top', linespacing=1.5)


def _write_figure(figure: Figure, output_path: Path) -> None:
    # Dropping the creation date, with the fixed svg.hashsalt, keeps re-renders byte-identical in the tree.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format='svg', metadata={'Date': None})
    plt.close(figure)


def render_strength_figure(run: RunSeries, reference: ReferenceCurve | None, output_path: Path) -> None:
    figure, (elo_axes, score_axes) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    if run.ladder_elo.is_empty:
        raise ValueError(f'Run {run.label} carries no ladder Elo scalars; the strength figure needs them.')
    _plot_series(elo_axes, run.ladder_elo.hours, run.ladder_elo.values, run.label, SERIES_COLORS[0])
    elo_axes.set_title('Stockfish-ladder Elo')
    elo_axes.set_xlabel('wall-clock hours since run start')
    elo_axes.set_ylabel('fitted ladder Elo')
    elo_axes.legend(loc='lower right')

    _plot_series(score_axes, run.level_zero_score.hours, run.level_zero_score.values, run.label, SERIES_COLORS[0])
    if reference is not None:
        horizon = max(run.level_zero_score.hours) if not run.level_zero_score.is_empty else 0.0
        _plot_reference(score_axes, reference.hours, reference.level_zero_score, reference.label, horizon)
    score_axes.set_title('Score vs Stockfish level 0 (100 games, 64 visits)')
    score_axes.set_xlabel('wall-clock hours since run start')
    score_axes.set_ylabel('score')
    score_axes.set_ylim(0.0, 1.0)
    score_axes.legend(loc='upper left')

    _annotate_provenance(figure, (run,), reference)
    _write_figure(figure, output_path)


def render_training_figure(run: RunSeries, reference: ReferenceCurve | None, output_path: Path) -> None:
    figure, (loss_axes, entropy_axes) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    if run.training_policy_loss.is_empty:
        raise ValueError(f'Run {run.label} carries no training loss scalars.')
    steps = run.optimizer_steps.values if not run.optimizer_steps.is_empty else run.training_policy_loss.steps
    common = min(len(steps), len(run.training_policy_loss.values))
    _plot_series(loss_axes, steps[:common], run.training_policy_loss.values[:common], 'policy loss', SERIES_COLORS[0])
    if not run.training_value_loss.is_empty:
        value_common = min(len(steps), len(run.training_value_loss.values))
        _plot_series(
            loss_axes, steps[:value_common], run.training_value_loss.values[:value_common], 'WDL loss', SERIES_COLORS[1]
        )
    loss_axes.set_title(f'Training loss — {run.label}')
    loss_axes.set_xlabel('optimizer steps')
    loss_axes.set_ylabel('loss')
    loss_axes.legend(loc='upper right')

    _plot_series(
        entropy_axes,
        run.fixed_dataset_cross_entropy.hours,
        run.fixed_dataset_cross_entropy.values,
        run.label,
        SERIES_COLORS[0],
    )
    if reference is not None:
        horizon = max(run.fixed_dataset_cross_entropy.hours) if not run.fixed_dataset_cross_entropy.is_empty else 0.0
        _plot_reference(entropy_axes, reference.hours, reference.fixed_dataset_cross_entropy, reference.label, horizon)
    entropy_axes.set_title('Policy cross-entropy on the fixed reference dataset')
    entropy_axes.set_xlabel('wall-clock hours since run start')
    entropy_axes.set_ylabel('cross-entropy (nats)')
    entropy_axes.legend(loc='upper right')

    _annotate_provenance(figure, (run,), reference)
    _write_figure(figure, output_path)


def render_comparison_figure(runs: Sequence[RunSeries], reference: ReferenceCurve | None, output_path: Path) -> None:
    if not runs:
        raise ValueError('The comparison figure needs at least one run.')
    figure, axes = plt.subplots(1, 1, figsize=(8.0, 4.6))
    horizon = 0.0
    for index, run in enumerate(runs):
        if run.fixed_dataset_accuracy.is_empty:
            raise ValueError(f'Run {run.label} carries no fixed-dataset accuracy scalars.')
        hours = run.fixed_dataset_accuracy.hours
        horizon = max(horizon, max(hours))
        _plot_series(
            axes, hours, run.fixed_dataset_accuracy.values, run.label, SERIES_COLORS[index % len(SERIES_COLORS)]
        )
    if reference is not None:
        _plot_reference(axes, reference.hours, reference.fixed_dataset_accuracy, reference.label, horizon)
    axes.set_title('Experiment ladder — policy top-1 accuracy on the fixed reference dataset')
    axes.set_xlabel('wall-clock hours since run start')
    axes.set_ylabel('top-1 accuracy')
    axes.legend(loc='upper left')

    _annotate_provenance(figure, runs, reference)
    _write_figure(figure, output_path)


def _parse_labelled_archive(argument: str) -> LabelledArchive:
    label, separator, path = argument.partition('=')
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f'Expected label=path, got {argument!r}.')
    return LabelledArchive(label=label, path=Path(path))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render the showcase plots from fetched run archives.')
    parser.add_argument(
        '--primary',
        type=_parse_labelled_archive,
        required=True,
        help='label=path of the run archive that carries the headline strength and training figures.',
    )
    parser.add_argument(
        '--comparison',
        type=_parse_labelled_archive,
        action='append',
        default=[],
        help='label=path of a run archive for the experiment-ladder comparison figure; repeatable.',
    )
    parser.add_argument(
        '--reference',
        type=Path,
        default=None,
        help='Wall-clock yardstick CSV (columns h, acc, ce, sf0) overlaid as the reference curve.',
    )
    parser.add_argument(
        '--reference-label',
        type=str,
        default='four-day r3/r4 reference',
        help='Legend label for the reference curve.',
    )
    parser.add_argument('--output-directory', type=Path, default=Path('documentation/showcase'))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    _apply_style()

    primary = load_run_series(arguments.primary)
    comparison = [load_run_series(archive) for archive in arguments.comparison]
    reference = (
        load_reference_curve(arguments.reference, arguments.reference_label)
        if arguments.reference is not None
        else None
    )

    output_directory: Path = arguments.output_directory
    render_strength_figure(primary, reference, output_directory / 'chess-strength-vs-wall-clock.svg')
    render_training_figure(primary, reference, output_directory / 'chess-training-loss.svg')
    if comparison:
        render_comparison_figure(comparison, reference, output_directory / 'chess-experiment-ladder-comparison.svg')

    for name in sorted(path.name for path in output_directory.glob('*.svg')):
        print(f'{output_directory / name}')


if __name__ == '__main__':
    main()
