"""Extract selected-checkpoint diagnostics and render the report's appendix figures.

This replaces the exploratory renderer in the retention-fix worktree. Training series
join the frozen trajectory by run and generation; evaluation curves use the existing
stitched export and stop at 2.5 days. No later continuation enters these figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tools.export_final_training_trajectory import TrainingObservation
from tools.render_final_training_dynamics import configure_style, read_points, render_volume_paper, save_figure, smooth

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'documentation/evidence/final-chess-20260923'
FIGURES = ROOT / 'documentation/report/figures'
ARCHIVE_SHA256 = '72936c566026c3065cc07c9ce5d234a89dcb4cefd0efd035cea0db9e44e1c910'
BLUE, TEAL, ORANGE, GRAY = '#1f5875', '#1f7065', '#a45d1e', '#647b88'
FINAL_STEPS = 408_500
FINAL_SECONDS = 216_000


class Metric(StrEnum):
    NEXT_POLICY = 'training_auxiliary/0-next-policy-ply-1/loss'
    REMAINING_LENGTH = 'training_auxiliary/1-remaining-game-length/loss'
    GRADIENT = 'training/gradient_norm'
    VISITS = 'settings/self_play/baseline_visits'
    REPLAY_AGE = 'replay_diagnostics/age_seconds_mean'
    THRESHOLD = 'resignation/selected_threshold'
    SAFE = 'resignation/selected_threshold_safe'
    FALSE_RATE = 'resignation/false_nonloss_rate'
    UPPER_BOUND = 'resignation/false_nonloss_upper_bound'
    SAVED_PLIES = 'resignation/average_saved_plies'


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)


class Sample(FrozenModel):
    optimizer_steps: int
    value: float


class DiagnosticSeries(FrozenModel):
    metric: Metric
    samples: tuple[Sample, ...]


class Diagnostics(FrozenModel):
    archive_sha256: str
    trajectory_sha256: str
    selected_optimizer_steps: int
    series: tuple[DiagnosticSeries, ...]

    def samples(self, metric: Metric) -> tuple[Sample, ...]:
        return next(series.samples for series in self.series if series.metric == metric)


class LadderPoint(FrozenModel):
    seconds: int
    raw_seconds: int | None = None
    elo: float


class LadderSeries(FrozenModel):
    source: str
    points: tuple[LadderPoint, ...]


class LadderExport(FrozenModel):
    note: str
    series: dict[str, LadderSeries]


def read_trajectory(path: Path) -> tuple[TrainingObservation, ...]:
    adapter = TypeAdapter(TrainingObservation)
    with path.open(encoding='utf-8', newline='') as source:
        points = tuple(adapter.validate_python(row) for row in csv.DictReader(source))
    if not points or points[-1].optimizer_steps != FINAL_STEPS:
        raise ValueError('Expected the frozen trajectory through 408,500 optimizer steps.')
    return points


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def extract_diagnostics(archive_path: Path, trajectory_path: Path) -> Diagnostics:
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError('TensorBoard archive does not match the frozen evidence hash.')
    trajectory = read_trajectory(trajectory_path)
    collected: dict[Metric, list[Sample]] = {metric: [] for metric in Metric}
    with TemporaryDirectory() as temporary_name, tarfile.open(archive_path, 'r:gz') as archive:
        for run in dict.fromkeys(point.source_run for point in trajectory):
            prefix = f'tensorboard/{run}/coordinator/events.out.tfevents.'
            members = [member for member in archive.getmembers() if member.name.startswith(prefix)]
            if len(members) != 1:
                raise ValueError(f'Expected one coordinator event file for {run}.')
            source = archive.extractfile(members[0])
            if source is None:
                raise ValueError(f'Cannot read {members[0].name}.')
            event_path = Path(temporary_name) / f'{run}.tfevents'
            with source, event_path.open('wb') as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
            accumulator = EventAccumulator(str(event_path), size_guidance={'scalars': 0}).Reload()
            available = frozenset(accumulator.Tags()['scalars'])
            selected = {
                point.tensorboard_step: point.optimizer_steps for point in trajectory if point.source_run == run
            }
            for metric in Metric:
                if metric not in available:
                    raise ValueError(f'Missing diagnostic {metric} in {run}.')
                values = {event.step: float(event.value) for event in accumulator.Scalars(metric)}
                for generation, steps in selected.items():
                    if generation in values:
                        collected[metric].append(Sample(optimizer_steps=steps, value=values[generation]))
                    elif metric not in {Metric.THRESHOLD, Metric.SAVED_PLIES}:
                        raise ValueError(f'Missing {metric} at {run} generation {generation}.')
    series = tuple(
        DiagnosticSeries(metric=metric, samples=tuple(sorted(samples, key=lambda sample: sample.optimizer_steps)))
        for metric, samples in collected.items()
    )
    for item in series:
        steps = [sample.optimizer_steps for sample in item.samples]
        if not steps or len(steps) != len(set(steps)):
            raise ValueError(f'Empty or duplicated diagnostic: {item.metric}.')
    return Diagnostics(
        archive_sha256=ARCHIVE_SHA256,
        trajectory_sha256=digest(trajectory_path),
        selected_optimizer_steps=FINAL_STEPS,
        series=series,
    )


def axis_style(axes: Axes, label: str) -> None:
    axes.set_ylabel(label)
    axes.grid(axis='y', color='#c9d3dc', linewidth=0.7)
    axes.set_axisbelow(True)
    axes.spines[['top', 'right']].set_visible(False)
    axes.spines[['left', 'bottom']].set_color(GRAY)
    axes.tick_params(length=0, pad=5)


def training_axis(axes: Axes, label: str) -> None:
    axis_style(axes, label)
    axes.set_xlim(0, FINAL_STEPS / 1000)
    axes.axvline(240, color=GRAY, linestyle='--', linewidth=0.9)


def line(axes: Axes, samples: tuple[Sample, ...], color: str, *, scale: float = 1.0) -> None:
    x = [sample.optimizer_steps / 1000 for sample in samples]
    values = [sample.value / scale for sample in samples]
    axes.plot(x, values, color=color, alpha=0.3, linewidth=0.6)
    axes.plot(x, smooth(values), color=color, linewidth=1.4)


def training_figure(rows: int) -> tuple[Figure, list[Axes]]:
    figure, array = plt.subplots(rows, 1, figsize=(6.4, 2.05 * rows), sharex=True, squeeze=False)
    axes = [row[0] for row in array]
    figure.subplots_adjust(left=0.19, right=0.96, top=0.97, bottom=0.14 if rows == 2 else 0.10, hspace=0.17)
    return figure, axes


def render_auxiliary(diagnostics: Diagnostics) -> None:
    figure, axes = horizontal_training_figure(('Next-policy loss', 'Remaining-length loss'))
    figure.subplots_adjust(top=0.88, wspace=0.45)
    line(axes[0], diagnostics.samples(Metric.NEXT_POLICY), BLUE)
    line(axes[1], diagnostics.samples(Metric.REMAINING_LENGTH), TEAL)
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    save_figure(figure, FIGURES / 'appendix-auxiliary-losses.svg')


def render_gradient(diagnostics: Diagnostics) -> None:
    figure, axes = plt.subplots(figsize=(6.4, 2.7))
    figure.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.24)
    training_axis(axes, 'Pre-clip gradient norm')
    line(axes, diagnostics.samples(Metric.GRADIENT), BLUE)
    axes.axhline(1, color=ORANGE, linestyle=':', label='Clip threshold: 1.0', linewidth=1.2)
    axes.legend(frameon=False, fontsize=10, loc='upper left')
    save_figure(figure, FIGURES / 'appendix-gradient-norm.svg')


def render_stages(trajectory: tuple[TrainingObservation, ...], diagnostics: Diagnostics) -> None:
    figure, axes = training_figure(3)
    x = [point.optimizer_steps / 1000 for point in trajectory]
    training_axis(axes[0], 'Trainer samples/s (k)')
    samples = tuple(Sample(optimizer_steps=p.optimizer_steps, value=p.training_samples_per_second) for p in trajectory)
    line(axes[0], samples, BLUE, scale=1000)
    training_axis(axes[1], 'Search visits / move')
    visits = diagnostics.samples(Metric.VISITS)
    axes[1].step([p.optimizer_steps / 1000 for p in visits], [p.value for p in visits], where='post', color=ORANGE)
    training_axis(axes[2], 'Active network')
    axes[2].step(x, [p.active_model_index for p in trajectory], where='post', color=TEAL)
    axes[2].set_yticks([0, 1], ['Small\n12 x 128', 'Medium\n14 x 160'])
    axes[2].set_ylim(-0.2, 1.2)
    save_figure(figure, FIGURES / 'appendix-training-stages.svg')


def render_replay(trajectory: tuple[TrainingObservation, ...], diagnostics: Diagnostics) -> None:
    figure, axes = horizontal_training_figure(('Replay positions (M)', 'Mean sampled age (h)'))
    x = [p.optimizer_steps / 1000 for p in trajectory]
    axes[0].plot(x, [p.replay_live_rows / 1e6 for p in trajectory], color=BLUE, label='Occupied')
    axes[0].step(
        x, [p.replay_capacity / 1e6 for p in trajectory], where='post', color=ORANGE, linestyle=':', label='Capacity'
    )
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    line(axes[1], diagnostics.samples(Metric.REPLAY_AGE), TEAL, scale=3600)
    save_figure(figure, FIGURES / 'appendix-replay-age.svg')


def render_resignation(diagnostics: Diagnostics) -> None:
    figure, axes = horizontal_training_figure(('Resignation threshold', 'False non-loss (%)', 'Mean saved plies'))
    figure.subplots_adjust(bottom=0.38)
    threshold = diagnostics.samples(Metric.THRESHOLD)
    axes[0].plot([p.optimizer_steps / 1000 for p in threshold], [p.value for p in threshold], color=BLUE, linewidth=1)
    safe_steps = {p.optimizer_steps for p in diagnostics.samples(Metric.SAFE) if p.value == 1}
    for metric, color, label in ((Metric.FALSE_RATE, TEAL, 'Observed'), (Metric.UPPER_BOUND, BLUE, '95% upper bound')):
        samples = diagnostics.samples(metric)
        x = [p.optimizer_steps / 1000 for p in samples]
        values = [100 * p.value if p.optimizer_steps in safe_steps else float('nan') for p in samples]
        axes[1].plot(x, values, color=color, alpha=0.3, linewidth=0.6)
        axes[1].plot(x, smooth(values), color=color, label=label, linewidth=1.3)
    axes[1].axhline(2.5, color=ORANGE, linestyle=':', linewidth=1.2, label='Safety limit')
    axes[1].set_ylim(0, 3.8)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=3)
    line(axes[2], diagnostics.samples(Metric.SAVED_PLIES), TEAL)
    save_figure(figure, FIGURES / 'appendix-resignation.svg')


def horizontal_training_figure(labels: tuple[str, ...]) -> tuple[Figure, list[Axes]]:
    figure, array = plt.subplots(1, len(labels), figsize=(7, 2.6), squeeze=False)
    axes = list(array[0])
    figure.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.25, wspace=0.38)
    for axis, label in zip(axes, labels, strict=True):
        training_axis(axis, '')
        axis.set_xticks([0, 200, 400])
        axis.tick_params(labelsize=8.5)
        axis.set_title(label, fontsize=10, pad=10)
    return figure, axes


def render_ladder(path: Path) -> None:
    export = LadderExport.model_validate_json(path.read_text(encoding='utf-8'))
    figure, axes = plt.subplots(figsize=(6.4, 3.1))
    figure.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.20)
    axis_style(axes, 'Training ladder Elo')
    for key, label, color in (
        ('final::evaluation/ladder_elo_64', '64 searches', BLUE),
        ('final::evaluation/ladder_elo_1', 'Policy only', TEAL),
    ):
        points = [p for p in export.series[key].points if p.seconds <= FINAL_SECONDS]
        x, values = [p.seconds / 86400 for p in points], [p.elo for p in points]
        axes.plot(x, values, color=color, linewidth=0.6, alpha=0.3)
        axes.plot(x, smooth(values), color=color, linewidth=1.5, label=label)
    axes.set_xlim(0, 2.5)
    axes.set_xlabel('Effective training time (days)')
    axes.legend(frameon=False, fontsize=10)
    save_figure(figure, FIGURES / 'appendix-policy-search-progress.svg')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Extract fresh diagnostics from the frozen TensorBoard archive.')
    arguments = parser.parse_args()
    trajectory_path = EVIDENCE / 'training-trajectory.csv'
    diagnostics_path = EVIDENCE / 'training-diagnostics.json'
    if arguments.archive is not None:
        diagnostics = extract_diagnostics(arguments.archive, trajectory_path)
        diagnostics_path.write_text(diagnostics.model_dump_json(indent=2) + '\n', encoding='utf-8')
    else:
        diagnostics = Diagnostics.model_validate_json(diagnostics_path.read_text(encoding='utf-8'))
    if diagnostics.trajectory_sha256 != digest(trajectory_path):
        raise ValueError('Diagnostic export and training trajectory differ.')
    configure_style(paper=True)
    trajectory = read_trajectory(trajectory_path)
    render_volume_paper(
        read_points(trajectory_path),
        tuple((sample.optimizer_steps, sample.value) for sample in diagnostics.samples(Metric.VISITS)),
    )
    render_auxiliary(diagnostics)
    render_gradient(diagnostics)
    render_stages(trajectory, diagnostics)
    render_replay(trajectory, diagnostics)
    render_resignation(diagnostics)
    render_ladder(EVIDENCE / 'ladder-elo-export.json')
    print('Rendered six appendix figures from selected-checkpoint evidence.')


if __name__ == '__main__':
    main()
