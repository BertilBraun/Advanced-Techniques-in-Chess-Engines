"""Renders the technical report's figures from a fetched TensorBoard bundle.

Every figure records the runs and tags it was built from in a manifest beside it, so a reader can
tie a curve back to archived evidence instead of trusting the picture. Series that the report asks
for but which the runs never logged are reported as missing rather than approximated.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use('svg')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

# The reported lineage in order. V90 and V96 are excluded: both were reverted, so their boundaries
# describe discarded work and would consume the time axis of a curve that never benefited from them.
LINEAGE: tuple[str, ...] = (
    'vast-chess-8gpu-v89-v35-progressive-int8-sgd-reuse4-plateau',
    'vast-chess-8gpu-v91-resume-recalibrated-int8',
    'vast-chess-8gpu-v92-real-position-fidelity',
    'vast-chess-8gpu-v93-latch-reset',
    'vast-chess-8gpu-v94-windowed-plateau',
    'vast-chess-8gpu-v95-longer-catchup',
    'vast-chess-8gpu-v97-revert-promotion',
    'vast-chess-8gpu-v99-grown-promotion',
)

COMPARISONS: tuple[tuple[str, str], ...] = (
    ('v9', 'vast-chess-4day-production-v9'),
    ('v29', 'vast-chess-4day-production-v29'),
    ('v34', 'vast-chess-8gpu-integrated-v34'),
    ('v46', 'vast-chess-8gpu-integrated-v46-v35-g16-restart'),
)

EVALUATION_CADENCE_SECONDS = 1200
SERIES_COLORS = ('#1f4e79', '#c1440e', '#2e7d32', '#7b4fa8', '#b8860b', '#0f766e')
LINEAGE_COLOR = '#1f4e79'
MARKER_COLOR = '#9aa0a6'


@dataclass
class FigureRecord:
    name: str
    title: str
    runs: tuple[str, ...]
    tags: tuple[str, ...]
    missing_tags: tuple[str, ...] = ()
    note: str = ''


@dataclass
class Manifest:
    source_bundle: str
    figures: list[FigureRecord] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            'source_bundle': self.source_bundle,
            'lineage': list(LINEAGE),
            'excluded_from_lineage': [
                'vast-chess-8gpu-v90-resume-catchup-schedule',
                'vast-chess-8gpu-v96-candidate-multiplier',
            ],
            'figures': [
                {
                    'name': record.name,
                    'title': record.title,
                    'runs': list(record.runs),
                    'tags': list(record.tags),
                    'missing_tags': list(record.missing_tags),
                    'note': record.note,
                }
                for record in self.figures
            ],
        }
        return json.dumps(payload, indent=2) + '\n'


RunScalars = dict[str, dict[float, float]]

_memory_cache: dict[str, RunScalars] = {}
_cache_directory: Path | None = None


def _parse_run(root: Path, run: str) -> RunScalars:
    """Reads every scalar tag of a run in one pass; reopening event files per tag is minutes per run."""
    tags: RunScalars = {}
    for path in sorted((root / run).glob('*/events.out.tfevents.*')):
        accumulator = EventAccumulator(str(path), size_guidance={'scalars': 0})
        try:
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get('scalars', []):
            points = tags.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                points[float(event.step)] = float(event.value)
    return tags


def _read_run(root: Path, run: str) -> RunScalars:
    cached = _memory_cache.get(run)
    if cached is not None:
        return cached
    cache_file = _cache_directory / f'{run}.json' if _cache_directory else None
    if cache_file is not None and cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding='utf-8'))
        tags = {tag: {float(step): value for step, value in points.items()} for tag, points in raw.items()}
    else:
        tags = _parse_run(root, run)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(tags), encoding='utf-8')
    _memory_cache[run] = tags
    return tags


def load_scalars(root: Path, run: str, tag: str) -> dict[float, float]:
    return dict(_read_run(root, run).get(tag, {}))


def lineage_series(root: Path, tag: str) -> tuple[list[float], list[float], list[tuple[float, str]]]:
    """Merges a tag across the lineage, returning x, y and the segment boundaries for markers."""
    merged: list[tuple[float, float, str]] = []
    for run in LINEAGE:
        for step, value in load_scalars(root, run, tag).items():
            merged.append((step, value, run))
    merged.sort(key=lambda row: row[0])
    xs = [row[0] for row in merged]
    ys = [row[1] for row in merged]
    boundaries: list[tuple[float, str]] = []
    seen: set[str] = set()
    for step, _, run in merged:
        if run not in seen:
            seen.add(run)
            boundaries.append((step, run))
    return xs, ys, boundaries[1:]


def stitched_evaluation_series(root: Path, tag: str) -> tuple[list[float], list[float], list[tuple[float, str]]]:
    """Evaluation boundaries restart per run, so the axis is the evaluation index times the cadence."""
    merged: list[tuple[float, float, str]] = []
    for run in LINEAGE:
        for step, value in load_scalars(root, run, tag).items():
            merged.append((step, value, run))
    merged.sort(key=lambda row: row[0])
    xs = [(index + 1) * EVALUATION_CADENCE_SECONDS / 3600 for index in range(len(merged))]
    ys = [row[1] for row in merged]
    boundaries: list[tuple[float, str]] = []
    seen: set[str] = set()
    for index, (_, _, run) in enumerate(merged):
        if run not in seen:
            seen.add(run)
            boundaries.append((xs[index], run))
    return xs, ys, boundaries[1:]


def apply_style() -> None:
    plt.rcParams.update(
        {
            'figure.figsize': (9.0, 5.0),
            'figure.dpi': 120,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.size': 10,
            'legend.frameon': False,
            'savefig.bbox': 'tight',
        }
    )


def mark_segments(axes: Axes, boundaries: Iterable[tuple[float, str]]) -> None:
    for index, (position, run) in enumerate(boundaries):
        axes.axvline(position, color=MARKER_COLOR, linewidth=0.8, linestyle=':', zorder=0)
        label = run.split('-')[3] if len(run.split('-')) > 3 else run
        axes.annotate(
            label.upper(),
            xy=(position, 1.0),
            xycoords=('data', 'axes fraction'),
            xytext=(2, -10 - 9 * (index % 2)),
            textcoords='offset points',
            fontsize=7,
            color=MARKER_COLOR,
        )


def save(figure: plt.Figure, output: Path, name: str) -> Path:
    path = output / f'{name}.svg'
    figure.savefig(path, format='svg', metadata={'Date': None})
    plt.close(figure)
    return path


def plot_lines(axes: Axes, series: Sequence[tuple[str, list[float], list[float]]]) -> None:
    for index, (label, xs, ys) in enumerate(series):
        if not xs:
            continue
        axes.plot(xs, ys, label=label, color=SERIES_COLORS[index % len(SERIES_COLORS)], linewidth=1.4)


def figure_cross_lineage(root: Path, output: Path, manifest: Manifest) -> None:
    figure, axes = plt.subplots()
    series: list[tuple[str, list[float], list[float]]] = []
    for label, run in COMPARISONS:
        points = load_scalars(root, run, 'evaluation/ladder_elo_64')
        steps = sorted(points)
        series.append((label, [step / 3600 for step in steps], [points[step] for step in steps]))
    xs, ys, boundaries = stitched_evaluation_series(root, 'evaluation/ladder_elo_64')
    plot_lines(axes, series)
    axes.plot(xs, ys, label='final lineage', color=LINEAGE_COLOR, linewidth=2.0)
    mark_segments(axes, boundaries)
    axes.set_xlabel('effective training time (hours, evaluation boundaries)')
    axes.set_ylabel('ladder Elo at 64 searches')
    axes.set_title('64-search ladder Elo across chess lineages')
    axes.legend(loc='lower right')
    axes.annotate(
        'v9, v29, v34 and v46 log a single-rung fit; the final lineage logs a three-rung bracketed fit.\n'
        'The estimators are not interchangeable: see the result record for the matched comparison.',
        xy=(0.01, 0.02),
        xycoords='axes fraction',
        fontsize=7,
        color='#444444',
    )
    save(figure, output, '01-cross-lineage-ladder-elo')
    manifest.figures.append(
        FigureRecord(
            name='01-cross-lineage-ladder-elo',
            title='64-search ladder Elo across chess lineages',
            runs=tuple(run for _, run in COMPARISONS) + LINEAGE,
            tags=('evaluation/ladder_elo_64',),
            note='Final lineage x-axis is the stitched evaluation index times the 1200 s cadence.',
        )
    )


def figure_losses(root: Path, output: Path, manifest: Manifest) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    used: list[str] = []
    for index, (tag, label) in enumerate(
        (('training/total_loss', 'total'), ('training/policy_loss', 'policy'), ('training/wdl_loss', 'WDL'))
    ):
        xs, ys, _ = lineage_series(root, tag)
        if not xs:
            continue
        used.append(tag)
        left.plot(xs, ys, label=label, color=SERIES_COLORS[index], linewidth=1.2)
    missing: list[str] = []
    for index, (tag, label) in enumerate(
        (
            ('training_auxiliary/0-next-policy-ply-1/loss', 'next policy'),
            ('training_auxiliary/1-remaining-game-length/loss', 'remaining game length'),
        )
    ):
        xs, ys, _ = lineage_series(root, tag)
        if not xs:
            missing.append(tag)
            continue
        used.append(tag)
        right.plot(xs, ys, label=label, color=SERIES_COLORS[index + 3], linewidth=1.2)
    for axes, title in ((left, 'primary losses'), (right, 'auxiliary losses')):
        axes.set_xlabel('optimizer steps')
        axes.set_ylabel('loss')
        axes.set_title(title)
        axes.legend()
    figure.suptitle('Training losses across the final lineage')
    save(figure, output, '02-training-losses')
    manifest.figures.append(
        FigureRecord(
            name='02-training-losses',
            title='Training losses across the final lineage',
            runs=LINEAGE,
            tags=tuple(used),
            missing_tags=tuple(missing),
            note='x-axis is optimizer steps, which run continuously across resumes.',
        )
    )


def figure_optimization(root: Path, output: Path, manifest: Manifest) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    lr_x, lr_y, _ = lineage_series(root, 'training/learning_rate')
    left.plot(lr_x, lr_y, color=SERIES_COLORS[0], linewidth=1.4)
    left.set_xlabel('optimizer steps')
    left.set_ylabel('learning rate')
    left.set_title('learning rate')
    norm_x, norm_y, _ = lineage_series(root, 'training/gradient_norm')
    right.plot(norm_x, norm_y, color=SERIES_COLORS[1], linewidth=0.8)
    right.axhline(1.0, color=MARKER_COLOR, linewidth=0.8, linestyle='--')
    right.set_xlabel('optimizer steps')
    right.set_ylabel('gradient norm')
    right.set_title('gradient norm against the 1.0 clip')
    figure.suptitle('Optimization schedule and gradient scale')
    save(figure, output, '03-optimization')
    manifest.figures.append(
        FigureRecord(
            name='03-optimization',
            title='Optimization schedule and gradient scale',
            runs=LINEAGE,
            tags=('training/learning_rate', 'training/gradient_norm'),
            missing_tags=('clipped step fraction',),
            note='No clipped-step fraction was ever logged; the norm against the configured 1.0 cap is the proxy.',
        )
    )


def figure_ladder_detail(root: Path, output: Path, manifest: Manifest) -> None:
    figure, axes = plt.subplots()
    searched_x, searched_y, boundaries = stitched_evaluation_series(root, 'evaluation/ladder_elo_64')
    single_x, single_y, _ = stitched_evaluation_series(root, 'evaluation/ladder_elo_single_rung_64')
    policy_x, policy_y, _ = stitched_evaluation_series(root, 'evaluation/ladder_elo_1')
    axes.plot(searched_x, searched_y, label='64 searches, three-rung', color=SERIES_COLORS[0], linewidth=1.6)
    axes.plot(single_x, single_y, label='64 searches, single rung', color=SERIES_COLORS[1], linewidth=0.9, alpha=0.7)
    axes.plot(policy_x, policy_y, label='policy only, three-rung', color=SERIES_COLORS[2], linewidth=1.4)
    mark_segments(axes, boundaries)
    axes.set_xlabel('effective training time (hours)')
    axes.set_ylabel('ladder Elo')
    axes.set_title('Final lineage ladder Elo, both estimators')
    axes.legend(loc='lower right')
    save(figure, output, '04-final-lineage-ladder')
    manifest.figures.append(
        FigureRecord(
            name='04-final-lineage-ladder',
            title='Final lineage ladder Elo, both estimators',
            runs=LINEAGE,
            tags=('evaluation/ladder_elo_64', 'evaluation/ladder_elo_single_rung_64', 'evaluation/ladder_elo_1'),
            note='The single-rung series is drawn beside the bracketed one so the estimator gap is visible.',
        )
    )


def figure_volume(root: Path, output: Path, manifest: Manifest) -> None:
    figure, axes = plt.subplots()
    used: list[str] = []
    missing: list[str] = []
    for index, (tag, label) in enumerate(
        (
            ('self_play/completed_games', 'completed self-play games'),
            ('credit/materialized_samples', 'materialized positions'),
            ('credit/consumed_presentations', 'training presentations'),
            ('training/optimizer_steps', 'optimizer steps'),
        )
    ):
        xs, ys, boundaries = lineage_series(root, tag)
        if not xs:
            missing.append(tag)
            continue
        used.append(tag)
        axes.plot(xs, ys, label=label, color=SERIES_COLORS[index], linewidth=1.3)
    axes.set_yscale('log')
    axes.set_xlabel('step (tag native)')
    axes.set_ylabel('cumulative count, log scale')
    axes.set_title('Training volume across the final lineage')
    axes.legend(loc='lower right')
    save(figure, output, '05-training-volume')
    manifest.figures.append(
        FigureRecord(
            name='05-training-volume',
            title='Training volume across the final lineage',
            runs=LINEAGE,
            tags=tuple(used),
            missing_tags=tuple(missing),
            note='Counts are cumulative per run; resume boundaries restart some counters, so read within a segment.',
        )
    )


def figure_throughput(root: Path, output: Path, manifest: Manifest) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    xs, ys, _ = lineage_series(root, 'throughput/training_samples_per_second')
    left.plot(xs, ys, color=SERIES_COLORS[0], linewidth=0.9)
    left.set_xlabel('step')
    left.set_ylabel('training samples per second')
    left.set_title('trainer throughput')
    visits_x, visits_y, _ = lineage_series(root, 'settings/self_play/baseline_visits')
    right.step(visits_x, visits_y, where='post', color=SERIES_COLORS[1], linewidth=1.4)
    right.set_xlabel('model generation')
    right.set_ylabel('baseline visits')
    right.set_title('self-play visit budget')
    figure.suptitle('Throughput and the search budget that drives it')
    save(figure, output, '06-throughput')
    manifest.figures.append(
        FigureRecord(
            name='06-throughput',
            title='Throughput and the search budget that drives it',
            runs=LINEAGE,
            tags=('throughput/training_samples_per_second', 'settings/self_play/baseline_visits'),
            note='Self-play throughput was not logged directly; the visit budget is the setting that governs it.',
        )
    )


def figure_replay(root: Path, output: Path, manifest: Manifest) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    rows_x, rows_y, _ = lineage_series(root, 'replay/live_rows')
    capacity_x, capacity_y, _ = lineage_series(root, 'replay/logical_capacity')
    left.plot(rows_x, rows_y, label='live rows', color=SERIES_COLORS[0], linewidth=1.3)
    left.plot(capacity_x, capacity_y, label='logical capacity', color=MARKER_COLOR, linewidth=1.0, linestyle='--')
    left.set_xlabel('step')
    left.set_ylabel('rows')
    left.set_title('replay occupancy against capacity')
    left.legend()
    age_x, age_y, _ = lineage_series(root, 'replay_diagnostics/generation_age_mean')
    right.plot(age_x, age_y, color=SERIES_COLORS[2], linewidth=1.1)
    right.set_xlabel('step')
    right.set_ylabel('mean generation age of sampled rows')
    right.set_title('replay age')
    figure.suptitle('Replay occupancy and staleness')
    save(figure, output, '07-replay')
    manifest.figures.append(
        FigureRecord(
            name='07-replay',
            title='Replay occupancy and staleness',
            runs=LINEAGE,
            tags=(
                'replay/live_rows',
                'replay/logical_capacity',
                'replay_diagnostics/generation_age_mean',
            ),
            note='The replay store itself was not retained; these are the scalars logged while it lived.',
        )
    )


def figure_promotion(root: Path, output: Path, manifest: Manifest) -> None:
    figure, axes = plt.subplots()
    active_x, active_y, boundaries = lineage_series(root, 'progressive/active_model_index')
    axes.step(active_x, active_y, where='post', color=SERIES_COLORS[0], linewidth=1.6, label='active stage index')
    axes.set_xlabel('model generation')
    axes.set_ylabel('active progressive stage')
    axes.set_yticks([0, 1, 2])
    axes.set_yticklabels(['12x128', '14x160', '19x176'])
    twin = axes.twinx()
    gain_x, gain_y, _ = lineage_series(root, 'progressive/candidate_start/instantaneous_ema_gain_per_hour')
    twin.plot(gain_x, gain_y, color=SERIES_COLORS[1], linewidth=0.9, alpha=0.8, label='Elo gain per hour')
    twin.set_ylabel('instantaneous Elo gain per hour')
    twin.grid(False)
    axes.set_title('Progressive stage and the plateau signal that drives promotion')
    save(figure, output, '08-promotion')
    manifest.figures.append(
        FigureRecord(
            name='08-promotion',
            title='Progressive stage and the plateau signal that drives promotion',
            runs=LINEAGE,
            tags=('progressive/active_model_index', 'progressive/candidate_start/instantaneous_ema_gain_per_hour'),
            note='Replaces the INT8 fidelity figure the chapter requested: no fidelity series was ever logged.',
        )
    )


def figure_resignation(root: Path, output: Path, manifest: Manifest) -> None:
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(15.0, 4.4))
    threshold_x, threshold_y, _ = lineage_series(root, 'resignation/selected_threshold')
    left.plot(threshold_x, threshold_y, color=SERIES_COLORS[0], linewidth=1.3)
    left.set_xlabel('step')
    left.set_ylabel('selected threshold')
    left.set_title('resignation threshold')
    rate_x, rate_y, _ = lineage_series(root, 'resignation/false_nonloss_rate')
    bound_x, bound_y, _ = lineage_series(root, 'resignation/false_nonloss_upper_bound')
    middle.plot(rate_x, rate_y, label='false non-loss rate', color=SERIES_COLORS[1], linewidth=1.1)
    middle.plot(bound_x, bound_y, label='upper bound', color=MARKER_COLOR, linewidth=0.9, linestyle='--')
    middle.set_xlabel('step')
    middle.set_ylabel('rate')
    middle.set_title('resignation safety')
    middle.legend()
    resignations_x, resignations_y, _ = lineage_series(root, 'resignation/actual_resignations')
    continuations_x, continuations_y, _ = lineage_series(root, 'resignation/continuation_games')
    right.plot(resignations_x, resignations_y, label='resignations', color=SERIES_COLORS[2], linewidth=1.1)
    right.plot(continuations_x, continuations_y, label='audited continuations', color=SERIES_COLORS[3], linewidth=1.1)
    right.set_xlabel('step')
    right.set_ylabel('games')
    right.set_title('trigger volume')
    right.legend()
    saved_x, saved_y, _ = lineage_series(root, 'resignation/average_saved_plies')
    saved = right.twinx()
    saved.plot(saved_x, saved_y, color=SERIES_COLORS[4], linewidth=0.9, alpha=0.8)
    saved.set_ylabel('mean saved plies per resignation')
    saved.grid(False)
    figure.suptitle('Resignation calibration, volume and saved search')
    save(figure, output, '09-resignation')
    manifest.figures.append(
        FigureRecord(
            name='09-resignation',
            title='Resignation calibration, volume and saved search',
            runs=LINEAGE,
            tags=(
                'resignation/selected_threshold',
                'resignation/false_nonloss_rate',
                'resignation/false_nonloss_upper_bound',
                'resignation/actual_resignations',
                'resignation/continuation_games',
                'resignation/average_saved_plies',
            ),
            note='Saved plies is the logged per-resignation mean; the search saved in node-seconds was never logged.',
        )
    )


FIGURES = (
    figure_cross_lineage,
    figure_losses,
    figure_optimization,
    figure_ladder_detail,
    figure_volume,
    figure_throughput,
    figure_replay,
    figure_promotion,
    figure_resignation,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--tensorboard-root',
        type=Path,
        required=True,
        help='Directory holding the extracted run directories, normally <bundle>/tensorboard.',
    )
    parser.add_argument('--output-directory', type=Path, default=Path('documentation/report/figures'))
    parser.add_argument(
        '--cache-directory',
        type=Path,
        help='Parsed scalars are stored here so a re-render does not reparse hundreds of megabytes of event files.',
    )
    parser.add_argument(
        '--source-bundle',
        default='.codex-diagnostics/final-2026-09-23/evidence-tensorboard.tgz',
        help='Recorded in the manifest so a figure ties back to archived evidence.',
    )
    return parser.parse_args()


def main() -> None:
    global _cache_directory
    arguments = parse_arguments()
    if not arguments.tensorboard_root.is_dir():
        raise SystemExit(f'TensorBoard root does not exist: {arguments.tensorboard_root}')
    _cache_directory = arguments.cache_directory
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    apply_style()
    manifest = Manifest(source_bundle=arguments.source_bundle)
    for render in FIGURES:
        render(arguments.tensorboard_root, arguments.output_directory, manifest)
        print(f'rendered {manifest.figures[-1].name}')
    (arguments.output_directory / 'figures-manifest.json').write_text(manifest.to_json(), encoding='utf-8')
    print(f'{len(manifest.figures)} figures and a manifest in {arguments.output_directory}')


if __name__ == '__main__':
    main()
