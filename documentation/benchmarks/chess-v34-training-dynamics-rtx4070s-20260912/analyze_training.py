from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent

EMA_DECAY = 0.95
FINAL_GENERATION = 1465
HEADLINE_BILLED_HOURS = 72.0
NODE_HOURLY_PRICE = 17.36 / 24
MILESTONE_HOURS = (1, 2, 4, 8, 16, 32, 64, 72)


@dataclass(frozen=True)
class Milestone:
    requested_hour: int
    recorded_hour: float
    generation: int
    ladder_elo_1_raw: float
    ladder_elo_1_ema: float
    ladder_elo_64_raw: float
    ladder_elo_64_ema: float
    top_action_accuracy_raw: float
    top_action_accuracy_ema: float


@dataclass(frozen=True)
class DoublingReturn:
    start_hour: int
    end_hour: int
    additional_gpu_hours: int
    ladder_elo_1_gain: float
    ladder_elo_64_gain: float
    ladder_elo_1_gain_per_hour: float
    ladder_elo_64_gain_per_hour: float


@dataclass(frozen=True)
class Phase:
    name: str
    start_generation: int
    end_generation: int
    model: str
    visits: int
    learning_rate: float


@dataclass(frozen=True)
class PhaseThroughput:
    name: str
    generations: int
    duration_hours: float
    generations_per_hour: float
    games: int
    games_per_hour: float
    fresh_positions: int
    fresh_positions_per_second: float
    estimated_searches: int


PHASES = (
    Phase('small, 300 visits', 1, 9, '12x128', 300, 0.005),
    Phase('small, 400 visits', 10, 49, '12x128', 400, 0.005),
    Phase('small, 500 visits', 50, 89, '12x128', 500, 0.005),
    Phase('small, 600 visits, initial LR', 90, 99, '12x128', 600, 0.005),
    Phase('small, steady 600 visits', 100, 598, '12x128', 600, 0.004),
    Phase('candidate catch-up', 599, 633, '12x128 + 14x160 candidate', 600, 0.004),
    Phase('medium, 600 visits, high LR', 634, 799, '14x160', 600, 0.004),
    Phase('medium, 600 visits', 800, 999, '14x160', 600, 0.003),
    Phase('medium, 800 visits', 1000, 1219, '14x160', 800, 0.003),
    Phase('medium, terminal LR', 1220, FINAL_GENERATION, '14x160', 800, 0.001),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export v34 training dynamics from a preserved TensorBoard run.')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def smoothed_values(events: list[ScalarEvent], decay: float = EMA_DECAY) -> list[float]:
    numerator = 0.0
    denominator = 0.0
    values: list[float] = []
    for event in events:
        numerator = numerator * decay + event.value
        denominator = denominator * decay + 1.0
        values.append(numerator / denominator)
    return values


def event_nearest_hour(events: list[ScalarEvent], hour: float) -> tuple[int, ScalarEvent]:
    index = min(range(len(events)), key=lambda candidate: abs(events[candidate].step / 3600 - hour))
    return index, events[index]


def values_by_step(events: list[ScalarEvent]) -> dict[int, float]:
    return {event.step: event.value for event in events}


def scalar_at_generation(events: list[ScalarEvent], generation: int) -> float:
    matching = next((event.value for event in events if event.step == generation), None)
    if matching is None:
        raise ValueError(f'No scalar was recorded at generation {generation}.')
    return matching


def generation_hours(accumulator: EventAccumulator) -> tuple[np.ndarray, np.ndarray]:
    generation_by_elapsed_step = values_by_step(accumulator.Scalars('evaluation_metadata/progress/model_generation'))
    pairs = sorted((generation, step / 3600) for step, generation in generation_by_elapsed_step.items())
    generations = np.asarray([0.0, *(pair[0] for pair in pairs)], dtype=np.float64)
    hours = np.asarray([0.0, *(pair[1] for pair in pairs)], dtype=np.float64)
    unique_generations, unique_indices = np.unique(generations, return_index=True)
    return unique_generations, hours[unique_indices]


def hour_at_generation(generation: int, generations: np.ndarray, hours: np.ndarray) -> float:
    return float(np.interp(generation, generations, hours))


def build_milestones(accumulator: EventAccumulator) -> list[Milestone]:
    elo_1_events = accumulator.Scalars('evaluation/ladder_elo_1')
    elo_64_events = accumulator.Scalars('evaluation/ladder_elo_64')
    accuracy_events = accumulator.Scalars('evaluation/fixed-dataset/top_action_accuracy')
    generation_by_step = values_by_step(accumulator.Scalars('evaluation_metadata/progress/model_generation'))
    elo_1_ema = smoothed_values(elo_1_events)
    elo_64_ema = smoothed_values(elo_64_events)
    accuracy_ema = smoothed_values(accuracy_events)
    elo_1_by_step = {event.step: (event.value, elo_1_ema[index]) for index, event in enumerate(elo_1_events)}
    accuracy_by_step = {event.step: (event.value, accuracy_ema[index]) for index, event in enumerate(accuracy_events)}

    milestones: list[Milestone] = []
    for hour in MILESTONE_HOURS:
        elo_64_index, elo_64_event = event_nearest_hour(elo_64_events, hour)
        elo_1_raw, elo_1_smoothed = elo_1_by_step[elo_64_event.step]
        accuracy_raw, accuracy_smoothed = accuracy_by_step[elo_64_event.step]
        milestones.append(
            Milestone(
                requested_hour=hour,
                recorded_hour=elo_64_event.step / 3600,
                generation=int(generation_by_step[elo_64_event.step]),
                ladder_elo_1_raw=elo_1_raw,
                ladder_elo_1_ema=elo_1_smoothed,
                ladder_elo_64_raw=elo_64_event.value,
                ladder_elo_64_ema=elo_64_ema[elo_64_index],
                top_action_accuracy_raw=accuracy_raw,
                top_action_accuracy_ema=accuracy_smoothed,
            )
        )
    return milestones


def build_doubling_returns(milestones: list[Milestone]) -> list[DoublingReturn]:
    by_hour = {milestone.requested_hour: milestone for milestone in milestones}
    intervals = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
    results: list[DoublingReturn] = []
    for start_hour, end_hour in intervals:
        start = by_hour[start_hour]
        end = by_hour[end_hour]
        duration = end_hour - start_hour
        elo_1_gain = end.ladder_elo_1_ema - start.ladder_elo_1_ema
        elo_64_gain = end.ladder_elo_64_ema - start.ladder_elo_64_ema
        results.append(
            DoublingReturn(
                start_hour=start_hour,
                end_hour=end_hour,
                additional_gpu_hours=duration * 8,
                ladder_elo_1_gain=elo_1_gain,
                ladder_elo_64_gain=elo_64_gain,
                ladder_elo_1_gain_per_hour=elo_1_gain / duration,
                ladder_elo_64_gain_per_hour=elo_64_gain / duration,
            )
        )
    return results


def build_phase_throughput(accumulator: EventAccumulator) -> list[PhaseThroughput]:
    games = accumulator.Scalars('self_play/completed_games')
    fresh_positions = accumulator.Scalars('credit/materialized_samples')
    generations, hours = generation_hours(accumulator)
    results: list[PhaseThroughput] = []
    for phase in PHASES:
        start_hour = hour_at_generation(max(phase.start_generation - 1, 0), generations, hours)
        end_hour = hour_at_generation(phase.end_generation, generations, hours)
        duration_hours = end_hour - start_hour
        phase_games = round(
            sum(event.value for event in games if phase.start_generation <= event.step <= phase.end_generation)
        )
        prior_positions = (
            scalar_at_generation(fresh_positions, phase.start_generation - 1) if phase.start_generation > 1 else 0
        )
        final_positions = scalar_at_generation(fresh_positions, phase.end_generation)
        phase_positions = round(final_positions - prior_positions)
        results.append(
            PhaseThroughput(
                name=phase.name,
                generations=phase.end_generation - phase.start_generation + 1,
                duration_hours=duration_hours,
                generations_per_hour=(phase.end_generation - phase.start_generation + 1) / duration_hours,
                games=phase_games,
                games_per_hour=phase_games / duration_hours,
                fresh_positions=phase_positions,
                fresh_positions_per_second=phase_positions / (duration_hours * 3600),
                estimated_searches=phase_positions * phase.visits,
            )
        )
    return results


def write_csv(path: Path, rows: list[object]) -> None:
    dictionaries = [asdict(row) for row in rows]
    with path.open('w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=dictionaries[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(dictionaries)


def style_axis(axis: Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.spines[['top', 'right']].set_visible(False)


def draw_elo_figure(accumulator: EventAccumulator, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, tag, label, colour in (
        (axes[0], 'evaluation/ladder_elo_64', '64-search ladder Elo', '#2563eb'),
        (axes[1], 'evaluation/ladder_elo_1', 'policy-only ladder Elo', '#059669'),
    ):
        events = accumulator.Scalars(tag)
        hours = [event.step / 3600 for event in events]
        raw = [event.value for event in events]
        axis.plot(hours, raw, color=colour, alpha=0.22, linewidth=1, label='100-game evaluation')
        axis.plot(hours, smoothed_values(events), color=colour, linewidth=2.5, label='0.95 EMA')
        axis.axvline(18.5, color='#d97706', linestyle='--', linewidth=1, label='candidate training starts')
        axis.axvline(20.0, color='#7c3aed', linestyle='--', linewidth=1, label='medium model active')
        axis.axvline(72.0, color='#111827', linestyle=':', linewidth=1.5, label='three-day checkpoint')
        style_axis(axis, label)
    axes[0].legend(ncol=4, frameon=False, fontsize=9)
    axes[1].set_xlabel('Cumulative training time (hours)')
    figure.suptitle('v34 playing strength during training')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_accuracy_figure(accumulator: EventAccumulator, output_path: Path) -> None:
    events = accumulator.Scalars('evaluation/fixed-dataset/top_action_accuracy')
    hours = [event.step / 3600 for event in events]
    figure: Figure
    axis: Axes
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    axis.plot(hours, [event.value * 100 for event in events], color='#9333ea', alpha=0.22, linewidth=1)
    axis.plot(hours, [value * 100 for value in smoothed_values(events)], color='#9333ea', linewidth=2.5)
    axis.axvline(72.0, color='#111827', linestyle=':', linewidth=1.5)
    axis.set_xlabel('Cumulative training time (hours)')
    style_axis(axis, 'Top-action accuracy (%)')
    axis.set_title('Fixed-dataset policy accuracy')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_throughput_figure(accumulator: EventAccumulator, output_path: Path) -> None:
    games = accumulator.Scalars('self_play/completed_games')
    generations, elapsed_hours = generation_hours(accumulator)
    generation_values = np.asarray([event.step for event in games], dtype=np.float64)
    game_values = np.asarray([event.value for event in games], dtype=np.float64)
    hours = np.interp(generation_values, generations, elapsed_hours)
    window = 40
    rolling_games = np.convolve(game_values, np.ones(window), mode='valid')
    rolling_hours = hours[window - 1 :] - np.concatenate(([0.0], hours[:-window]))
    games_per_hour = rolling_games / rolling_hours
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(hours, generation_values, color='#2563eb', linewidth=2)
    style_axis(axes[0], 'Generation')
    axes[1].plot(hours[window - 1 :], games_per_hour, color='#dc2626', linewidth=1.8)
    style_axis(axes[1], f'Self-play games/hour ({window}-generation window)')
    axes[1].set_xlabel('Cumulative training time (hours)')
    for axis in axes:
        axis.axvline(18.5, color='#d97706', linestyle='--', linewidth=1)
        axis.axvline(20.0, color='#7c3aed', linestyle='--', linewidth=1)
        axis.axvline(72.0, color='#111827', linestyle=':', linewidth=1.5)
    figure.suptitle('v34 generation and self-play throughput')
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    coordinator = arguments.archive / 'tensorboard' / 'coordinator'
    if not coordinator.is_dir():
        raise ValueError(f'Coordinator TensorBoard directory does not exist: {coordinator}')
    arguments.output.mkdir(parents=True, exist_ok=True)
    accumulator = EventAccumulator(str(coordinator), size_guidance={'scalars': 0}).Reload()
    milestones = build_milestones(accumulator)
    doubling_returns = build_doubling_returns(milestones)
    phase_throughput = build_phase_throughput(accumulator)
    generations, elapsed_hours = generation_hours(accumulator)
    final_recorded_hours = hour_at_generation(FINAL_GENERATION, generations, elapsed_hours)

    games = accumulator.Scalars('self_play/completed_games')
    fresh_positions = accumulator.Scalars('credit/materialized_samples')
    consumed_presentations = accumulator.Scalars('credit/consumed_presentations')
    final_games = round(sum(event.value for event in games if event.step <= FINAL_GENERATION))
    final_positions = round(scalar_at_generation(fresh_positions, FINAL_GENERATION))
    final_presentations = round(scalar_at_generation(consumed_presentations, FINAL_GENERATION))
    summary = {
        'final_generation': FINAL_GENERATION,
        'optimizer_steps': FINAL_GENERATION * 500,
        'tensorboard_elapsed_hours': final_recorded_hours,
        'headline_billed_hours': HEADLINE_BILLED_HOURS,
        'self_play_games': final_games,
        'self_play_games_per_hour': final_games / final_recorded_hours,
        'self_play_games_per_second': final_games / (final_recorded_hours * 3600),
        'fresh_positions': final_positions,
        'fresh_positions_per_hour': final_positions / final_recorded_hours,
        'training_presentations': final_presentations,
        'empirical_reuse_factor': final_presentations / final_positions,
        'recorded_gpu_hours': final_recorded_hours * 8,
        'headline_gpu_hours': HEADLINE_BILLED_HOURS * 8,
        'headline_node_cost_usd': HEADLINE_BILLED_HOURS * NODE_HOURLY_PRICE,
        'estimated_searches': sum(phase.estimated_searches for phase in phase_throughput),
    }
    (arguments.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    write_csv(arguments.output / 'milestones.csv', milestones)
    write_csv(arguments.output / 'compute-doubling.csv', doubling_returns)
    write_csv(arguments.output / 'phase-throughput.csv', phase_throughput)
    draw_elo_figure(accumulator, arguments.output / 'elo-vs-hours.png')
    draw_accuracy_figure(accumulator, arguments.output / 'top-action-accuracy.png')
    draw_throughput_figure(accumulator, arguments.output / 'throughput.png')


if __name__ == '__main__':
    main()
