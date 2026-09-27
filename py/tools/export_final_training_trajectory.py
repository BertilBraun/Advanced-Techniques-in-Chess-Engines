"""Export the accepted final chess lineage from frozen coordinator TensorBoard events."""

from __future__ import annotations

import argparse
import csv
import tarfile
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from tempfile import TemporaryDirectory

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent

RUN_NAMES = (
    'vast-chess-8gpu-v89-v35-progressive-int8-sgd-reuse4-plateau',
    'vast-chess-8gpu-v90-resume-catchup-schedule',
    'vast-chess-8gpu-v91-resume-recalibrated-int8',
    'vast-chess-8gpu-v92-real-position-fidelity',
    'vast-chess-8gpu-v93-latch-reset',
)


@dataclass(frozen=True)
class TrainingObservation:
    source_run: str
    tensorboard_step: int
    wall_time_unix: float
    optimizer_steps: int
    completed_games: int
    materialized_positions: int
    consumed_presentations: int
    replay_live_rows: int
    replay_capacity: int
    active_model_index: int
    policy_loss: float
    wdl_loss: float
    total_loss: float
    learning_rate: float
    training_samples_per_second: float
    quantum_duration_seconds: float
    credit_wait_seconds: float


def scalar_value(accumulator: EventAccumulator, tag: str, index: int, step: int) -> float:
    events: list[ScalarEvent] = accumulator.Scalars(tag)
    if len(events) <= index or events[index].step != step:
        raise ValueError(f'Missing or misaligned {tag} at TensorBoard step {step}.')
    return float(events[index].value)


def load_run(
    archive: tarfile.TarFile,
    source_run: str,
    temporary_directory: Path,
) -> list[TrainingObservation]:
    prefix = f'tensorboard/{source_run}/coordinator/events.out.tfevents.'
    members = [member for member in archive.getmembers() if member.name.startswith(prefix)]
    if len(members) != 1:
        raise ValueError(f'Expected one coordinator event file for {source_run}; found {len(members)}.')
    source = archive.extractfile(members[0])
    if source is None:
        raise ValueError(f'Cannot read coordinator event file for {source_run}.')
    event_path = temporary_directory / f'{source_run}.tfevents'
    with event_path.open('wb') as destination:
        while block := source.read(1024 * 1024):
            destination.write(block)

    accumulator = EventAccumulator(str(event_path)).Reload()
    optimizer_events: list[ScalarEvent] = accumulator.Scalars('training/optimizer_steps')
    observations: list[TrainingObservation] = []
    for index, event in enumerate(optimizer_events):
        step = int(event.step)
        observations.append(
            TrainingObservation(
                source_run=source_run,
                tensorboard_step=step,
                wall_time_unix=float(event.wall_time),
                optimizer_steps=round(event.value),
                completed_games=round(scalar_value(accumulator, 'self_play/completed_games', index, step)),
                materialized_positions=round(scalar_value(accumulator, 'credit/materialized_samples', index, step)),
                consumed_presentations=round(scalar_value(accumulator, 'credit/consumed_presentations', index, step)),
                replay_live_rows=round(scalar_value(accumulator, 'replay/live_rows', index, step)),
                replay_capacity=round(scalar_value(accumulator, 'replay/logical_capacity', index, step)),
                active_model_index=round(scalar_value(accumulator, 'progressive/active_model_index', index, step)),
                policy_loss=scalar_value(accumulator, 'training/policy_loss', index, step),
                wdl_loss=scalar_value(accumulator, 'training/wdl_loss', index, step),
                total_loss=scalar_value(accumulator, 'training/total_loss', index, step),
                learning_rate=scalar_value(accumulator, 'training/learning_rate', index, step),
                training_samples_per_second=scalar_value(
                    accumulator, 'throughput/training_samples_per_second', index, step
                ),
                quantum_duration_seconds=scalar_value(accumulator, 'training/quantum_duration_seconds', index, step),
                credit_wait_seconds=scalar_value(accumulator, 'credit/wait_seconds', index, step),
            )
        )
    return observations


def export_trajectory(archive_path: Path, output_path: Path, selected_optimizer_steps: int) -> None:
    with TemporaryDirectory() as temporary_name, tarfile.open(archive_path, mode='r:gz') as archive:
        temporary_directory = Path(temporary_name)
        observations = [
            observation
            for source_run in RUN_NAMES
            for observation in load_run(archive, source_run, temporary_directory)
            if observation.optimizer_steps <= selected_optimizer_steps
        ]
    observations.sort(key=lambda observation: observation.tensorboard_step)
    expected_steps = list(range(1, len(observations) + 1))
    if [observation.tensorboard_step for observation in observations] != expected_steps:
        raise ValueError('Accepted lineage has missing or duplicated TensorBoard steps.')
    if not observations or observations[-1].optimizer_steps != selected_optimizer_steps:
        raise ValueError('Selected checkpoint optimizer step is absent from the coordinator events.')
    if any(observation.optimizer_steps != observation.tensorboard_step * 500 for observation in observations):
        raise ValueError('Optimizer step and 500-step quantum identities disagree.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow([field.name for field in fields(TrainingObservation)])
        writer.writerows(astuple(observation) for observation in observations)
    selected = observations[-1]
    print(f'quanta={len(observations)} completed_games={sum(row.completed_games for row in observations)}')
    print(
        f'materialized_positions={selected.materialized_positions} '
        f'presentations={selected.consumed_presentations} '
        f'replay_live_rows={selected.replay_live_rows}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--selected-optimizer-steps', type=int, default=408500)
    arguments = parser.parse_args()
    export_trajectory(arguments.archive, arguments.output, arguments.selected_optimizer_steps)


if __name__ == '__main__':
    main()
