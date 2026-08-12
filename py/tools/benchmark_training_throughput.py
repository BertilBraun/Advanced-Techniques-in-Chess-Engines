from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import time

from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.configuration import TrainingCompilation, TrainingPrecision
from src.training.progress import TrainingProgress
from src.training.trainer import TrainerGroup
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


@dataclass(frozen=True)
class Arguments:
    resolved_configuration: Path
    run_directory: Path
    checkpoint_generation: int
    device_ids: tuple[int, ...]
    optimizer_steps: int
    quantum_count: int
    global_batch_size: int | None
    precision: TrainingPrecision
    compilation: TrainingCompilation
    output_directory: Path


class TrainingThroughputBenchmarkResult(FrozenModel):
    checkpoint_generation: int
    device_ids: tuple[int, ...]
    world_size: int
    global_batch_size: int
    local_batch_size: int
    optimizer_steps: int
    quantum_count: int
    precision: TrainingPrecision
    compilation: TrainingCompilation
    replay_rows: int
    initialization_and_training_seconds: float
    training_quantum_seconds: float
    replay_rows_per_second: float
    training_samples_per_second: float
    output_checkpoint_generation: int


def _stage_checkpoint(source: CheckpointReference, destination: Path) -> CheckpointReference:
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_checkpoint_manifest(source.generation, source.manifest_path.parent)
    for source_path in (source.model_path, source.optimizer_path, source.inference_model_path):
        shutil.copy2(source_path, destination / source_path.name)
    shutil.copy2(source.manifest_path, destination / source.manifest_path.name)
    staged = CheckpointReference.load(destination, manifest.generation)
    return staged


def run_benchmark(arguments: Arguments) -> TrainingThroughputBenchmarkResult:
    configuration = load_experiment_configuration(arguments.resolved_configuration)
    world_size = len(arguments.device_ids)
    global_batch_size = arguments.global_batch_size or configuration.training.trainer.global_batch_size
    if global_batch_size % world_size:
        raise ValueError('Global batch size must divide evenly over the requested devices.')

    benchmark_directory = arguments.output_directory / f'ddp-{world_size}-gpu-batch-{global_batch_size}'
    checkpoint = _stage_checkpoint(
        CheckpointReference.load(arguments.run_directory, arguments.checkpoint_generation),
        benchmark_directory,
    )
    trainer = configuration.training.trainer.validated_copy(
        update={
            'global_batch_size': global_batch_size,
            'local_batch_size': global_batch_size // world_size,
            'precision': arguments.precision,
            'compilation': arguments.compilation,
        }
    )
    trainer_topology = configuration.training.topology.trainer.validated_copy(
        update={
            'rank_zero_device_id': arguments.device_ids[0],
            'ddp_device_ids': arguments.device_ids,
        }
    )
    topology = configuration.training.topology.validated_copy(
        update={'trainer': trainer_topology.model_dump(mode='json')}
    )
    credit = configuration.training.lifecycle.credit.validated_copy(
        update={
            'optimizer_steps_per_quantum': arguments.optimizer_steps,
            'maximum_optimizer_steps': arguments.optimizer_steps * 1_000_000,
        }
    )
    lifecycle = configuration.training.lifecycle.validated_copy(update={'credit': credit.model_dump(mode='json')})
    training = configuration.training.validated_copy(
        update={
            'save_path': str(benchmark_directory),
            'trainer': trainer.model_dump(mode='json'),
            'topology': topology.model_dump(mode='json'),
            'lifecycle': lifecycle.model_dump(mode='json'),
        }
    )
    benchmark_configuration = configuration.validated_copy(update={'training': training.model_dump(mode='json')})
    game = create_game_implementation(benchmark_configuration)
    replay_layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=benchmark_configuration.training.lifecycle.replay.maximum_policy_entries,
    )
    replay_store = ReplayStore.open(arguments.run_directory / 'replay.bin', replay_layout, writable=False)
    replay_state = replay_store.state
    replay = ReplayDescription(
        path=replay_store.path,
        head=replay_state.head,
        size=replay_state.size,
        logical_capacity=replay_state.logical_capacity,
        maximum_capacity=replay_state.maximum_capacity,
        layout=replay_layout,
    )
    replay_store.close()

    started_at = time.perf_counter()
    trainer_group = TrainerGroup(benchmark_configuration, game, checkpoint)
    try:
        progress = TrainingProgress(
            completed_optimizer_steps=arguments.checkpoint_generation * arguments.optimizer_steps,
            optimizer_steps_per_generation=arguments.optimizer_steps,
        )
        for _ in range(arguments.quantum_count):
            result = trainer_group.train_quantum(replay, progress)
            progress = progress.next_generation
    finally:
        trainer_group.close()
    elapsed_seconds = time.perf_counter() - started_at
    benchmark_result = TrainingThroughputBenchmarkResult(
        checkpoint_generation=arguments.checkpoint_generation,
        device_ids=arguments.device_ids,
        world_size=world_size,
        global_batch_size=global_batch_size,
        local_batch_size=global_batch_size // world_size,
        optimizer_steps=arguments.optimizer_steps,
        quantum_count=arguments.quantum_count,
        precision=arguments.precision,
        compilation=arguments.compilation,
        replay_rows=replay.size,
        initialization_and_training_seconds=elapsed_seconds,
        training_quantum_seconds=result.statistics.elapsed_seconds,
        replay_rows_per_second=result.statistics.replay_rows_per_second,
        training_samples_per_second=result.statistics.training_samples_per_second,
        output_checkpoint_generation=result.checkpoint.generation,
    )
    write_text_atomically(
        benchmark_directory / 'result.json',
        benchmark_result.model_dump_json(indent=2) + '\n',
    )
    return benchmark_result


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Benchmark the production mapped-replay DDP training path.')
    parser.add_argument('--resolved-configuration', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--device-ids', required=True, nargs='+', type=int)
    parser.add_argument('--optimizer-steps', default=200, type=int)
    parser.add_argument('--quantum-count', default=1, type=int)
    parser.add_argument('--global-batch-size', type=int)
    parser.add_argument(
        '--precision',
        choices=tuple(precision.value for precision in TrainingPrecision),
        required=True,
        type=TrainingPrecision,
    )
    parser.add_argument(
        '--compilation',
        choices=tuple(compilation.value for compilation in TrainingCompilation),
        required=True,
        type=TrainingCompilation,
    )
    parser.add_argument('--output-directory', required=True, type=Path)
    namespace = parser.parse_args()
    arguments = Arguments(
        resolved_configuration=namespace.resolved_configuration,
        run_directory=namespace.run_directory,
        checkpoint_generation=namespace.checkpoint_generation,
        device_ids=tuple(namespace.device_ids),
        optimizer_steps=namespace.optimizer_steps,
        quantum_count=namespace.quantum_count,
        global_batch_size=namespace.global_batch_size,
        precision=namespace.precision,
        compilation=namespace.compilation,
        output_directory=namespace.output_directory,
    )
    if not arguments.resolved_configuration.is_file():
        raise ValueError(f'Resolved configuration does not exist: {arguments.resolved_configuration}')
    if not arguments.run_directory.is_dir():
        raise ValueError(f'Run directory does not exist: {arguments.run_directory}')
    if (
        arguments.checkpoint_generation < 0
        or arguments.optimizer_steps <= 0
        or arguments.quantum_count <= 0
        or (arguments.global_batch_size is not None and arguments.global_batch_size <= 0)
    ):
        raise ValueError('Checkpoint generation must be nonnegative and benchmark sizes must be positive.')
    if not arguments.device_ids or len(set(arguments.device_ids)) != len(arguments.device_ids):
        raise ValueError('Device IDs must be nonempty and unique.')
    if any(device_id < 0 for device_id in arguments.device_ids):
        raise ValueError('Device IDs must be nonnegative.')
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    return arguments


def main() -> None:
    print(run_benchmark(parse_arguments()).model_dump_json())


if __name__ == '__main__':
    main()
