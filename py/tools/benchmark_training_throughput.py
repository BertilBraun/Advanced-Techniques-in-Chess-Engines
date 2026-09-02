from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.replay.store import ReplayStore
from src.search_stopping.calibration import StopDecisionReason, StopPolicyPublication
from src.search_stopping.policy import closed_policy
from src.self_play.protocol import RunningSelfPlayState, StatisticsLevel
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.configuration import TrainingCompilation, TrainingPrecision
from src.training.progress import TrainingProgress
from src.training.self_play_group import SelfPlayGroup
from src.training.trainer import TrainerGroup
from src.training.trainer.contracts import TrainerQuantum, TrainerStartup
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
    self_play_workers: int


@dataclass(frozen=True)
class _DeviceLoad:
    utilization_percent: float
    power_watts: float


class _DeviceLoadSampler:
    """nvidia-smi rather than a pynvml dependency; one sample a second is plenty for a minute-long window."""

    def __init__(self) -> None:
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(1.0):
            try:
                completed = subprocess.run(
                    ('nvidia-smi', '--query-gpu=utilization.gpu,power.draw', '--format=csv,noheader,nounits'),
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=True,
                )
            except (subprocess.SubprocessError, OSError):
                continue
            for line in completed.stdout.strip().splitlines():
                utilization, power = line.split(',')
                self._samples.append((float(utilization), float(power)))

    def __enter__(self) -> _DeviceLoadSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15.0)

    def result(self) -> _DeviceLoad:
        if not self._samples:
            return _DeviceLoad(0.0, 0.0)
        return _DeviceLoad(
            statistics.fmean(utilization for utilization, _ in self._samples),
            statistics.fmean(power for _, power in self._samples),
        )


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
    self_play_workers: int
    completed_searches: int
    measured_quantum_seconds: float
    searches_per_second: float
    mean_gpu_utilization_percent: float
    mean_gpu_power_watts: float


def _stage_checkpoint(source: CheckpointReference, destination: Path) -> CheckpointReference:
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_checkpoint_manifest(source.generation, source.manifest_path.parent)
    for source_path in (source.model_path, source.optimizer_path, source.inference_model_path):
        shutil.copy2(source_path, destination / source_path.name)
    shutil.copy2(source.manifest_path, destination / source.manifest_path.name)
    staged = CheckpointReference.load(destination, manifest.generation)
    return staged


def _paused_worker_ids(device_ids: tuple[int, ...], running_count: int) -> tuple[int, ...]:
    """Spread the running workers over the GPUs the way production does. DDP runs at the speed of its
    slowest rank, so leaving consecutive worker IDs running piles them onto the first GPUs and the
    trainer measures that skew rather than the self-play load."""
    by_device: dict[int, list[int]] = {}
    for worker_id, device_id in enumerate(device_ids):
        by_device.setdefault(device_id, []).append(worker_id)
    ordered: list[int] = []
    for slot in range(max(len(workers) for workers in by_device.values())):
        for device_id in sorted(by_device):
            if slot < len(by_device[device_id]):
                ordered.append(by_device[device_id][slot])
    return tuple(sorted(ordered[running_count:]))


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
        maximum_legal_actions=game.state.maximum_legal_action_count,
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

    (benchmark_directory / 'completed-games' / 'inbox').mkdir(parents=True, exist_ok=True)
    stop_publication = StopPolicyPublication(
        policy=closed_policy(benchmark_configuration.training.lifecycle.search_stopping),
        application_generation=0,
        decision_reason=StopDecisionReason.INITIAL,
    )
    self_play_group = SelfPlayGroup(game) if arguments.self_play_workers else None
    paused_worker_ids = (
        ()
        if self_play_group is None
        else _paused_worker_ids(
            benchmark_configuration.training.topology.self_play.device_ids, arguments.self_play_workers
        )
    )

    def activate(active_checkpoint: CheckpointReference, collect: bool) -> int:
        """Mirror the production generation boundary: resume every worker, then pause the complement."""
        if self_play_group is None:
            return 0
        responses = self_play_group.apply(
            tuple(
                RunningSelfPlayState(
                    checkpoint=active_checkpoint,
                    search_stopping=stop_publication,
                    completed_generation_statistics=StatisticsLevel.BASIC if collect else None,
                )
                for _ in range(self_play_group.worker_count)
            )
        )
        self_play_group.request_pause(paused_worker_ids)
        return sum(
            response.completed_generation_statistics.completed_searches
            for response in responses
            if response.kind == 'running' and response.completed_generation_statistics is not None
        )

    started_at = time.perf_counter()
    trainer_group = TrainerGroup(
        benchmark_configuration,
        game,
        TrainerStartup(
            network=benchmark_configuration.training.initial_model.network,
            save_path=benchmark_directory,
            starting_generation=checkpoint.generation,
        ),
    )
    try:
        progress = TrainingProgress(
            completed_optimizer_steps=arguments.checkpoint_generation * arguments.optimizer_steps,
            optimizer_steps_per_generation=arguments.optimizer_steps,
        )
        activate(checkpoint, collect=False)
        # The first quantum absorbs compilation, cold caches and self-play model load; only the last is measured.
        for quantum_index in range(arguments.quantum_count):
            measured = quantum_index == arguments.quantum_count - 1
            sampler = _DeviceLoadSampler() if measured else None
            quantum_started_at = time.perf_counter()
            with sampler or nullcontext():
                result = trainer_group.train_quantum(
                    TrainerQuantum(replay=replay, model_progress=progress, replay_source_progress=progress)
                )
            quantum_seconds = time.perf_counter() - quantum_started_at
            completed_searches = activate(result.checkpoint, collect=True)
            progress = progress.next_generation
    finally:
        trainer_group.close()
        if self_play_group is not None:
            self_play_group.close()
    elapsed_seconds = time.perf_counter() - started_at
    device_load = sampler.result() if sampler is not None else _DeviceLoad(0.0, 0.0)
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
        replay_rows_per_second=result.statistics.training_samples_per_second,
        training_samples_per_second=result.statistics.training_samples_per_second,
        self_play_workers=arguments.self_play_workers,
        completed_searches=completed_searches,
        measured_quantum_seconds=quantum_seconds,
        searches_per_second=completed_searches / quantum_seconds,
        mean_gpu_utilization_percent=device_load.utilization_percent,
        mean_gpu_power_watts=device_load.power_watts,
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
    parser.add_argument('--quantum-count', default=2, type=int)
    parser.add_argument('--self-play-workers', default=0, type=int)
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
        self_play_workers=namespace.self_play_workers,
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
    if arguments.self_play_workers < 0:
        raise ValueError('Concurrent self-play worker count must be nonnegative.')
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
