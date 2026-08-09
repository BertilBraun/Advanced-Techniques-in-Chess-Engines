from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Literal

from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.go.configuration import GoExperimentConfiguration
from src.self_play.worker import SelfPlayWorker
from src.training.checkpoint import CheckpointReference


@dataclass(frozen=True)
class Arguments:
    run_config: Path
    model: Path
    device: int
    worker_id: int
    inference_device: Literal['cpu', 'cuda']
    games: int
    generation: int
    warmup_batches: int
    duration_seconds: float
    ready_file: Path | None
    start_barrier: Path | None
    parallel_searches: int | None
    inference_workers: int | None
    inference_batch_size: int | None
    outstanding_batches_per_worker: int | None


@dataclass(frozen=True)
class BatchSizeCount:
    batch_size: int
    calls: int


@dataclass(frozen=True)
class BenchmarkResult:
    process_id: int
    worker_id: int
    game: str
    device_id: int
    inference_device: Literal['cpu', 'cuda']
    model_generation: int
    parallel_games: int
    parallel_searches: int
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int
    warmup_batches: int
    elapsed_seconds: float
    search_batches: int
    searches_completed: int
    searches_per_second: float
    completed_games: int
    process_cpu_percent: float
    peak_rss_mib: float
    inference_evaluations: int
    inference_model_calls: int
    inference_model_positions: int
    inference_average_batch_size: float
    inference_batch_size_distribution: tuple[BatchSizeCount, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(arguments: Arguments) -> CheckpointReference:
    return CheckpointReference(
        generation=arguments.generation,
        manifest_path=arguments.model.with_suffix('.benchmark-manifest.json'),
        model_path=arguments.model,
        optimizer_path=arguments.model.with_suffix('.benchmark-optimizer.pt'),
        inference_model_path=arguments.model,
        inference_model_sha256=_sha256(arguments.model),
    )


def _wait_for_start(arguments: Arguments) -> None:
    if arguments.ready_file is None and arguments.start_barrier is None:
        return
    if arguments.ready_file is None or arguments.start_barrier is None:
        raise ValueError('--ready-file and --start-barrier must be supplied together.')
    arguments.ready_file.touch()
    while not arguments.start_barrier.exists():
        time.sleep(0.05)


def _difference(current: int, initial: int) -> int:
    difference = current - initial
    assert difference >= 0
    return difference


def _apply_self_play_overrides(
    experiment: ChessExperimentConfiguration | GoExperimentConfiguration,
    arguments: Arguments,
) -> ChessExperimentConfiguration | GoExperimentConfiguration:
    match experiment:
        case ChessExperimentConfiguration(chess=game_configuration):
            field_name = 'chess'
        case GoExperimentConfiguration(go=game_configuration):
            field_name = 'go'
    self_play = game_configuration.self_play
    search_update = {} if arguments.parallel_searches is None else {'parallel_searches': arguments.parallel_searches}
    inference_update = {
        field_name: value
        for field_name, value in (
            ('inference_workers', arguments.inference_workers),
            ('inference_batch_size', arguments.inference_batch_size),
            ('outstanding_batches_per_worker', arguments.outstanding_batches_per_worker),
        )
        if value is not None
    }
    updated_self_play = self_play.validated_copy(
        update={
            'search': self_play.search.validated_copy(update=search_update).model_dump(mode='json'),
            'inference': self_play.inference.validated_copy(update=inference_update).model_dump(mode='json'),
        }
    )
    updated_game_configuration = game_configuration.validated_copy(
        update={'self_play': updated_self_play.model_dump(mode='json')}
    )
    return experiment.validated_copy(update={field_name: updated_game_configuration.model_dump(mode='json')})


def run_benchmark(arguments: Arguments) -> BenchmarkResult:
    experiment = load_experiment_configuration(arguments.run_config)
    experiment = _apply_self_play_overrides(experiment, arguments)
    configured_trainer = experiment.training.topology.trainer
    trainer = configured_trainer.validated_copy(
        update={
            'device_type': arguments.inference_device,
            'process_group_backend': 'nccl' if arguments.inference_device == 'cuda' else 'gloo',
            'rank_zero_device_id': 0,
            'ddp_device_ids': (0,),
        }
    )
    topology = experiment.training.topology.validated_copy(update={'trainer': trainer.model_dump(mode='json')})
    training = experiment.training.validated_copy(update={'topology': topology.model_dump(mode='json')})
    experiment = experiment.validated_copy(update={'training': training.model_dump(mode='json')})
    game = create_game_implementation(experiment)
    with tempfile.TemporaryDirectory(prefix='self-play-search-benchmark-') as temporary_directory:
        inbox = Path(temporary_directory)
        worker = SelfPlayWorker(
            game=game,
            parallel_game_count=arguments.games,
            worker_id=arguments.worker_id,
            device_id=arguments.device,
            inbox_path=inbox,
        )
        worker.refresh_published_model(_checkpoint(arguments))
        for _ in range(arguments.warmup_batches):
            worker.run_batch()
        initial = worker.snapshot_statistics()
        initial_files = len(tuple(inbox.iterdir()))
        _wait_for_start(arguments)

        process_started_at = time.process_time()
        started_at = time.perf_counter()
        search_batches = 0
        while time.perf_counter() - started_at < arguments.duration_seconds:
            worker.run_batch()
            search_batches += 1
        elapsed_seconds = time.perf_counter() - started_at
        process_seconds = time.process_time() - process_started_at
        final = worker.snapshot_statistics()
        completed_games = len(tuple(inbox.iterdir())) - initial_files

    searches_completed = _difference(final.completed_searches, initial.completed_searches)
    evaluations = _difference(final.inference.evaluations, initial.inference.evaluations)
    model_calls = _difference(final.inference.modelInferenceCalls, initial.inference.modelInferenceCalls)
    model_positions = _difference(
        final.inference.modelInferencePositions,
        initial.inference.modelInferencePositions,
    )
    batch_sizes = tuple(
        BatchSizeCount(
            batch_size=batch_size,
            calls=_difference(calls, initial.inference.modelBatchSizeHistogram[batch_size]),
        )
        for batch_size, calls in enumerate(final.inference.modelBatchSizeHistogram)
        if calls > initial.inference.modelBatchSizeHistogram[batch_size]
    )
    return BenchmarkResult(
        process_id=os.getpid(),
        worker_id=arguments.worker_id,
        game=experiment.game,
        device_id=arguments.device,
        inference_device=arguments.inference_device,
        model_generation=arguments.generation,
        parallel_games=arguments.games,
        parallel_searches=game.self_play_configuration.search.parallel_searches,
        inference_workers=game.self_play_configuration.inference.inference_workers,
        inference_batch_size=game.self_play_configuration.inference.inference_batch_size,
        outstanding_batches_per_worker=game.self_play_configuration.inference.outstanding_batches_per_worker,
        warmup_batches=arguments.warmup_batches,
        elapsed_seconds=elapsed_seconds,
        search_batches=search_batches,
        searches_completed=searches_completed,
        searches_per_second=searches_completed / elapsed_seconds,
        completed_games=completed_games,
        process_cpu_percent=100.0 * process_seconds / elapsed_seconds,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        inference_evaluations=evaluations,
        inference_model_calls=model_calls,
        inference_model_positions=model_positions,
        inference_average_batch_size=model_positions / model_calls if model_calls else 0.0,
        inference_batch_size_distribution=batch_sizes,
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Benchmark the current production self-play search path.')
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--worker-id', required=True, type=int)
    parser.add_argument('--inference-device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--games', required=True, type=int)
    parser.add_argument('--generation', default=0, type=int)
    parser.add_argument('--warmup-batches', default=2, type=int)
    parser.add_argument('--duration-seconds', default=60.0, type=float)
    parser.add_argument('--ready-file', type=Path)
    parser.add_argument('--start-barrier', type=Path)
    parser.add_argument('--parallel-searches', type=int)
    parser.add_argument('--inference-workers', type=int)
    parser.add_argument('--inference-batch-size', type=int)
    parser.add_argument('--outstanding-batches-per-worker', type=int)
    namespace = parser.parse_args()
    arguments = Arguments(
        run_config=namespace.run_config,
        model=namespace.model,
        device=namespace.device,
        worker_id=namespace.worker_id,
        inference_device=namespace.inference_device,
        games=namespace.games,
        generation=namespace.generation,
        warmup_batches=namespace.warmup_batches,
        duration_seconds=namespace.duration_seconds,
        ready_file=namespace.ready_file,
        start_barrier=namespace.start_barrier,
        parallel_searches=namespace.parallel_searches,
        inference_workers=namespace.inference_workers,
        inference_batch_size=namespace.inference_batch_size,
        outstanding_batches_per_worker=namespace.outstanding_batches_per_worker,
    )
    if not arguments.model.is_file():
        raise ValueError(f'Benchmark model does not exist: {arguments.model}')
    if arguments.device < 0 or arguments.worker_id < 0 or arguments.generation < 0 or arguments.warmup_batches < 0:
        raise ValueError('Device, worker, generation, and warm-up batches must be nonnegative.')
    if arguments.games <= 0 or arguments.duration_seconds <= 0.0:
        raise ValueError('Games and duration must be positive.')
    overrides = (
        arguments.parallel_searches,
        arguments.inference_workers,
        arguments.inference_batch_size,
        arguments.outstanding_batches_per_worker,
    )
    if any(value is not None and value <= 0 for value in overrides):
        raise ValueError('Self-play benchmark overrides must be positive.')
    return arguments


def main() -> None:
    print(json.dumps(asdict(run_benchmark(parse_arguments())), sort_keys=True))


if __name__ == '__main__':
    main()
