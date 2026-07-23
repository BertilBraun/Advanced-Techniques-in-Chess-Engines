from __future__ import annotations

import argparse
import copy
import json
import os
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from AlphaZeroCpp import (
    DirectSelfPlayInferenceParams,
    InferenceClientParams,
    InferenceStatistics,
    MCTS,
    MCTSParams,
)
from src.self_play.SelfPlayCpp import SelfPlayCpp
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TRAINING_ARGS


@dataclass(frozen=True)
class Arguments:
    model: Path
    device: int
    games: int
    warmup_steps: int
    duration_seconds: float | None
    steps: int | None
    searches: int
    fast_searches: int | None
    parallel_searches: int
    threads: int
    maximum_batch_size: int
    inference_timeout_microseconds: int
    cache_capacity: int
    use_inference_cache: bool
    direct_inference_workers: int
    direct_inference_batch_size: int
    direct_outstanding_batches_per_worker: int
    seed: int
    iteration: int
    ready_file: Path | None
    start_barrier: Path | None


@dataclass(frozen=True)
class BatchSizeCount:
    batch_size: int
    calls: int


@dataclass(frozen=True)
class BenchmarkResult:
    process_id: int
    device_id: int
    seed: int
    use_inference_cache: bool
    direct_inference_workers: int
    direct_inference_batch_size: int
    direct_outstanding_batches_per_worker: int
    parallel_games: int
    warmup_steps: int
    requested_duration_seconds: float | None
    requested_steps: int | None
    target_full_searches_per_ply: int
    target_fast_searches_per_ply: int
    parallel_searches: int
    mcts_threads: int
    elapsed_seconds: float
    searches_completed: int
    searches_per_second: float
    self_play_steps: int
    game_updates: int
    completed_games: int
    completed_game_plies: int
    active_game_plies_at_end: int
    generated_samples: int
    retained_samples: int
    retained_sample_proportion: float
    live_materialized_nodes: int
    total_child_records: int
    arena_capacity_per_game: int
    process_cpu_percent: float
    peak_rss_mib: float
    inference_evaluations: int
    inference_cache_hits: int
    inference_cache_hit_rate_percent: float
    inference_unique_positions: int
    inference_cache_size_mib: int
    inference_cache_evictions: int
    inference_cache_fingerprint_collisions: int
    inference_model_calls: int
    inference_model_positions: int
    inference_average_batch_size: float
    inference_batch_size_distribution: tuple[BatchSizeCount, ...]
    tree_selection_seconds: float
    board_encoding_seconds: float
    result_processing_seconds: float
    tree_backup_seconds: float
    tree_owner_wait_seconds: float
    direct_inference_seconds: float
    direct_worker_utilization_percent: float


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        description='Benchmark the production stochastic C++ self-play game-generation path.'
    )
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--games', type=int, default=96)
    parser.add_argument('--warmup-steps', type=int, default=1)
    measurement = parser.add_mutually_exclusive_group(required=True)
    measurement.add_argument('--duration-seconds', type=float)
    measurement.add_argument('--steps', type=int)
    parser.add_argument('--searches', type=int, default=600)
    parser.add_argument('--fast-searches', type=int)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--maximum-batch-size', type=int, default=256)
    parser.add_argument('--inference-timeout-microseconds', type=int, default=500)
    parser.add_argument('--cache-capacity', type=int, default=1_500_000)
    parser.add_argument('--no-inference-cache', action='store_false', dest='use_inference_cache')
    parser.add_argument('--direct-inference-workers', type=int, default=0)
    parser.add_argument('--direct-inference-batch-size', type=int, default=64)
    parser.add_argument('--direct-outstanding-batches-per-worker', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--iteration',
        type=int,
        default=max(
            TRAINING_ARGS.self_play_search_warmup_iterations,
            TRAINING_ARGS.self_play_endgame_shortcut_fade_iterations,
        ),
    )
    parser.add_argument('--ready-file', type=Path)
    parser.add_argument('--start-barrier', type=Path)
    namespace = parser.parse_args()
    return Arguments(
        model=namespace.model,
        device=namespace.device,
        games=namespace.games,
        warmup_steps=namespace.warmup_steps,
        duration_seconds=namespace.duration_seconds,
        steps=namespace.steps,
        searches=namespace.searches,
        fast_searches=namespace.fast_searches,
        parallel_searches=namespace.parallel_searches,
        threads=namespace.threads,
        maximum_batch_size=namespace.maximum_batch_size,
        inference_timeout_microseconds=namespace.inference_timeout_microseconds,
        cache_capacity=namespace.cache_capacity,
        use_inference_cache=namespace.use_inference_cache,
        direct_inference_workers=namespace.direct_inference_workers,
        direct_inference_batch_size=namespace.direct_inference_batch_size,
        direct_outstanding_batches_per_worker=namespace.direct_outstanding_batches_per_worker,
        seed=namespace.seed,
        iteration=namespace.iteration,
        ready_file=namespace.ready_file,
        start_barrier=namespace.start_barrier,
    )


def validate_arguments(arguments: Arguments) -> None:
    if not arguments.model.is_file():
        raise ValueError(f'Model does not exist: {arguments.model}')
    if not arguments.model.name.endswith('.jit.pt'):
        raise ValueError(f'Model must be a TorchScript .jit.pt file: {arguments.model}')
    positive_integers = (
        ('games', arguments.games),
        ('searches', arguments.searches),
        ('parallel searches', arguments.parallel_searches),
        ('threads', arguments.threads),
        ('maximum batch size', arguments.maximum_batch_size),
        ('inference timeout', arguments.inference_timeout_microseconds),
        ('cache capacity', arguments.cache_capacity),
    )
    for name, value in positive_integers:
        if value < 1:
            raise ValueError(f'{name} must be positive, found {value}.')
    if arguments.device < 0:
        raise ValueError(f'device must be nonnegative, found {arguments.device}.')
    if arguments.warmup_steps < 0:
        raise ValueError(f'warmup steps cannot be negative, found {arguments.warmup_steps}.')
    if arguments.direct_inference_workers < 0:
        raise ValueError('direct inference workers cannot be negative.')
    if arguments.direct_inference_batch_size < 1:
        raise ValueError('direct inference batch size must be positive.')
    if arguments.direct_outstanding_batches_per_worker not in (1, 2):
        raise ValueError('direct outstanding batches per worker must be 1 or 2.')
    if arguments.direct_inference_workers > 0 and arguments.use_inference_cache:
        raise ValueError('direct self-play inference does not support the inference cache.')
    if arguments.duration_seconds is not None and arguments.duration_seconds <= 0:
        raise ValueError(f'duration must be positive, found {arguments.duration_seconds}.')
    if arguments.steps is not None and arguments.steps < 1:
        raise ValueError(f'steps must be positive, found {arguments.steps}.')
    if arguments.searches <= arguments.parallel_searches:
        raise ValueError('searches must exceed parallel searches.')
    if arguments.fast_searches is not None and not (
        arguments.parallel_searches < arguments.fast_searches < arguments.searches
    ):
        raise ValueError('fast searches must exceed parallel searches and remain below full searches.')
    if (arguments.ready_file is None) != (arguments.start_barrier is None):
        raise ValueError('--ready-file and --start-barrier must be provided together.')


def seed_python_libraries(seed: int, device: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_self_play(arguments: Arguments) -> SelfPlayCpp:
    configuration = copy.deepcopy(TRAINING_ARGS)
    configuration.random_seed = arguments.seed
    configuration.self_play.num_parallel_games = arguments.games
    configuration.self_play.inference_cache_capacity = arguments.cache_capacity
    configuration.self_play.mcts.num_searches_per_turn = arguments.searches
    configuration.self_play.mcts.num_parallel_searches = arguments.parallel_searches
    configuration.self_play.mcts.num_threads = arguments.threads

    self_play = SelfPlayCpp(arguments.device, configuration)
    self_play.iteration = arguments.iteration
    self_play.num_searches_per_turn = arguments.searches
    self_play.endgame_shortcut_strength = 0.0

    fast_searches = (
        arguments.fast_searches
        if arguments.fast_searches is not None
        else int(arguments.searches * configuration.self_play.mcts.fast_searches_proportion_of_full_searches)
    )
    if fast_searches <= arguments.parallel_searches:
        raise ValueError(
            f'Fast searches ({fast_searches}) must exceed parallel searches ({arguments.parallel_searches}).'
        )
    self_play.mcts = MCTS(
        InferenceClientParams(
            arguments.device,
            str(arguments.model.resolve()),
            arguments.maximum_batch_size,
            arguments.inference_timeout_microseconds,
            arguments.cache_capacity,
        ),
        MCTSParams(
            arguments.parallel_searches,
            arguments.searches,
            fast_searches,
            configuration.self_play.mcts.c_param,
            configuration.self_play.mcts.dirichlet_alpha,
            configuration.self_play.mcts.dirichlet_epsilon,
            configuration.self_play.mcts.min_visit_count,
            arguments.threads,
        ),
        use_inference_cache=arguments.use_inference_cache,
        direct_inference_params=(
            DirectSelfPlayInferenceParams(
                arguments.direct_inference_workers,
                arguments.direct_inference_batch_size,
                arguments.direct_outstanding_batches_per_worker,
            )
            if arguments.direct_inference_workers > 0
            else None
        ),
    )
    return self_play


def run_warmup(self_play: SelfPlayCpp, steps: int) -> None:
    for _ in range(steps):
        self_play.self_play()
    self_play.dataset = SelfPlayDataset()


def wait_for_synchronized_start(arguments: Arguments) -> None:
    if arguments.ready_file is None or arguments.start_barrier is None:
        return
    arguments.ready_file.touch()
    while not arguments.start_barrier.exists():
        time.sleep(0.05)


def should_continue(arguments: Arguments, completed_steps: int, started_at: float) -> bool:
    if arguments.steps is not None:
        return completed_steps < arguments.steps
    assert arguments.duration_seconds is not None
    return time.perf_counter() - started_at < arguments.duration_seconds


def difference(current: int, initial: int) -> int:
    result = current - initial
    assert result >= 0
    return result


def build_result(
    arguments: Arguments,
    self_play: SelfPlayCpp,
    initial_statistics: InferenceStatistics,
    final_statistics: InferenceStatistics,
    elapsed_seconds: float,
    process_seconds: float,
    self_play_steps: int,
    initial_completed_searches: int,
) -> BenchmarkResult:
    dataset = self_play.dataset
    retained_samples = len(dataset)

    evaluations = difference(final_statistics.evaluations, initial_statistics.evaluations)
    cache_hits = difference(final_statistics.cacheHits, initial_statistics.cacheHits)
    model_calls = difference(final_statistics.modelInferenceCalls, initial_statistics.modelInferenceCalls)
    model_positions = difference(
        final_statistics.modelInferencePositions,
        initial_statistics.modelInferencePositions,
    )
    direct_inference_nanoseconds = difference(
        final_statistics.directInferenceNanoseconds,
        initial_statistics.directInferenceNanoseconds,
    )
    batch_size_distribution = tuple(
        BatchSizeCount(
            batch_size=batch_size,
            calls=difference(calls, initial_statistics.modelBatchSizeHistogram[batch_size]),
        )
        for batch_size, calls in enumerate(final_statistics.modelBatchSizeHistogram)
        if calls > initial_statistics.modelBatchSizeHistogram[batch_size]
    )
    active_roots = [
        game.already_expanded_node for game in self_play.self_play_games if game.already_expanded_node is not None
    ]
    assert self_play.mcts is not None
    searches_completed = self_play.completed_searches - initial_completed_searches
    assert searches_completed >= 0
    return BenchmarkResult(
        process_id=os.getpid(),
        device_id=arguments.device,
        seed=arguments.seed,
        use_inference_cache=arguments.use_inference_cache,
        direct_inference_workers=arguments.direct_inference_workers,
        direct_inference_batch_size=arguments.direct_inference_batch_size,
        direct_outstanding_batches_per_worker=arguments.direct_outstanding_batches_per_worker,
        parallel_games=arguments.games,
        warmup_steps=arguments.warmup_steps,
        requested_duration_seconds=arguments.duration_seconds,
        requested_steps=arguments.steps,
        target_full_searches_per_ply=arguments.searches,
        target_fast_searches_per_ply=(
            arguments.fast_searches
            if arguments.fast_searches is not None
            else int(arguments.searches * self_play.args.mcts.fast_searches_proportion_of_full_searches)
        ),
        parallel_searches=arguments.parallel_searches,
        mcts_threads=arguments.threads,
        elapsed_seconds=elapsed_seconds,
        searches_completed=searches_completed,
        searches_per_second=searches_completed / elapsed_seconds,
        self_play_steps=self_play_steps,
        game_updates=self_play_steps * arguments.games,
        completed_games=dataset.stats.num_games,
        completed_game_plies=sum(dataset.stats.game_lengths),
        active_game_plies_at_end=sum(len(game.played_moves) for game in self_play.self_play_games),
        generated_samples=len(dataset),
        retained_samples=retained_samples,
        retained_sample_proportion=1.0,
        live_materialized_nodes=sum(root.live_nodes for root in active_roots),
        total_child_records=sum(root.total_child_records for root in active_roots),
        arena_capacity_per_game=self_play.mcts.arena_capacity,
        process_cpu_percent=100 * process_seconds / elapsed_seconds,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        inference_evaluations=evaluations,
        inference_cache_hits=cache_hits,
        inference_cache_hit_rate_percent=100 * cache_hits / evaluations if evaluations else 0.0,
        inference_unique_positions=final_statistics.uniquePositions,
        inference_cache_size_mib=final_statistics.cacheSizeMB,
        inference_cache_evictions=difference(
            final_statistics.cacheEvictions,
            initial_statistics.cacheEvictions,
        ),
        inference_cache_fingerprint_collisions=difference(
            final_statistics.cacheFingerprintCollisions,
            initial_statistics.cacheFingerprintCollisions,
        ),
        inference_model_calls=model_calls,
        inference_model_positions=model_positions,
        inference_average_batch_size=model_positions / model_calls if model_calls else 0.0,
        inference_batch_size_distribution=batch_size_distribution,
        tree_selection_seconds=difference(
            final_statistics.treeSelectionNanoseconds,
            initial_statistics.treeSelectionNanoseconds,
        )
        / 1e9,
        board_encoding_seconds=difference(
            final_statistics.boardEncodingNanoseconds,
            initial_statistics.boardEncodingNanoseconds,
        )
        / 1e9,
        result_processing_seconds=difference(
            final_statistics.resultProcessingNanoseconds,
            initial_statistics.resultProcessingNanoseconds,
        )
        / 1e9,
        tree_backup_seconds=difference(
            final_statistics.treeBackupNanoseconds,
            initial_statistics.treeBackupNanoseconds,
        )
        / 1e9,
        tree_owner_wait_seconds=difference(
            final_statistics.treeOwnerWaitNanoseconds,
            initial_statistics.treeOwnerWaitNanoseconds,
        )
        / 1e9,
        direct_inference_seconds=direct_inference_nanoseconds / 1e9,
        direct_worker_utilization_percent=(
            100 * direct_inference_nanoseconds / (elapsed_seconds * 1e9 * arguments.direct_inference_workers)
            if arguments.direct_inference_workers > 0
            else 0.0
        ),
    )


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    seed_python_libraries(arguments.seed, arguments.device)
    self_play = create_self_play(arguments)
    assert self_play.mcts is not None

    run_warmup(self_play, arguments.warmup_steps)
    initial_completed_searches = self_play.completed_searches
    initial_statistics, _ = self_play.mcts.get_inference_statistics()
    wait_for_synchronized_start(arguments)

    process_started_at = time.process_time()
    started_at = time.perf_counter()
    self_play_steps = 0
    while should_continue(arguments, self_play_steps, started_at):
        self_play.self_play()
        self_play_steps += 1
    elapsed_seconds = time.perf_counter() - started_at
    process_seconds = time.process_time() - process_started_at

    final_statistics, _ = self_play.mcts.get_inference_statistics()
    result = build_result(
        arguments,
        self_play,
        initial_statistics,
        final_statistics,
        elapsed_seconds,
        process_seconds,
        self_play_steps,
        initial_completed_searches,
    )
    print(json.dumps(asdict(result), separators=(',', ':')))


if __name__ == '__main__':
    main()
