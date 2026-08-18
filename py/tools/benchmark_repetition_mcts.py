from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from AlphaZeroCpp import (
    BatchedInferenceParameters,
    ChessPosition,
    ChessSearchRoot,
    ChessSelfPlaySearch,
    ChessSelfPlaySearchRequest,
    FirstPlayUrgencyKind,
    FirstPlayUrgencyParameters,
    InferenceConfiguration,
    SelfPlaySearchParameters,
    TreeSearchParameters,
)

if TYPE_CHECKING:
    from AlphaZeroCpp import ChessSelfPlaySearchResult, InferenceStatistics


@dataclass(frozen=True)
class GpuSample:
    utilization_percent: float
    memory_mib: float


@dataclass(frozen=True)
class BatchSizeCount:
    batch_size: int
    calls: int


@dataclass(frozen=True)
class CacheCounters:
    total_positions: int
    unique_hashes: int
    repeated_hashes: int
    same_batch_repeats: int
    prior_batch_repeats: int
    set_size: int


@dataclass(frozen=True)
class PlyResult:
    measured_ply: int
    elapsed_seconds: float
    searches_completed: int
    searches_per_second: float
    inference_positions: int
    unique_hashes: int
    repeated_hashes: int
    repeat_rate: float
    same_batch_repeats: int
    prior_batch_repeats: int
    cumulative_set_size: int
    terminal_roots: int


@dataclass(frozen=True)
class BenchmarkResult:
    process_id: int
    device_id: int
    games: int
    warmup_games: int
    warmup_steps: int
    measurement_steps: int
    target_searches_per_ply: int
    parallel_searches: int
    maximum_opening_plies: int
    random_seed: int
    unique_start_positions: int
    starting_position_digest: str
    minimum_starting_ply: int
    maximum_starting_ply: int
    mean_starting_ply: float
    root_noise: bool
    elapsed_seconds: float
    completed_game_plies: int
    completed_game_plies_per_second: float
    searches_completed: int
    searches_per_second: float
    process_cpu_percent: float
    peak_rss_mib: float
    mean_gpu_utilization_percent: float | None
    peak_gpu_memory_mib: float | None
    terminal_roots: int
    inference_evaluations: int
    inference_model_calls: int
    inference_model_positions: int
    inference_average_batch_size: float
    inference_batch_size_distribution: tuple[BatchSizeCount, ...]
    cache_total_positions: int
    cache_unique_hashes: int
    cache_repeated_hashes: int
    cache_repeat_rate: float
    cache_same_batch_repeats: int
    cache_prior_batch_repeats: int
    cache_set_size_before_measurement: int
    cache_set_size_after_measurement: int
    per_ply: tuple[PlyResult, ...]


@dataclass(frozen=True)
class Arguments:
    model: Path
    device: int
    games: int
    warmup_games: int
    warmup_steps: int
    steps: int
    searches: int
    parallel_searches: int
    maximum_opening_plies: int
    random_seed: int
    inference_workers: int
    maximum_batch_size: int
    outstanding_batches_per_worker: int
    gpu_sampling_interval_seconds: float


@dataclass(frozen=True)
class RandomStart:
    position: ChessPosition
    opening_plies: int
    encoding: bytes


@dataclass(frozen=True)
class SearchStepsResult:
    roots: list[ChessSearchRoot]
    game_plies: list[int]
    terminal_roots: int
    searches_completed: int
    per_ply: tuple[PlyResult, ...]


class UniqueRandomStartGenerator:
    def __init__(self, random_seed: int, maximum_opening_plies: int) -> None:
        if maximum_opening_plies < 1:
            raise ValueError('Maximum random opening plies must be positive.')
        self._random = random.Random(random_seed)
        self._maximum_opening_plies = maximum_opening_plies
        self._seen_encodings: set[bytes] = set()

    @property
    def unique_positions(self) -> int:
        return len(self._seen_encodings)

    def next(self) -> RandomStart:
        while True:
            position = ChessPosition()
            opening_plies = self._random.randint(1, self._maximum_opening_plies)
            completed_plies = 0
            for _ in range(opening_plies):
                legal_actions = position.legal_actions()
                position = position.child(self._random.choice(legal_actions))
                completed_plies += 1
                if position.is_terminal:
                    break
            if position.is_terminal:
                continue
            encoding = bytes(position.packed_encoding())
            if encoding in self._seen_encodings:
                continue
            self._seen_encodings.add(encoding)
            return RandomStart(position=position, opening_plies=completed_plies, encoding=encoding)


def query_gpu(device_id: int) -> GpuSample:
    completed = subprocess.run(
        [
            'nvidia-smi',
            f'--id={device_id}',
            '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utilization, memory = completed.stdout.strip().split(',', maxsplit=1)
    return GpuSample(
        utilization_percent=float(utilization.strip()),
        memory_mib=float(memory.strip()),
    )


def sample_gpu_until_stopped(
    device_id: int,
    interval_seconds: float,
    stop_event: threading.Event,
    samples: list[GpuSample],
) -> None:
    while not stop_event.wait(interval_seconds):
        samples.append(query_gpu(device_id))


def cache_counters(statistics: InferenceStatistics) -> CacheCounters:
    return CacheCounters(
        total_positions=statistics.cacheTotalPositions,
        unique_hashes=statistics.cacheUniqueHashes,
        repeated_hashes=statistics.cacheRepeatedHashes,
        same_batch_repeats=statistics.cacheSameBatchRepeats,
        prior_batch_repeats=statistics.cachePriorBatchRepeats,
        set_size=statistics.cacheSetSize,
    )


def subtract_cache_counters(after: CacheCounters, before: CacheCounters) -> CacheCounters:
    return CacheCounters(
        total_positions=after.total_positions - before.total_positions,
        unique_hashes=after.unique_hashes - before.unique_hashes,
        repeated_hashes=after.repeated_hashes - before.repeated_hashes,
        same_batch_repeats=after.same_batch_repeats - before.same_batch_repeats,
        prior_batch_repeats=after.prior_batch_repeats - before.prior_batch_repeats,
        set_size=after.set_size,
    )


def select_action(result: ChessSelfPlaySearchResult, game_ply: int, generator: random.Random) -> int:
    visits = result.search_visits
    if not visits:
        raise ValueError('MCTS returned no visits for a nonterminal root.')
    greedy_after_ply = 60
    progress = min(game_ply / greedy_after_ply, 1.0)
    temperature = 1.3 + (0.1 - 1.3) * progress
    weights = [float(visit.visit_count) ** (1.0 / temperature) for visit in visits]
    return generator.choices([visit.action_id for visit in visits], weights=weights, k=1)[0]


def create_search(args: Arguments, model_generation: int = 0) -> ChessSelfPlaySearch:
    return ChessSelfPlaySearch(
        InferenceConfiguration(args.device, str(args.model)),
        SelfPlaySearchParameters(
            args.parallel_searches,
            args.searches,
            args.searches,
            TreeSearchParameters(
                1.5,
                FirstPlayUrgencyParameters(FirstPlayUrgencyKind.REDUCED_PARENT_VALUE, 0.2),
                1.5,
                1.0,
            ),
            0.3,
            0.25,
        ),
        BatchedInferenceParameters(
            args.inference_workers,
            args.maximum_batch_size,
            args.outstanding_batches_per_worker,
        ),
        model_generation,
    )


def new_random_roots(
    search: ChessSelfPlaySearch,
    start_generator: UniqueRandomStartGenerator,
    count: int,
) -> tuple[list[ChessSearchRoot], list[int], tuple[bytes, ...]]:
    starts = tuple(start_generator.next() for _ in range(count))
    return (
        [search.new_root(start.position) for start in starts],
        [start.opening_plies for start in starts],
        tuple(start.encoding for start in starts),
    )


def starting_position_digest(encodings: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for encoding in encodings:
        digest.update(len(encoding).to_bytes(8, byteorder='little'))
        digest.update(encoding)
    return digest.hexdigest()


def run_search_steps(
    search: ChessSelfPlaySearch,
    roots: list[ChessSearchRoot],
    game_plies: list[int],
    start_generator: UniqueRandomStartGenerator,
    move_generator: random.Random,
    steps: int,
    collect_per_ply: bool,
) -> SearchStepsResult:
    terminal_roots = 0
    searches_completed = 0
    per_ply: list[PlyResult] = []
    for measured_ply in range(1, steps + 1):
        cache_before = cache_counters(search.inference_statistics())
        wall_time_start = time.perf_counter()
        search_results = search.search([ChessSelfPlaySearchRequest(root, True) for root in roots])
        ply_elapsed_seconds = time.perf_counter() - wall_time_start
        ply_searches = search_results.simulations_completed
        searches_completed += ply_searches
        next_roots: list[ChessSearchRoot] = []
        next_game_plies: list[int] = []
        ply_terminal_roots = 0
        for root, game_ply, result in zip(roots, game_plies, search_results.results, strict=True):
            action_id = select_action(result, game_ply, move_generator)
            root.play(action_id)
            if root.is_terminal:
                ply_terminal_roots += 1
                replacement = start_generator.next()
                root = search.new_root(replacement.position)
                game_ply = replacement.opening_plies
            else:
                game_ply += 1
            next_roots.append(root)
            next_game_plies.append(game_ply)
        roots = next_roots
        game_plies = next_game_plies
        terminal_roots += ply_terminal_roots
        if collect_per_ply:
            counters = subtract_cache_counters(cache_counters(search.inference_statistics()), cache_before)
            per_ply.append(
                PlyResult(
                    measured_ply=measured_ply,
                    elapsed_seconds=ply_elapsed_seconds,
                    searches_completed=ply_searches,
                    searches_per_second=ply_searches / ply_elapsed_seconds,
                    inference_positions=counters.total_positions,
                    unique_hashes=counters.unique_hashes,
                    repeated_hashes=counters.repeated_hashes,
                    repeat_rate=(
                        counters.repeated_hashes / counters.total_positions if counters.total_positions else 0.0
                    ),
                    same_batch_repeats=counters.same_batch_repeats,
                    prior_batch_repeats=counters.prior_batch_repeats,
                    cumulative_set_size=counters.set_size,
                    terminal_roots=ply_terminal_roots,
                )
            )
    return SearchStepsResult(
        roots=roots,
        game_plies=game_plies,
        terminal_roots=terminal_roots,
        searches_completed=searches_completed,
        per_ply=tuple(per_ply),
    )


def run_benchmark(args: Arguments) -> BenchmarkResult:
    positive_values = (
        args.games,
        args.warmup_games,
        args.warmup_steps,
        args.steps,
        args.searches,
        args.parallel_searches,
        args.inference_workers,
        args.maximum_batch_size,
        args.outstanding_batches_per_worker,
    )
    if any(value < 1 for value in positive_values):
        raise ValueError('Game, search, inference, warm-up, and step counts must be positive.')
    if args.parallel_searches > 2:
        raise ValueError('This self-play cache benchmark permits at most two parallel searches.')
    if args.searches < 150 or args.searches > 800:
        raise ValueError('Search budget must lie in the self-play experiment range [150, 800].')
    if args.gpu_sampling_interval_seconds < 0:
        raise ValueError('GPU sampling interval cannot be negative.')

    search = create_search(args)
    start_generator = UniqueRandomStartGenerator(args.random_seed, args.maximum_opening_plies)
    move_generator = random.Random(args.random_seed + 1)
    warmup_roots, warmup_game_plies, _ = new_random_roots(search, start_generator, args.warmup_games)
    run_search_steps(
        search,
        warmup_roots,
        warmup_game_plies,
        start_generator,
        move_generator,
        args.warmup_steps,
        collect_per_ply=False,
    )
    search.reset_inference_cache_tracker()
    roots, game_plies, initial_encodings = new_random_roots(search, start_generator, args.games)
    initial_game_plies = tuple(game_plies)
    initial_unique_positions = len(set(initial_encodings))
    if initial_unique_positions != args.games:
        raise RuntimeError('Random start generation did not produce unique encoded positions.')
    inference_before = search.inference_statistics()
    cache_before = cache_counters(inference_before)

    gpu_samples: list[GpuSample] = []
    stop_event = threading.Event()
    sampler: threading.Thread | None = None
    if args.gpu_sampling_interval_seconds > 0:
        sampler = threading.Thread(
            target=sample_gpu_until_stopped,
            args=(args.device, args.gpu_sampling_interval_seconds, stop_event, gpu_samples),
            daemon=True,
        )
        sampler.start()

    process_time_start = time.process_time()
    wall_time_start = time.perf_counter()
    measurement_result = run_search_steps(
        search,
        roots,
        game_plies,
        start_generator,
        move_generator,
        args.steps,
        collect_per_ply=True,
    )
    elapsed_seconds = time.perf_counter() - wall_time_start
    process_seconds = time.process_time() - process_time_start
    stop_event.set()
    if sampler is not None:
        sampler.join()

    inference_after = search.inference_statistics()
    counters = subtract_cache_counters(cache_counters(inference_after), cache_before)
    measurement_evaluations = inference_after.evaluations - inference_before.evaluations
    measurement_model_calls = inference_after.modelInferenceCalls - inference_before.modelInferenceCalls
    measurement_model_positions = inference_after.modelInferencePositions - inference_before.modelInferencePositions
    batch_size_distribution = tuple(
        BatchSizeCount(
            batch_size=batch_size,
            calls=calls - inference_before.modelBatchSizeHistogram[batch_size],
        )
        for batch_size, calls in enumerate(inference_after.modelBatchSizeHistogram)
        if calls > inference_before.modelBatchSizeHistogram[batch_size]
    )
    completed_game_plies = args.games * args.steps
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return BenchmarkResult(
        process_id=os.getpid(),
        device_id=args.device,
        games=args.games,
        warmup_games=args.warmup_games,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.steps,
        target_searches_per_ply=args.searches,
        parallel_searches=args.parallel_searches,
        maximum_opening_plies=args.maximum_opening_plies,
        random_seed=args.random_seed,
        unique_start_positions=initial_unique_positions,
        starting_position_digest=starting_position_digest(initial_encodings),
        minimum_starting_ply=min(initial_game_plies),
        maximum_starting_ply=max(initial_game_plies),
        mean_starting_ply=sum(initial_game_plies) / len(initial_game_plies),
        root_noise=True,
        elapsed_seconds=elapsed_seconds,
        completed_game_plies=completed_game_plies,
        completed_game_plies_per_second=completed_game_plies / elapsed_seconds,
        searches_completed=measurement_result.searches_completed,
        searches_per_second=measurement_result.searches_completed / elapsed_seconds,
        process_cpu_percent=100 * process_seconds / elapsed_seconds,
        peak_rss_mib=peak_rss_mib,
        mean_gpu_utilization_percent=(
            sum(sample.utilization_percent for sample in gpu_samples) / len(gpu_samples) if gpu_samples else None
        ),
        peak_gpu_memory_mib=(max(sample.memory_mib for sample in gpu_samples) if gpu_samples else None),
        terminal_roots=measurement_result.terminal_roots,
        inference_evaluations=measurement_evaluations,
        inference_model_calls=measurement_model_calls,
        inference_model_positions=measurement_model_positions,
        inference_average_batch_size=(
            measurement_model_positions / measurement_model_calls if measurement_model_calls else 0.0
        ),
        inference_batch_size_distribution=batch_size_distribution,
        cache_total_positions=counters.total_positions,
        cache_unique_hashes=counters.unique_hashes,
        cache_repeated_hashes=counters.repeated_hashes,
        cache_repeat_rate=(counters.repeated_hashes / counters.total_positions if counters.total_positions else 0.0),
        cache_same_batch_repeats=counters.same_batch_repeats,
        cache_prior_batch_repeats=counters.prior_batch_repeats,
        cache_set_size_before_measurement=cache_before.set_size,
        cache_set_size_after_measurement=counters.set_size,
        per_ply=measurement_result.per_ply,
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--warmup-games', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--searches', type=int, choices=range(150, 801), default=800)
    parser.add_argument('--parallel-searches', type=int, choices=(1, 2), default=1)
    parser.add_argument('--maximum-opening-plies', type=int, default=12)
    parser.add_argument('--random-seed', type=int, default=20260818)
    parser.add_argument('--inference-workers', type=int, default=2)
    parser.add_argument('--maximum-batch-size', type=int, default=64)
    parser.add_argument('--outstanding-batches-per-worker', type=int, default=2)
    parser.add_argument('--gpu-sampling-interval-seconds', type=float, default=1.0)
    namespace = parser.parse_args()
    return Arguments(
        model=namespace.model,
        device=namespace.device,
        games=namespace.games,
        warmup_games=namespace.warmup_games,
        warmup_steps=namespace.warmup_steps,
        steps=namespace.steps,
        searches=namespace.searches,
        parallel_searches=namespace.parallel_searches,
        maximum_opening_plies=namespace.maximum_opening_plies,
        random_seed=namespace.random_seed,
        inference_workers=namespace.inference_workers,
        maximum_batch_size=namespace.maximum_batch_size,
        outstanding_batches_per_worker=namespace.outstanding_batches_per_worker,
        gpu_sampling_interval_seconds=namespace.gpu_sampling_interval_seconds,
    )


def main() -> None:
    result = run_benchmark(parse_arguments())
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == '__main__':
    main()
