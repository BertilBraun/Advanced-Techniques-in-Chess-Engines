from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from AlphaZeroCpp import InferenceClientParams, MCTS, MCTSBoard, MCTSParams, MCTSRoot


@dataclass(frozen=True)
class GpuSample:
    utilization_percent: float
    memory_mib: float


@dataclass(frozen=True)
class BatchSizeCount:
    batch_size: int
    calls: int


@dataclass(frozen=True)
class BenchmarkResult:
    process_id: int
    device_id: int
    games: int
    warmup_steps: int
    measurement_steps: int
    target_searches_per_ply: int
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
    inference_cache_hits: int
    inference_cache_hit_rate_percent: float
    inference_unique_positions: int
    inference_cache_size_mib: int
    inference_cache_fingerprint_collisions: int
    inference_model_calls: int
    inference_model_positions: int
    inference_average_batch_size: float
    inference_batch_size_distribution: tuple[BatchSizeCount, ...]


@dataclass(frozen=True)
class Arguments:
    model: Path
    openings: Path
    device: int
    games: int
    warmup_steps: int
    steps: int
    searches: int
    parallel_searches: int
    threads: int
    maximum_batch_size: int
    inference_timeout_microseconds: int
    cache_capacity: int
    gpu_sampling_interval_seconds: float
    ready_file: Path | None
    start_barrier: Path | None


@dataclass(frozen=True)
class SearchStepsResult:
    roots: list[MCTSRoot]
    terminal_roots: int
    searches_completed: int


def load_openings(path: Path, number_of_games: int) -> tuple[str, ...]:
    openings = tuple(
        line.split('\t', maxsplit=1)[1]
        for line in path.read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#')
    )
    if not openings:
        raise ValueError(f'Opening suite is empty: {path}')
    return tuple(itertools.islice(itertools.cycle(openings), number_of_games))


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


def choose_root(result_root: MCTSRoot, visits: list[tuple[int, int]]) -> MCTSRoot:
    if not visits:
        raise ValueError('MCTS returned no visits for a nonterminal root.')
    child_index = max(range(len(visits)), key=lambda index: visits[index][1])
    return result_root.make_new_root(child_index)


def wait_for_synchronized_start(args: Arguments) -> None:
    if args.ready_file is None and args.start_barrier is None:
        return
    if args.ready_file is None or args.start_barrier is None:
        raise ValueError('--ready-file and --start-barrier must be provided together.')

    args.ready_file.touch()
    while not args.start_barrier.exists():
        time.sleep(0.05)


def run_search_steps(
    mcts: MCTS,
    roots: list[MCTSRoot],
    openings: tuple[str, ...],
    steps: int,
) -> SearchStepsResult:
    terminal_roots = 0
    searches_completed = 0
    for _ in range(steps):
        visits_before = sum(root.visits for root in roots)
        search_results = mcts.search([MCTSBoard(root, False) for root in roots])
        searches_completed += sum(result.root.visits for result in search_results.results) - visits_before

        next_roots: list[MCTSRoot] = []
        for opening_index, result in enumerate(search_results.results):
            root = choose_root(result.root, result.visits)
            if root.is_terminal:
                terminal_roots += 1
                root = mcts.new_root(openings[opening_index])
            next_roots.append(root)
        roots = next_roots
    return SearchStepsResult(
        roots=roots,
        terminal_roots=terminal_roots,
        searches_completed=searches_completed,
    )


def run_benchmark(args: Arguments) -> BenchmarkResult:
    if args.games < 1 or args.steps < 1 or args.searches < 1 or args.threads < 1:
        raise ValueError('games, steps, searches, and threads must be positive.')
    if args.warmup_steps < 0:
        raise ValueError('warmup steps cannot be negative.')
    if args.gpu_sampling_interval_seconds < 0:
        raise ValueError('GPU sampling interval cannot be negative.')

    mcts = MCTS(
        InferenceClientParams(
            args.device,
            str(args.model),
            args.maximum_batch_size,
            args.inference_timeout_microseconds,
            args.cache_capacity,
        ),
        MCTSParams(
            args.parallel_searches,
            args.searches,
            args.searches,
            1.0,
            0.3,
            0.0,
            0,
            args.threads,
        ),
    )
    openings = load_openings(args.openings, args.games)
    roots = [mcts.new_root(fen) for fen in openings]
    warmup_result = run_search_steps(mcts, roots, openings, args.warmup_steps)
    roots = warmup_result.roots
    warmup_inference_statistics, _ = mcts.get_inference_statistics()
    wait_for_synchronized_start(args)

    gpu_samples: list[GpuSample] = []
    stop_event = threading.Event()
    sampler: threading.Thread | None = None
    if args.gpu_sampling_interval_seconds > 0:
        sampler = threading.Thread(
            target=sample_gpu_until_stopped,
            args=(
                args.device,
                args.gpu_sampling_interval_seconds,
                stop_event,
                gpu_samples,
            ),
            daemon=True,
        )
        sampler.start()

    process_time_start = time.process_time()
    wall_time_start = time.perf_counter()
    measurement_result = run_search_steps(mcts, roots, openings, args.steps)
    elapsed_seconds = time.perf_counter() - wall_time_start
    process_seconds = time.process_time() - process_time_start
    stop_event.set()
    if sampler is not None:
        sampler.join()

    inference_statistics, _ = mcts.get_inference_statistics()
    measurement_evaluations = inference_statistics.evaluations - warmup_inference_statistics.evaluations
    measurement_cache_hits = inference_statistics.cacheHits - warmup_inference_statistics.cacheHits
    measurement_model_calls = inference_statistics.modelInferenceCalls - warmup_inference_statistics.modelInferenceCalls
    measurement_model_positions = (
        inference_statistics.modelInferencePositions - warmup_inference_statistics.modelInferencePositions
    )
    batch_size_distribution = tuple(
        BatchSizeCount(
            batch_size=batch_size,
            calls=calls - warmup_inference_statistics.modelBatchSizeHistogram[batch_size],
        )
        for batch_size, calls in enumerate(inference_statistics.modelBatchSizeHistogram)
        if calls > warmup_inference_statistics.modelBatchSizeHistogram[batch_size]
    )
    completed_game_plies = args.games * args.steps
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return BenchmarkResult(
        process_id=os.getpid(),
        device_id=args.device,
        games=args.games,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.steps,
        target_searches_per_ply=args.searches,
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
        inference_cache_hits=measurement_cache_hits,
        inference_cache_hit_rate_percent=(
            100 * measurement_cache_hits / measurement_evaluations if measurement_evaluations else 0.0
        ),
        inference_unique_positions=inference_statistics.uniquePositions,
        inference_cache_size_mib=inference_statistics.cacheSizeMB,
        inference_cache_fingerprint_collisions=(
            inference_statistics.cacheFingerprintCollisions - warmup_inference_statistics.cacheFingerprintCollisions
        ),
        inference_model_calls=measurement_model_calls,
        inference_model_positions=measurement_model_positions,
        inference_average_batch_size=(
            measurement_model_positions / measurement_model_calls if measurement_model_calls else 0.0
        ),
        inference_batch_size_distribution=batch_size_distribution,
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--openings', required=True, type=Path)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--games', type=int, default=16)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--searches', type=int, default=600)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--maximum-batch-size', type=int, default=256)
    parser.add_argument('--inference-timeout-microseconds', type=int, default=500)
    parser.add_argument('--cache-capacity', type=int, default=100_000)
    parser.add_argument('--gpu-sampling-interval-seconds', type=float, default=1.0)
    parser.add_argument('--ready-file', type=Path)
    parser.add_argument('--start-barrier', type=Path)
    namespace = parser.parse_args()
    return Arguments(
        model=namespace.model,
        openings=namespace.openings,
        device=namespace.device,
        games=namespace.games,
        warmup_steps=namespace.warmup_steps,
        steps=namespace.steps,
        searches=namespace.searches,
        parallel_searches=namespace.parallel_searches,
        threads=namespace.threads,
        maximum_batch_size=namespace.maximum_batch_size,
        inference_timeout_microseconds=namespace.inference_timeout_microseconds,
        cache_capacity=namespace.cache_capacity,
        gpu_sampling_interval_seconds=namespace.gpu_sampling_interval_seconds,
        ready_file=namespace.ready_file,
        start_barrier=namespace.start_barrier,
    )


def main() -> None:
    result = run_benchmark(parse_arguments())
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == '__main__':
    main()
