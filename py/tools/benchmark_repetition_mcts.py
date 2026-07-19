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

from AlphaZeroCpp import InferenceClientParams, MCTS, MCTSNode, MCTSParams, new_root


@dataclass(frozen=True)
class GpuSample:
    utilization_percent: float
    memory_mib: float


@dataclass(frozen=True)
class BenchmarkResult:
    process_id: int
    device_id: int
    games: int
    steps: int
    searches_per_step: int
    elapsed_seconds: float
    positions_per_second: float
    process_cpu_percent: float
    peak_rss_mib: float
    mean_gpu_utilization_percent: float | None
    peak_gpu_memory_mib: float | None
    terminal_roots: int
    inference_unique_positions: int
    inference_cache_size_mib: int
    inference_cache_fingerprint_collisions: int


@dataclass(frozen=True)
class Arguments:
    model: Path
    openings: Path
    device: int
    games: int
    steps: int
    searches: int
    parallel_searches: int
    threads: int
    maximum_batch_size: int
    inference_timeout_microseconds: int
    cache_capacity: int
    ready_file: Path | None
    start_barrier: Path | None


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
    stop_event: threading.Event,
    samples: list[GpuSample],
) -> None:
    while not stop_event.wait(1.0):
        samples.append(query_gpu(device_id))


def choose_root(result_root: MCTSNode, visits: list[tuple[int, int]]) -> MCTSNode:
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


def run_benchmark(args: Arguments) -> BenchmarkResult:
    if args.games < 1 or args.steps < 1 or args.searches < 1:
        raise ValueError('games, steps, and searches must be positive.')

    openings = load_openings(args.openings, args.games)
    roots = [new_root(fen) for fen in openings]
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
    mcts.search([(new_root(openings[0]), False)])
    wait_for_synchronized_start(args)

    gpu_samples: list[GpuSample] = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu_until_stopped,
        args=(args.device, stop_event, gpu_samples),
        daemon=True,
    )
    sampler.start()

    terminal_roots = 0
    process_time_start = time.process_time()
    wall_time_start = time.perf_counter()
    for _ in range(args.steps):
        search_results = mcts.search([(root, False) for root in roots])
        next_roots: list[MCTSNode] = []
        for opening_index, result in enumerate(search_results.results):
            root = choose_root(result.root, result.visits)
            if root.is_terminal:
                terminal_roots += 1
                root = new_root(openings[opening_index])
            next_roots.append(root)
        roots = next_roots

    elapsed_seconds = time.perf_counter() - wall_time_start
    process_seconds = time.process_time() - process_time_start
    stop_event.set()
    sampler.join()

    inference_statistics, _ = mcts.get_inference_statistics()
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return BenchmarkResult(
        process_id=os.getpid(),
        device_id=args.device,
        games=args.games,
        steps=args.steps,
        searches_per_step=args.searches,
        elapsed_seconds=elapsed_seconds,
        positions_per_second=(args.games * args.steps) / elapsed_seconds,
        process_cpu_percent=100 * process_seconds / elapsed_seconds,
        peak_rss_mib=peak_rss_mib,
        mean_gpu_utilization_percent=(
            sum(sample.utilization_percent for sample in gpu_samples) / len(gpu_samples) if gpu_samples else None
        ),
        peak_gpu_memory_mib=(max(sample.memory_mib for sample in gpu_samples) if gpu_samples else None),
        terminal_roots=terminal_roots,
        inference_unique_positions=inference_statistics.uniquePositions,
        inference_cache_size_mib=inference_statistics.cacheSizeMB,
        inference_cache_fingerprint_collisions=inference_statistics.cacheFingerprintCollisions,
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--openings', required=True, type=Path)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--games', type=int, default=16)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--searches', type=int, default=600)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--maximum-batch-size', type=int, default=256)
    parser.add_argument('--inference-timeout-microseconds', type=int, default=500)
    parser.add_argument('--cache-capacity', type=int, default=100_000)
    parser.add_argument('--ready-file', type=Path)
    parser.add_argument('--start-barrier', type=Path)
    namespace = parser.parse_args()
    return Arguments(
        model=namespace.model,
        openings=namespace.openings,
        device=namespace.device,
        games=namespace.games,
        steps=namespace.steps,
        searches=namespace.searches,
        parallel_searches=namespace.parallel_searches,
        threads=namespace.threads,
        maximum_batch_size=namespace.maximum_batch_size,
        inference_timeout_microseconds=namespace.inference_timeout_microseconds,
        cache_capacity=namespace.cache_capacity,
        ready_file=namespace.ready_file,
        start_barrier=namespace.start_barrier,
    )


def main() -> None:
    result = run_benchmark(parse_arguments())
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == '__main__':
    main()
