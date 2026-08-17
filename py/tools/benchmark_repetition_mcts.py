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
from typing import Literal

from AlphaZeroCpp import (
    BatchedInferenceParameters,
    ChessPosition,
    ChessSearchRoot,
    ChessSelfPlaySearch,
    ChessSelfPlaySearchRequest,
    FirstPlayUrgencyKind,
    FirstPlayUrgencyParameters,
    GameSearchVisit,
    InferenceConfiguration,
    MonteCarloGraphSearchParameters,
    MonteCarloTreeSearchParameters,
    SelfPlaySearchParameters,
    TreeSearchParameters,
)


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
    minimum_measurement_seconds: float
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
    inference_model_calls: int
    inference_model_positions: int
    inference_average_batch_size: float
    inference_batch_size_distribution: tuple[BatchSizeCount, ...]
    graph: GraphSearchMetrics


@dataclass(frozen=True)
class GraphSearchMetrics:
    transposition_table_probes: int = 0
    transposition_table_hits: int = 0
    transposition_links: int = 0
    unique_nodes_created: int = 0
    edges_created: int = 0
    evaluations_avoided: int = 0
    transposition_corrections: int = 0
    correction_clips: int = 0
    continued_transpositions: int = 0
    cycle_cutoffs: int = 0
    nodes_retained: int = 0
    nodes_reclaimed: int = 0
    edges_reclaimed: int = 0
    nodes_pruned: int = 0
    hash_collision_checks: int = 0
    identity_lookup_nanoseconds: int = 0
    reroot_nanoseconds: int = 0
    pruning_nanoseconds: int = 0


@dataclass(frozen=True)
class Arguments:
    model: Path
    openings: Path
    device: int
    games: int
    warmup_steps: int
    steps: int
    minimum_measurement_seconds: float
    searches: int
    parallel_searches: int
    maximum_batch_size: int
    gpu_sampling_interval_seconds: float
    ready_file: Path | None
    start_barrier: Path | None
    algorithm: Literal['tree', 'graph']
    transposition_value_threshold: float


@dataclass(frozen=True)
class SearchStepsResult:
    roots: list[ChessSearchRoot]
    terminal_roots: int
    searches_completed: int
    completed_steps: int
    graph: GraphSearchMetrics


def graph_metrics(roots: list[ChessSearchRoot]) -> GraphSearchMetrics:
    result = GraphSearchMetrics()
    for root in roots:
        statistics = root.graph_statistics
        result = add_graph_metrics(
            result,
            GraphSearchMetrics(
                transposition_table_probes=statistics.transposition_table_probes,
                transposition_table_hits=statistics.transposition_table_hits,
                transposition_links=statistics.transposition_links,
                unique_nodes_created=statistics.unique_nodes_created,
                edges_created=statistics.edges_created,
                evaluations_avoided=statistics.evaluations_avoided,
                transposition_corrections=statistics.transposition_corrections,
                correction_clips=statistics.correction_clips,
                continued_transpositions=statistics.continued_transpositions,
                cycle_cutoffs=statistics.cycle_cutoffs,
                nodes_retained=statistics.nodes_retained,
                nodes_reclaimed=statistics.nodes_reclaimed,
                edges_reclaimed=statistics.edges_reclaimed,
                nodes_pruned=statistics.nodes_pruned,
                hash_collision_checks=statistics.hash_collision_checks,
                identity_lookup_nanoseconds=statistics.identity_lookup_nanoseconds,
                reroot_nanoseconds=statistics.reroot_nanoseconds,
                pruning_nanoseconds=statistics.pruning_nanoseconds,
            ),
        )
    return result


def subtract_graph_metrics(after: GraphSearchMetrics, before: GraphSearchMetrics) -> GraphSearchMetrics:
    return GraphSearchMetrics(
        transposition_table_probes=after.transposition_table_probes - before.transposition_table_probes,
        transposition_table_hits=after.transposition_table_hits - before.transposition_table_hits,
        transposition_links=after.transposition_links - before.transposition_links,
        unique_nodes_created=after.unique_nodes_created - before.unique_nodes_created,
        edges_created=after.edges_created - before.edges_created,
        evaluations_avoided=after.evaluations_avoided - before.evaluations_avoided,
        transposition_corrections=after.transposition_corrections - before.transposition_corrections,
        correction_clips=after.correction_clips - before.correction_clips,
        continued_transpositions=after.continued_transpositions - before.continued_transpositions,
        cycle_cutoffs=after.cycle_cutoffs - before.cycle_cutoffs,
        nodes_retained=after.nodes_retained - before.nodes_retained,
        nodes_reclaimed=after.nodes_reclaimed - before.nodes_reclaimed,
        edges_reclaimed=after.edges_reclaimed - before.edges_reclaimed,
        nodes_pruned=after.nodes_pruned - before.nodes_pruned,
        hash_collision_checks=after.hash_collision_checks - before.hash_collision_checks,
        identity_lookup_nanoseconds=after.identity_lookup_nanoseconds - before.identity_lookup_nanoseconds,
        reroot_nanoseconds=after.reroot_nanoseconds - before.reroot_nanoseconds,
        pruning_nanoseconds=after.pruning_nanoseconds - before.pruning_nanoseconds,
    )


def add_graph_metrics(left: GraphSearchMetrics, right: GraphSearchMetrics) -> GraphSearchMetrics:
    return GraphSearchMetrics(
        transposition_table_probes=left.transposition_table_probes + right.transposition_table_probes,
        transposition_table_hits=left.transposition_table_hits + right.transposition_table_hits,
        transposition_links=left.transposition_links + right.transposition_links,
        unique_nodes_created=left.unique_nodes_created + right.unique_nodes_created,
        edges_created=left.edges_created + right.edges_created,
        evaluations_avoided=left.evaluations_avoided + right.evaluations_avoided,
        transposition_corrections=left.transposition_corrections + right.transposition_corrections,
        correction_clips=left.correction_clips + right.correction_clips,
        continued_transpositions=left.continued_transpositions + right.continued_transpositions,
        cycle_cutoffs=left.cycle_cutoffs + right.cycle_cutoffs,
        nodes_retained=left.nodes_retained + right.nodes_retained,
        nodes_reclaimed=left.nodes_reclaimed + right.nodes_reclaimed,
        edges_reclaimed=left.edges_reclaimed + right.edges_reclaimed,
        nodes_pruned=left.nodes_pruned + right.nodes_pruned,
        hash_collision_checks=left.hash_collision_checks + right.hash_collision_checks,
        identity_lookup_nanoseconds=left.identity_lookup_nanoseconds + right.identity_lookup_nanoseconds,
        reroot_nanoseconds=left.reroot_nanoseconds + right.reroot_nanoseconds,
        pruning_nanoseconds=left.pruning_nanoseconds + right.pruning_nanoseconds,
    )


def load_openings(path: Path, number_of_games: int) -> tuple[ChessPosition, ...]:
    openings = tuple(
        ChessPosition(line.rsplit('\t', maxsplit=1)[1])
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


def choose_root(result_root: ChessSearchRoot, visits: list[GameSearchVisit]) -> ChessSearchRoot:
    if not visits:
        raise ValueError('MCTS returned no visits for a nonterminal root.')
    child_index = max(range(len(visits)), key=lambda index: visits[index].visit_count)
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
    search: ChessSelfPlaySearch,
    roots: list[ChessSearchRoot],
    openings: tuple[ChessPosition, ...],
    steps: int,
    minimum_elapsed_seconds: float = 0.0,
) -> SearchStepsResult:
    terminal_roots = 0
    searches_completed = 0
    completed_steps = 0
    measured_graph = GraphSearchMetrics()
    start_time = time.perf_counter()
    while completed_steps < steps or time.perf_counter() - start_time < minimum_elapsed_seconds:
        visits_before = sum(root.visits for root in roots)
        graph_before = graph_metrics(roots)
        search_results = search.search([ChessSelfPlaySearchRequest(root, False) for root in roots])
        measured_graph = add_graph_metrics(
            measured_graph,
            subtract_graph_metrics(graph_metrics(roots), graph_before),
        )
        searches_completed += sum(result.root.visits for result in search_results.results) - visits_before

        next_roots: list[ChessSearchRoot] = []
        for opening_index, result in enumerate(search_results.results):
            root = choose_root(result.root, result.search_visits)
            if root.is_terminal:
                terminal_roots += 1
                root = search.new_root(openings[opening_index])
            next_roots.append(root)
        roots = next_roots
        completed_steps += 1
    return SearchStepsResult(
        roots=roots,
        terminal_roots=terminal_roots,
        searches_completed=searches_completed,
        completed_steps=completed_steps,
        graph=measured_graph,
    )


def run_benchmark(args: Arguments) -> BenchmarkResult:
    if args.games < 1 or args.steps < 1 or args.searches < 1:
        raise ValueError('games, steps, and searches must be positive.')
    if args.warmup_steps < 0:
        raise ValueError('warmup steps cannot be negative.')
    if args.minimum_measurement_seconds < 0:
        raise ValueError('minimum measurement seconds cannot be negative.')
    if args.gpu_sampling_interval_seconds < 0:
        raise ValueError('GPU sampling interval cannot be negative.')

    search_algorithm = (
        MonteCarloTreeSearchParameters()
        if args.algorithm == 'tree'
        else MonteCarloGraphSearchParameters(args.transposition_value_threshold)
    )
    search = ChessSelfPlaySearch(
        InferenceConfiguration(args.device, str(args.model)),
        SelfPlaySearchParameters(
            args.parallel_searches,
            args.searches,
            args.searches,
            TreeSearchParameters(
                1.0,
                FirstPlayUrgencyParameters(FirstPlayUrgencyKind.ZERO),
                0.0,
                1.0,
                search_algorithm,
            ),
            0.3,
            0.0,
        ),
        BatchedInferenceParameters(1, args.maximum_batch_size, 1),
    )
    openings = load_openings(args.openings, args.games)
    roots = [search.new_root(fen) for fen in openings]
    warmup_result = run_search_steps(search, roots, openings, args.warmup_steps)
    roots = warmup_result.roots
    warmup_inference_statistics = search.inference_statistics()
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
    measurement_result = run_search_steps(
        search,
        roots,
        openings,
        args.steps,
        args.minimum_measurement_seconds,
    )
    elapsed_seconds = time.perf_counter() - wall_time_start
    process_seconds = time.process_time() - process_time_start
    stop_event.set()
    if sampler is not None:
        sampler.join()

    inference_statistics = search.inference_statistics()
    measurement_evaluations = inference_statistics.evaluations - warmup_inference_statistics.evaluations
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
    completed_game_plies = args.games * measurement_result.completed_steps
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return BenchmarkResult(
        process_id=os.getpid(),
        device_id=args.device,
        games=args.games,
        warmup_steps=args.warmup_steps,
        measurement_steps=measurement_result.completed_steps,
        minimum_measurement_seconds=args.minimum_measurement_seconds,
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
        inference_model_calls=measurement_model_calls,
        inference_model_positions=measurement_model_positions,
        inference_average_batch_size=(
            measurement_model_positions / measurement_model_calls if measurement_model_calls else 0.0
        ),
        inference_batch_size_distribution=batch_size_distribution,
        graph=measurement_result.graph,
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--openings', required=True, type=Path)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--games', type=int, default=16)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--minimum-measurement-seconds', type=float, default=0.0)
    parser.add_argument('--searches', type=int, default=600)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--maximum-batch-size', type=int, default=256)
    parser.add_argument('--gpu-sampling-interval-seconds', type=float, default=1.0)
    parser.add_argument('--ready-file', type=Path)
    parser.add_argument('--start-barrier', type=Path)
    parser.add_argument('--algorithm', choices=('tree', 'graph'), default='tree')
    parser.add_argument('--transposition-value-threshold', type=float, default=0.01)
    namespace = parser.parse_args()
    return Arguments(
        model=namespace.model,
        openings=namespace.openings,
        device=namespace.device,
        games=namespace.games,
        warmup_steps=namespace.warmup_steps,
        steps=namespace.steps,
        minimum_measurement_seconds=namespace.minimum_measurement_seconds,
        searches=namespace.searches,
        parallel_searches=namespace.parallel_searches,
        maximum_batch_size=namespace.maximum_batch_size,
        gpu_sampling_interval_seconds=namespace.gpu_sampling_interval_seconds,
        ready_file=namespace.ready_file,
        start_barrier=namespace.start_barrier,
        algorithm=namespace.algorithm,
        transposition_value_threshold=namespace.transposition_value_threshold,
    )


def main() -> None:
    result = run_benchmark(parse_arguments())
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == '__main__':
    main()
