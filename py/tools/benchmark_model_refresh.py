from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import torch

from AlphaZeroCpp import (
    DirectSelfPlayInferenceParams,
    InferenceClientParams,
    MCTS,
    MCTSBoard,
    MCTSParams,
    MCTSRoot,
)


INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
BYTES_PER_MIB = 2**20


class InferenceMode(StrEnum):
    DIRECT = 'direct'
    QUEUED_CACHED = 'queued-cached'
    QUEUED_NONCACHED = 'queued-noncached'


@dataclass(frozen=True)
class Arguments:
    model_path: Path
    output_path: Path
    inference_mode: InferenceMode
    device_id: int
    measured_refreshes: int
    warmup_refreshes: int
    roots: int
    searches: int
    parallel_searches: int
    threads: int
    direct_workers: int
    direct_batch_size: int
    direct_outstanding_batches: int
    maximum_rss_growth_mib: float | None
    maximum_gpu_growth_mib: float | None
    monitor_interval_seconds: float


@dataclass(frozen=True)
class RefreshMeasurement:
    model_version: int
    refresh_seconds: float
    rss_before_mib: float
    rss_after_mib: float
    gpu_used_before_mib: float
    gpu_used_after_mib: float
    retained_root_searches: int


@dataclass(frozen=True)
class ResourcePeak:
    rss_mib: float
    gpu_used_mib: float


@dataclass(frozen=True)
class RefreshBenchmarkResult:
    source_revision: str
    model_path: str
    model_sha256: str
    device_name: str
    inference_mode: str
    measured_refreshes: int
    warmup_refreshes: int
    roots: int
    searches: int
    direct_workers: int
    direct_batch_size: int
    direct_outstanding_batches: int
    latency_mean_seconds: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    rss_growth_mib: float
    gpu_growth_mib: float
    peak_rss_mib: float
    peak_gpu_used_mib: float
    measurements: tuple[RefreshMeasurement, ...]


class ResourceMonitor:
    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[ResourcePeak] = []
        self.thread = threading.Thread(target=self._sample_until_stopped, name='model-refresh-monitor')

    def __enter__(self) -> ResourceMonitor:
        self.thread.start()
        return self

    def __exit__(
        self, exception_type: type[BaseException] | None, exception: BaseException | None, traceback: object
    ) -> None:
        self.stop_event.set()
        self.thread.join()

    def _sample_until_stopped(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.samples.append(
                ResourcePeak(
                    rss_mib=resident_memory_mib(),
                    gpu_used_mib=gpu_used_memory_mib(self.device),
                )
            )

    def peak(self) -> ResourcePeak:
        if not self.samples:
            return ResourcePeak(rss_mib=resident_memory_mib(), gpu_used_mib=gpu_used_memory_mib(self.device))
        return ResourcePeak(
            rss_mib=max(sample.rss_mib for sample in self.samples),
            gpu_used_mib=max(sample.gpu_used_mib for sample in self.samples),
        )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Benchmark transactional in-place self-play model refresh.')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument(
        '--inference-mode', type=InferenceMode, choices=tuple(InferenceMode), default=InferenceMode.DIRECT
    )
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--measured-refreshes', type=int, default=20)
    parser.add_argument('--warmup-refreshes', type=int, default=2)
    parser.add_argument('--roots', type=int, default=24)
    parser.add_argument('--searches', type=int, default=64)
    parser.add_argument('--parallel-searches', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--direct-workers', type=int, default=2)
    parser.add_argument('--direct-batch-size', type=int, default=64)
    parser.add_argument('--direct-outstanding-batches', type=int, default=1)
    parser.add_argument('--maximum-rss-growth-mib', type=float)
    parser.add_argument('--maximum-gpu-growth-mib', type=float)
    parser.add_argument('--monitor-interval-seconds', type=float, default=0.01)
    namespace = parser.parse_args()
    arguments = Arguments(
        model_path=namespace.model_path,
        output_path=namespace.output_path,
        inference_mode=namespace.inference_mode,
        device_id=namespace.device_id,
        measured_refreshes=namespace.measured_refreshes,
        warmup_refreshes=namespace.warmup_refreshes,
        roots=namespace.roots,
        searches=namespace.searches,
        parallel_searches=namespace.parallel_searches,
        threads=namespace.threads,
        direct_workers=namespace.direct_workers,
        direct_batch_size=namespace.direct_batch_size,
        direct_outstanding_batches=namespace.direct_outstanding_batches,
        maximum_rss_growth_mib=namespace.maximum_rss_growth_mib,
        maximum_gpu_growth_mib=namespace.maximum_gpu_growth_mib,
        monitor_interval_seconds=namespace.monitor_interval_seconds,
    )
    validate_arguments(arguments)
    return arguments


def validate_arguments(arguments: Arguments) -> None:
    if not arguments.model_path.is_file():
        raise ValueError(f'Model does not exist: {arguments.model_path}')
    if arguments.measured_refreshes <= 1 or arguments.warmup_refreshes < 0:
        raise ValueError('At least two measured refreshes and a nonnegative warm-up count are required.')
    if arguments.roots <= 0 or arguments.searches <= arguments.parallel_searches:
        raise ValueError('Roots must be positive and searches must exceed parallel searches.')
    if arguments.direct_workers <= 0 or arguments.direct_batch_size <= 0:
        raise ValueError('Direct worker and batch counts must be positive.')
    if arguments.monitor_interval_seconds <= 0.0:
        raise ValueError('The resource monitor interval must be positive.')


def resident_memory_mib() -> float:
    resident_pages = int(Path('/proc/self/statm').read_text(encoding='utf-8').split()[1])
    return resident_pages * os.sysconf('SC_PAGE_SIZE') / BYTES_PER_MIB


def gpu_used_memory_mib(device: torch.device) -> float:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / BYTES_PER_MIB


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def percentile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(BYTES_PER_MIB):
            digest.update(chunk)
    return digest.hexdigest()


def create_search(arguments: Arguments) -> MCTS:
    client_parameters = InferenceClientParams(
        arguments.device_id,
        currentModelPath=str(arguments.model_path),
        maxBatchSize=256,
        microsecondsTimeoutInferenceThread=500,
        cacheCapacity=4096,
    )
    search_parameters = MCTSParams(
        num_parallel_searches=arguments.parallel_searches,
        num_full_searches=arguments.searches,
        num_fast_searches=arguments.searches,
        c_param=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        min_visit_count=0,
        num_threads=arguments.threads,
    )
    direct_parameters = (
        DirectSelfPlayInferenceParams(
            arguments.direct_workers,
            arguments.direct_batch_size,
            arguments.direct_outstanding_batches,
        )
        if arguments.inference_mode is InferenceMode.DIRECT
        else None
    )
    return MCTS(
        client_parameters,
        search_parameters,
        use_inference_cache=arguments.inference_mode is InferenceMode.QUEUED_CACHED,
        direct_inference_params=direct_parameters,
        initial_model_version=0,
    )


def populate_roots(search: MCTS, root_count: int) -> list[MCTSRoot]:
    roots = [search.new_root(INITIAL_FEN) for _ in range(root_count)]
    results = search.search([MCTSBoard(root, False) for root in roots])
    return [result.root for result in results.results]


def verify_retained_roots(search: MCTS, roots: list[MCTSRoot]) -> None:
    results = search.search([MCTSBoard(root, False) for root in roots])
    if results.searchesCompleted != 0:
        raise RuntimeError('A pure model refresh invalidated retained search roots.')


def benchmark(arguments: Arguments) -> RefreshBenchmarkResult:
    device = torch.device('cuda', arguments.device_id)
    torch.cuda.set_device(device)
    search = create_search(arguments)
    roots = populate_roots(search, arguments.roots)
    next_version = 1

    for _ in range(arguments.warmup_refreshes):
        search.refresh_model(next_version, str(arguments.model_path))
        verify_retained_roots(search, roots)
        next_version += 1

    measurements: list[RefreshMeasurement] = []
    with ResourceMonitor(device, arguments.monitor_interval_seconds) as resource_monitor:
        for _ in range(arguments.measured_refreshes):
            synchronize(device)
            rss_before_mib = resident_memory_mib()
            gpu_before_mib = gpu_used_memory_mib(device)
            started_at = time.perf_counter()
            search.refresh_model(next_version, str(arguments.model_path))
            synchronize(device)
            refresh_seconds = time.perf_counter() - started_at
            rss_after_mib = resident_memory_mib()
            gpu_after_mib = gpu_used_memory_mib(device)
            retained_results = search.search([MCTSBoard(root, False) for root in roots])
            measurements.append(
                RefreshMeasurement(
                    model_version=next_version,
                    refresh_seconds=refresh_seconds,
                    rss_before_mib=rss_before_mib,
                    rss_after_mib=rss_after_mib,
                    gpu_used_before_mib=gpu_before_mib,
                    gpu_used_after_mib=gpu_after_mib,
                    retained_root_searches=retained_results.searchesCompleted,
                )
            )
            next_version += 1
    resource_peak = resource_monitor.peak()

    if any(measurement.retained_root_searches != 0 for measurement in measurements):
        raise RuntimeError('A pure model refresh invalidated retained search roots.')
    latencies = tuple(measurement.refresh_seconds for measurement in measurements)
    rss_growth_mib = measurements[-1].rss_after_mib - measurements[0].rss_after_mib
    gpu_growth_mib = measurements[-1].gpu_used_after_mib - measurements[0].gpu_used_after_mib
    if arguments.maximum_rss_growth_mib is not None and rss_growth_mib > arguments.maximum_rss_growth_mib:
        raise RuntimeError(
            f'RSS grew by {rss_growth_mib:.3f} MiB; limit is {arguments.maximum_rss_growth_mib:.3f} MiB.'
        )
    if arguments.maximum_gpu_growth_mib is not None and gpu_growth_mib > arguments.maximum_gpu_growth_mib:
        raise RuntimeError(
            f'GPU memory grew by {gpu_growth_mib:.3f} MiB; limit is {arguments.maximum_gpu_growth_mib:.3f} MiB.'
        )
    return RefreshBenchmarkResult(
        source_revision=subprocess.run(
            ('git', 'rev-parse', 'HEAD'),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        model_path=str(arguments.model_path.resolve()),
        model_sha256=file_sha256(arguments.model_path),
        device_name=torch.cuda.get_device_name(device),
        inference_mode=str(arguments.inference_mode),
        measured_refreshes=arguments.measured_refreshes,
        warmup_refreshes=arguments.warmup_refreshes,
        roots=arguments.roots,
        searches=arguments.searches,
        direct_workers=arguments.direct_workers,
        direct_batch_size=arguments.direct_batch_size,
        direct_outstanding_batches=arguments.direct_outstanding_batches,
        latency_mean_seconds=sum(latencies) / len(latencies),
        latency_p50_seconds=percentile(latencies, 0.50),
        latency_p95_seconds=percentile(latencies, 0.95),
        rss_growth_mib=rss_growth_mib,
        gpu_growth_mib=gpu_growth_mib,
        peak_rss_mib=resource_peak.rss_mib,
        peak_gpu_used_mib=resource_peak.gpu_used_mib,
        measurements=tuple(measurements),
    )


def main() -> None:
    arguments = parse_arguments()
    result = benchmark(arguments)
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(json.dumps(asdict(result), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(asdict(result), indent=2))


if __name__ == '__main__':
    main()
