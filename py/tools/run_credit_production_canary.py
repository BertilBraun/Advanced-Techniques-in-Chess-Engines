from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import IO

import psutil
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tools.benchmark_stochastic_self_play_cpp import BenchmarkResult
from tools.smoke_production_ddp import SmokeResult


class GpuCanaryTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    device_id: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    mean_utilization_percent: float = Field(ge=0)
    peak_utilization_percent: float = Field(ge=0)
    peak_memory_mib: float = Field(ge=0)
    total_memory_mib: float = Field(gt=0)
    minimum_oom_margin_mib: float = Field(ge=0)


class CreditProductionCanaryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    model_path: str
    model_version: int = Field(ge=0)
    device_ids: tuple[int, ...]
    processes_per_device: int = Field(gt=0)
    games_per_process: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    searches_per_turn: int = Field(gt=0)
    fast_searches_per_turn: int = Field(gt=0)
    search_threads_per_process: int = Field(gt=0)
    inference_workers_per_process: int = Field(gt=0)
    outstanding_batches_per_inference_worker: int = Field(gt=0)
    worker_results: tuple[BenchmarkResult, ...]
    ddp_result: SmokeResult
    aggregate_searches_per_second: float = Field(ge=0)
    aggregate_generated_unique_positions_per_hour: float = Field(ge=0)
    aggregate_completed_games_per_hour: float = Field(ge=0)
    aggregate_peak_rss_mib: float = Field(ge=0)
    mean_host_cpu_percent: float = Field(ge=0)
    peak_host_memory_percent: float = Field(ge=0)
    gpu_telemetry: tuple[GpuCanaryTelemetry, ...]


@dataclass(frozen=True)
class Arguments:
    model_path: Path
    model_version: int
    device_ids: tuple[int, ...]
    processes_per_device: int
    games_per_process: int
    duration_seconds: float
    work_directory: Path
    output_path: Path
    monitor_interval_seconds: float


@dataclass
class WorkerProcess:
    process: subprocess.Popen[str]
    log_file: IO[str]
    log_path: Path


@dataclass(frozen=True)
class ResourceSample:
    aggregate_rss_mib: float
    host_cpu_percent: float
    host_memory_percent: float
    gpu_samples: tuple[tuple[int, float, float, float], ...]


class ResourceMonitor:
    def __init__(self, device_ids: tuple[int, ...], interval_seconds: float) -> None:
        self.device_ids = device_ids
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[ResourceSample] = []
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name='credit-production-canary-monitor')

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Canary resource monitor did not stop.')
        if self.error is not None:
            raise RuntimeError('Canary resource monitor failed.') from self.error

    def _run(self) -> None:
        try:
            root_process = psutil.Process()
            while not self.stop_event.is_set():
                processes = (root_process, *root_process.children(recursive=True))
                aggregate_rss_mib = sum(_resident_memory_bytes(process) for process in processes) / 2**20
                completed = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits',
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                gpu_samples = tuple(
                    (
                        int(device_id),
                        float(utilization),
                        float(memory_used),
                        float(memory_total),
                    )
                    for device_id, utilization, memory_used, memory_total in (
                        tuple(field.strip() for field in line.split(',')) for line in completed.stdout.splitlines()
                    )
                    if int(device_id) in self.device_ids
                )
                self.samples.append(
                    ResourceSample(
                        aggregate_rss_mib=aggregate_rss_mib,
                        host_cpu_percent=psutil.cpu_percent(interval=None),
                        host_memory_percent=psutil.virtual_memory().percent,
                        gpu_samples=gpu_samples,
                    )
                )
                self.stop_event.wait(self.interval_seconds)
        except BaseException as error:
            self.error = error

    def aggregate(self) -> tuple[float, float, float, tuple[GpuCanaryTelemetry, ...]]:
        if not self.samples:
            raise RuntimeError('Canary resource monitor did not collect samples.')
        gpu_telemetry = tuple(self._gpu_telemetry(device_id) for device_id in self.device_ids)
        return (
            max(sample.aggregate_rss_mib for sample in self.samples),
            sum(sample.host_cpu_percent for sample in self.samples) / len(self.samples),
            max(sample.host_memory_percent for sample in self.samples),
            gpu_telemetry,
        )

    def _gpu_telemetry(self, device_id: int) -> GpuCanaryTelemetry:
        samples = tuple(
            gpu_sample
            for resource_sample in self.samples
            for gpu_sample in resource_sample.gpu_samples
            if gpu_sample[0] == device_id
        )
        if not samples:
            raise RuntimeError(f'Canary monitor did not collect GPU {device_id}.')
        total_memory_mib = samples[0][3]
        peak_memory_mib = max(sample[2] for sample in samples)
        return GpuCanaryTelemetry(
            device_id=device_id,
            sample_count=len(samples),
            mean_utilization_percent=sum(sample[1] for sample in samples) / len(samples),
            peak_utilization_percent=max(sample[1] for sample in samples),
            peak_memory_mib=peak_memory_mib,
            total_memory_mib=total_memory_mib,
            minimum_oom_margin_mib=total_memory_mib - peak_memory_mib,
        )


def _resident_memory_bytes(process: psutil.Process) -> int:
    try:
        return process.memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def _parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run the mixed self-play and four-rank DDP credit canary.')
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--model-version', required=True, type=int)
    parser.add_argument('--device-ids', nargs='+', default=(3, 2, 1, 0), type=int)
    parser.add_argument('--processes-per-device', default=4, type=int)
    parser.add_argument('--games-per-process', default=1024, type=int)
    parser.add_argument('--duration-seconds', default=180.0, type=float)
    parser.add_argument('--work-directory', required=True, type=Path)
    parser.add_argument('--output-path', required=True, type=Path)
    parser.add_argument('--monitor-interval-seconds', default=1.0, type=float)
    namespace = parser.parse_args()
    arguments = Arguments(
        model_path=namespace.model_path,
        model_version=namespace.model_version,
        device_ids=tuple(namespace.device_ids),
        processes_per_device=namespace.processes_per_device,
        games_per_process=namespace.games_per_process,
        duration_seconds=namespace.duration_seconds,
        work_directory=namespace.work_directory,
        output_path=namespace.output_path,
        monitor_interval_seconds=namespace.monitor_interval_seconds,
    )
    _validate_arguments(arguments)
    return arguments


def _validate_arguments(arguments: Arguments) -> None:
    if not arguments.model_path.is_file() or not arguments.model_path.name.endswith('.jit.pt'):
        raise ValueError('Canary model must be an existing TorchScript .jit.pt file.')
    if arguments.model_version < 0:
        raise ValueError('Canary model version cannot be negative.')
    if not arguments.device_ids or len(set(arguments.device_ids)) != len(arguments.device_ids):
        raise ValueError('Canary device IDs must be nonempty and unique.')
    if arguments.processes_per_device <= 0 or arguments.games_per_process <= 0:
        raise ValueError('Canary process and game counts must be positive.')
    if arguments.duration_seconds <= 0 or arguments.monitor_interval_seconds <= 0:
        raise ValueError('Canary durations must be positive.')
    if arguments.work_directory.exists():
        raise ValueError(f'Canary work directory already exists: {arguments.work_directory}')
    if arguments.output_path.exists():
        raise ValueError(f'Canary output already exists: {arguments.output_path}')


def _source_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _start_workers(arguments: Arguments, python_root: Path) -> tuple[WorkerProcess, ...]:
    workers: list[WorkerProcess] = []
    worker_index = 0
    for device_id in arguments.device_ids:
        for _ in range(arguments.processes_per_device):
            ready_path = arguments.work_directory / f'worker-{worker_index:02d}.ready'
            log_path = arguments.work_directory / f'worker-{worker_index:02d}.log'
            log_file = log_path.open('x', encoding='utf-8')
            command = [
                sys.executable,
                'tools/benchmark_stochastic_self_play_cpp.py',
                '--model',
                str(arguments.model_path),
                '--device',
                str(device_id),
                '--games',
                str(arguments.games_per_process),
                '--warmup-steps',
                '0',
                '--duration-seconds',
                str(arguments.duration_seconds),
                '--searches',
                '600',
                '--fast-searches',
                '150',
                '--parallel-searches',
                '1',
                '--threads',
                '2',
                '--maximum-batch-size',
                '256',
                '--no-inference-cache',
                '--direct-inference-workers',
                '2',
                '--direct-inference-batch-size',
                '64',
                '--direct-outstanding-batches-per-worker',
                '2',
                '--seed',
                str(20_260_724 + worker_index),
                '--iteration',
                str(arguments.model_version),
                '--ready-file',
                str(ready_path),
                '--start-barrier',
                str(arguments.work_directory / 'start.barrier'),
            ]
            workers.append(
                WorkerProcess(
                    process=subprocess.Popen(
                        command,
                        cwd=python_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    ),
                    log_file=log_file,
                    log_path=log_path,
                )
            )
            worker_index += 1
    return tuple(workers)


def _wait_for_workers_ready(arguments: Arguments, workers: tuple[WorkerProcess, ...]) -> None:
    deadline = time.monotonic() + 300
    ready_paths = tuple(
        arguments.work_directory / f'worker-{worker_index:02d}.ready' for worker_index in range(len(workers))
    )
    while not all(path.exists() for path in ready_paths):
        failed_workers = tuple(worker.process.pid for worker in workers if worker.process.poll() is not None)
        if failed_workers:
            raise RuntimeError(f'Self-play workers failed before the canary barrier: {failed_workers}')
        if time.monotonic() >= deadline:
            raise TimeoutError('Self-play workers did not reach the canary barrier.')
        time.sleep(0.25)


def _run_ddp_smoke(arguments: Arguments, python_root: Path) -> SmokeResult:
    ddp_output_path = arguments.work_directory / 'ddp-results.json'
    ddp_log_path = arguments.work_directory / 'ddp.log'
    with ddp_log_path.open('x', encoding='utf-8') as output:
        completed = subprocess.run(
            [
                sys.executable,
                'tools/smoke_production_ddp.py',
                '--device-ids',
                *(str(device_id) for device_id in arguments.device_ids),
                '--samples',
                '51200',
                '--work-directory',
                str(arguments.work_directory / 'ddp-work'),
                '--output-path',
                str(ddp_output_path),
                '--monitor-interval-seconds',
                '0.25',
            ],
            cwd=python_root,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    if completed.returncode != 0:
        raise RuntimeError('DDP canary failed.')
    return TypeAdapter(SmokeResult).validate_json(ddp_output_path.read_text(encoding='utf-8'))


def _wait_for_workers(arguments: Arguments, workers: tuple[WorkerProcess, ...]) -> None:
    deadline = time.monotonic() + arguments.duration_seconds + 300
    for worker in workers:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError('Self-play workers exceeded the canary timeout.')
        exit_code = worker.process.wait(timeout=remaining_seconds)
        worker.log_file.close()
        if exit_code:
            raise RuntimeError(f'Self-play worker {worker.process.pid} exited with code {exit_code}.')


def _terminate_workers(workers: tuple[WorkerProcess, ...]) -> None:
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    for worker in workers:
        if worker.process.poll() is None:
            try:
                worker.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=10)
        if not worker.log_file.closed:
            worker.log_file.close()


def _load_worker_result(worker: WorkerProcess) -> BenchmarkResult:
    lines = worker.log_path.read_text(encoding='utf-8').splitlines()
    json_lines = tuple(line for line in lines if line.startswith('{') and line.endswith('}'))
    if not json_lines:
        raise ValueError(f'Self-play worker log does not contain a result: {worker.log_path}')
    return TypeAdapter(BenchmarkResult).validate_json(json_lines[-1])


def main() -> None:
    arguments = _parse_arguments()
    python_root = Path(__file__).resolve().parents[1]
    source_root = python_root.parent
    arguments.work_directory.mkdir(parents=True)
    workers: tuple[WorkerProcess, ...] = ()
    monitor = ResourceMonitor(arguments.device_ids, arguments.monitor_interval_seconds)
    monitor_started = False
    try:
        workers = _start_workers(arguments, python_root)
        _wait_for_workers_ready(arguments, workers)
        monitor.start()
        monitor_started = True
        (arguments.work_directory / 'start.barrier').touch()
        time.sleep(5)
        ddp_result = _run_ddp_smoke(arguments, python_root)
        _wait_for_workers(arguments, workers)
        monitor.stop()
        monitor_started = False
    except BaseException:
        _terminate_workers(workers)
        if monitor_started:
            monitor.stop()
        raise

    worker_results = tuple(_load_worker_result(worker) for worker in workers)
    peak_rss, mean_cpu, peak_memory, gpu_telemetry = monitor.aggregate()
    wall_seconds = max(result.elapsed_seconds for result in worker_results)
    result = CreditProductionCanaryResult(
        source_revision=_source_revision(source_root),
        model_path=str(arguments.model_path.resolve()),
        model_version=arguments.model_version,
        device_ids=arguments.device_ids,
        processes_per_device=arguments.processes_per_device,
        games_per_process=arguments.games_per_process,
        duration_seconds=arguments.duration_seconds,
        searches_per_turn=600,
        fast_searches_per_turn=150,
        search_threads_per_process=2,
        inference_workers_per_process=2,
        outstanding_batches_per_inference_worker=2,
        worker_results=worker_results,
        ddp_result=ddp_result,
        aggregate_searches_per_second=sum(result.searches_completed for result in worker_results) / wall_seconds,
        aggregate_generated_unique_positions_per_hour=(
            sum(result.generated_samples for result in worker_results) * 3600 / wall_seconds
        ),
        aggregate_completed_games_per_hour=(
            sum(result.completed_games for result in worker_results) * 3600 / wall_seconds
        ),
        aggregate_peak_rss_mib=peak_rss,
        mean_host_cpu_percent=mean_cpu,
        peak_host_memory_percent=peak_memory,
        gpu_telemetry=gpu_telemetry,
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(result.model_dump_json(indent=2) + '\n', encoding='utf-8')
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
