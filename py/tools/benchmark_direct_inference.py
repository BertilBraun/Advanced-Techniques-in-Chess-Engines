from __future__ import annotations

import argparse
import hashlib
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class BenchmarkMode(str, Enum):
    DIRECT = 'direct'
    PIPELINE = 'pipeline'
    CACHED = 'cached'
    NONCACHED = 'noncached'
    REPLICAS = 'replicas'


class Comparison(str, Enum):
    DIRECT = 'direct'
    PIPELINE = 'pipeline'
    CACHED = 'cached'
    NONCACHED = 'noncached'
    REPLICAS = 'replicas'
    DIRECT_COMBINED_BATCH = 'direct_combined_batch'


@dataclass(frozen=True)
class Arguments:
    executable: Path
    model: Path
    output: Path
    device: str
    batch_sizes: tuple[int, ...]
    workers: tuple[int, ...]
    iterations: int
    seed: int


@dataclass(frozen=True)
class GpuSample:
    utilization_percent: int
    memory_used_mib: int


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class NativeBenchmarkResult(FrozenModel):
    batch_size: int
    checksum: float
    device: str
    elapsed_seconds: float
    iterations_per_worker: int
    mode: BenchmarkMode
    positions: int
    positions_per_second: float
    state_encoding_milliseconds: float
    state_encoding_positions_per_second: float
    state_generation_seed: int
    tensor_packing_milliseconds: float
    tensor_packing_positions_per_second: float
    workers: int


class BenchmarkResult(NativeBenchmarkResult):
    process_wall_seconds: float
    gpu_utilization_mean_percent: float | None
    gpu_utilization_peak_percent: int | None
    gpu_memory_peak_mib: int | None
    comparison: Comparison
    worker_batch_size: int


class BenchmarkFailure(FrozenModel):
    comparison: Comparison
    workers: int
    batch_size: int
    error: str


class Provenance(FrozenModel):
    command: tuple[str, ...]
    platform: str
    python: str
    compiler: str
    gpu: str
    torch: str
    executable_path: str
    executable_sha256: str
    model_path: str
    model_sha256: str
    model_bytes: int
    state_generation_seed: int


class Methodology(FrozenModel):
    iterations_per_worker: int
    states: str
    direct_combined_batch: str
    replicas: str
    pipeline: str
    gpu_sampling: str


class BenchmarkReport(FrozenModel):
    schema_version: int
    provenance: Provenance
    methodology: Methodology
    results: tuple[BenchmarkResult, ...]
    failures: tuple[BenchmarkFailure, ...]


def parse_integer_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(','))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('values must be positive integers')
    return values


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-sizes', default='16,32,48,50,64,128,256', type=parse_integer_list)
    parser.add_argument('--workers', default='1,2,3,4,5,6,7,8', type=parse_integer_list)
    parser.add_argument('--iterations', default=100, type=int)
    parser.add_argument('--seed', default=7, type=int)
    parsed = parser.parse_args()
    if parsed.iterations <= 0:
        parser.error('--iterations must be positive')
    return Arguments(
        executable=parsed.executable,
        model=parsed.model,
        output=parsed.output,
        device=parsed.device,
        batch_sizes=parsed.batch_sizes,
        workers=parsed.workers,
        iterations=parsed.iterations,
        seed=parsed.seed,
    )


def sample_gpu(stop_event: threading.Event, samples: list[GpuSample]) -> None:
    while not stop_event.wait(0.1):
        completed = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            continue
        utilization, memory_used = completed.stdout.strip().split(', ')
        samples.append(GpuSample(int(utilization), int(memory_used)))


def run_configuration(
    executable: Path,
    model: Path,
    device: str,
    mode: BenchmarkMode,
    comparison: Comparison,
    workers: int,
    batch_size: int,
    worker_batch_size: int,
    iterations: int,
    seed: int,
) -> BenchmarkResult:
    samples: list[GpuSample] = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop_event, samples))
    if device == 'cuda':
        sampler.start()
    started_at = time.perf_counter()
    completed = subprocess.run(
        [
            str(executable),
            '--model',
            str(model),
            '--mode',
            mode.value,
            '--device',
            device,
            '--batch-size',
            str(batch_size),
            '--workers',
            str(workers),
            '--iterations',
            str(iterations),
            '--seed',
            str(seed),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - started_at
    stop_event.set()
    if device == 'cuda':
        sampler.join()
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    native = NativeBenchmarkResult.model_validate_json(completed.stdout.strip())
    return BenchmarkResult(
        batch_size=native.batch_size,
        checksum=native.checksum,
        device=native.device,
        elapsed_seconds=native.elapsed_seconds,
        iterations_per_worker=native.iterations_per_worker,
        mode=native.mode,
        positions=native.positions,
        positions_per_second=native.positions_per_second,
        state_encoding_milliseconds=native.state_encoding_milliseconds,
        state_encoding_positions_per_second=native.state_encoding_positions_per_second,
        state_generation_seed=native.state_generation_seed,
        tensor_packing_milliseconds=native.tensor_packing_milliseconds,
        tensor_packing_positions_per_second=native.tensor_packing_positions_per_second,
        workers=native.workers,
        process_wall_seconds=wall_seconds,
        gpu_utilization_mean_percent=(
            statistics.fmean(sample.utilization_percent for sample in samples) if samples else None
        ),
        gpu_utilization_peak_percent=(max(sample.utilization_percent for sample in samples) if samples else None),
        gpu_memory_peak_mib=(max(sample.memory_used_mib for sample in samples) if samples else None),
        comparison=comparison,
        worker_batch_size=worker_batch_size,
    )


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else 'unavailable'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configurations(
    batch_size: int, worker_counts: tuple[int, ...]
) -> tuple[tuple[Comparison, BenchmarkMode, int, int], ...]:
    values = [
        (Comparison.DIRECT, BenchmarkMode.DIRECT, 1, batch_size),
        (Comparison.PIPELINE, BenchmarkMode.PIPELINE, 1, batch_size),
        (Comparison.CACHED, BenchmarkMode.CACHED, 1, batch_size),
        (Comparison.NONCACHED, BenchmarkMode.NONCACHED, 1, batch_size),
    ]
    values.extend((Comparison.REPLICAS, BenchmarkMode.REPLICAS, workers, batch_size) for workers in worker_counts)
    values.extend(
        (
            Comparison.DIRECT_COMBINED_BATCH,
            BenchmarkMode.DIRECT,
            1,
            workers * batch_size,
        )
        for workers in worker_counts
        if workers > 1
    )
    return tuple(values)


def main() -> None:
    arguments = parse_arguments()
    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []
    for worker_batch_size in arguments.batch_sizes:
        for comparison, mode, workers, batch_size in configurations(worker_batch_size, arguments.workers):
            try:
                result = run_configuration(
                    arguments.executable,
                    arguments.model,
                    arguments.device,
                    mode,
                    comparison,
                    workers,
                    batch_size,
                    worker_batch_size,
                    arguments.iterations,
                    arguments.seed,
                )
                results.append(result)
                print(result.model_dump_json(), flush=True)
            except RuntimeError as error:
                failure = BenchmarkFailure(
                    comparison=comparison,
                    workers=workers,
                    batch_size=batch_size,
                    error=str(error),
                )
                failures.append(failure)
                print(failure.model_dump_json(), flush=True)

    report = BenchmarkReport(
        schema_version=1,
        provenance=Provenance(
            command=tuple(sys.argv),
            platform=platform.platform(),
            python=platform.python_version(),
            compiler=command_output(['c++', '--version']).splitlines()[0],
            gpu=command_output(
                [
                    'nvidia-smi',
                    '--query-gpu=name,driver_version,memory.total',
                    '--format=csv,noheader',
                ]
            ),
            torch=command_output(
                [
                    'python',
                    '-c',
                    'import torch; print(torch.__version__, torch.version.cuda)',
                ]
            ),
            executable_path=str(arguments.executable.resolve()),
            executable_sha256=sha256(arguments.executable),
            model_path=str(arguments.model.resolve()),
            model_sha256=sha256(arguments.model),
            model_bytes=arguments.model.stat().st_size,
            state_generation_seed=arguments.seed,
        ),
        methodology=Methodology(
            iterations_per_worker=arguments.iterations,
            states='seeded random legal games, generated and compressed before timed inference',
            direct_combined_batch='one model forward with batch workers * worker_batch_size',
            replicas='one model replica and dedicated CUDA stream per worker',
            pipeline='one producer and one model-owning thread with three reusable SPSC slots',
            gpu_sampling=(
                'nvidia-smi process-level sampling every 100 ms, including model load and warmup; '
                'use only as a coarse indicator'
            ),
        ),
        results=tuple(results),
        failures=tuple(failures),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
