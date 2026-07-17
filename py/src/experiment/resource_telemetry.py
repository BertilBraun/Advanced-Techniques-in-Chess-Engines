from __future__ import annotations

import multiprocessing
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict

from src.experiment.cost_accounting import CostCurrency, estimated_cost


class GpuTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    index: int
    name: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_celsius: float
    power_watts: float


class ResourceTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    timestamp_utc: datetime
    elapsed_seconds: float
    cost_currency: CostCurrency
    estimated_cost: float
    process_rss_mib: float
    process_cpu_percent: float
    child_process_count: int
    maximum_process_open_file_count: int
    total_open_file_count: int
    host_ram_percent: float
    host_ram_available_mib: float
    disk_free_gib: float
    gpus: tuple[GpuTelemetry, ...]


def parse_nvidia_smi_output(output: str) -> tuple[GpuTelemetry, ...]:
    samples: list[GpuTelemetry] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 7:
            raise ValueError(f'Expected seven nvidia-smi fields, received: {line!r}')
        samples.append(
            GpuTelemetry(
                index=int(fields[0]),
                name=fields[1],
                utilization_percent=float(fields[2]),
                memory_used_mib=float(fields[3]),
                memory_total_mib=float(fields[4]),
                temperature_celsius=float(fields[5]),
                power_watts=float(fields[6]),
            )
        )
    return tuple(samples)


def collect_gpu_telemetry() -> tuple[GpuTelemetry, ...]:
    completed = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_output(completed.stdout)


def process_open_file_count(process: psutil.Process) -> int:
    if os.name == 'posix':
        return process.num_fds()
    return process.num_handles()


def process_tree_open_file_counts(parent_process: psutil.Process) -> tuple[int, int]:
    counts: list[int] = []
    for process in (parent_process, *parent_process.children(recursive=True)):
        try:
            counts.append(process_open_file_count(process))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not counts:
        raise ValueError('No live processes were available for open-file monitoring.')
    return max(counts), sum(counts)


def collect_resource_telemetry(
    parent_process: psutil.Process,
    output_path: Path,
    started_at: float,
    cost_currency: CostCurrency,
    hourly_price: float,
) -> ResourceTelemetry:
    elapsed_seconds = time.monotonic() - started_at
    virtual_memory = psutil.virtual_memory()
    maximum_process_open_file_count, total_open_file_count = process_tree_open_file_counts(parent_process)
    return ResourceTelemetry(
        timestamp_utc=datetime.now(timezone.utc),
        elapsed_seconds=elapsed_seconds,
        cost_currency=cost_currency,
        estimated_cost=estimated_cost(hourly_price, elapsed_seconds),
        process_rss_mib=parent_process.memory_info().rss / 2**20,
        process_cpu_percent=parent_process.cpu_percent(interval=None),
        child_process_count=len(parent_process.children(recursive=True)),
        maximum_process_open_file_count=maximum_process_open_file_count,
        total_open_file_count=total_open_file_count,
        host_ram_percent=virtual_memory.percent,
        host_ram_available_mib=virtual_memory.available / 2**20,
        disk_free_gib=psutil.disk_usage(output_path).free / 2**30,
        gpus=collect_gpu_telemetry(),
    )


def record_resource_telemetry(
    parent_process_id: int,
    output_path: Path,
    started_at: float,
    cost_currency: CostCurrency,
    hourly_price: float,
    interval_seconds: float,
) -> None:
    parent_process = psutil.Process(parent_process_id)
    telemetry_path = output_path / 'resource-telemetry.jsonl'
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    while parent_process.is_running():
        sample = collect_resource_telemetry(
            parent_process=parent_process,
            output_path=output_path,
            started_at=started_at,
            cost_currency=cost_currency,
            hourly_price=hourly_price,
        )
        with telemetry_path.open('a', encoding='utf-8') as telemetry_file:
            telemetry_file.write(sample.model_dump_json() + '\n')
            telemetry_file.flush()
        time.sleep(interval_seconds)


def start_resource_telemetry(
    output_path: Path,
    started_at: float,
    cost_currency: CostCurrency,
    hourly_price: float,
    interval_seconds: float,
) -> multiprocessing.Process:
    process = multiprocessing.Process(
        target=record_resource_telemetry,
        args=(
            psutil.Process().pid,
            output_path,
            started_at,
            cost_currency,
            hourly_price,
            interval_seconds,
        ),
        daemon=True,
    )
    process.start()
    return process
