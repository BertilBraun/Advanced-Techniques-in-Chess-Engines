from pathlib import Path
import os
import time

import psutil
from pydantic import Field

from src.util.frozen_model import FrozenModel


class RuntimeLimits(FrozenModel):
    hourly_price: float = Field(ge=0.0)
    maximum_cost: float | None = Field(default=None, gt=0.0)
    maximum_wall_time_seconds: float | None = Field(default=None, gt=0.0)
    manual_stop_file: Path | None = None
    maximum_open_file_count: int = Field(gt=0)
    maximum_host_ram_percent: float = Field(gt=0.0, le=100.0)
    minimum_free_disk_gib: float = Field(ge=0.0)
    resource_telemetry_interval_seconds: float = Field(gt=0.0)


def estimated_cost(hourly_price: float, elapsed_seconds: float) -> float:
    return hourly_price * elapsed_seconds / 3600


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


class RunLimitMonitor:
    def __init__(
        self,
        limits: RuntimeLimits,
        run_path: Path,
        run_started_at: float,
        parent_process: psutil.Process | None = None,
    ) -> None:
        self.limits = limits
        self.run_path = run_path
        self.run_started_at = run_started_at
        self.parent_process = psutil.Process() if parent_process is None else parent_process

    def stop_reason(self) -> str | None:
        if self.limits.manual_stop_file is not None and self.limits.manual_stop_file.exists():
            return 'manual stop requested'
        elapsed_seconds = time.monotonic() - self.run_started_at
        if (
            self.limits.maximum_wall_time_seconds is not None
            and elapsed_seconds >= self.limits.maximum_wall_time_seconds
        ):
            return 'maximum wall time reached'
        cost = estimated_cost(self.limits.hourly_price, elapsed_seconds)
        if self.limits.maximum_cost is not None and cost >= self.limits.maximum_cost:
            return 'maximum cost reached'
        maximum_open_files, _ = process_tree_open_file_counts(self.parent_process)
        if maximum_open_files >= self.limits.maximum_open_file_count:
            return 'maximum open file count reached'
        if psutil.virtual_memory().percent >= self.limits.maximum_host_ram_percent:
            return 'maximum host RAM usage reached'
        if psutil.disk_usage(self.run_path).free / 2**30 <= self.limits.minimum_free_disk_gib:
            return 'minimum free disk space reached'
        return None
