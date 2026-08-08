from pathlib import Path
import time

import psutil

from src.experiment.cost_accounting import estimated_cost
from src.experiment.resource_telemetry import process_tree_open_file_counts
from src.training.configuration import RuntimeLimits


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
        elapsed_seconds = time.monotonic() - self.run_started_at
        if elapsed_seconds >= self.limits.maximum_wall_time_seconds:
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
