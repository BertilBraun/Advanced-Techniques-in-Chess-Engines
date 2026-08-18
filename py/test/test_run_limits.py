from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest

from src.training.run_limits import RunLimitMonitor, RuntimeLimits
import src.training.run_limits as run_limits_module


@dataclass(frozen=True)
class MemoryUsage:
    percent: float


@dataclass(frozen=True)
class DiskUsage:
    free: int


def limits(**updates: float | int | Path | None) -> RuntimeLimits:
    values: dict[str, float | int | Path | None] = {
        'hourly_price': 2.0,
        'maximum_cost': 10.0,
        'maximum_wall_time_seconds': 100.0,
        'maximum_open_file_count': 100,
        'maximum_host_ram_percent': 90.0,
        'minimum_free_disk_gib': 5.0,
        'resource_telemetry_interval_seconds': 10.0,
    }
    values.update(updates)
    return RuntimeLimits.model_validate(values)


@pytest.mark.parametrize(
    ('configured_limits', 'elapsed_seconds', 'expected'),
    (
        (limits(), 101.0, 'maximum wall time reached'),
        (limits(maximum_cost=1.0, maximum_wall_time_seconds=2_000.0), 1_801.0, 'maximum cost reached'),
    ),
)
def test_run_limit_monitor_enforces_time_and_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_limits: RuntimeLimits,
    elapsed_seconds: float,
    expected: str,
) -> None:
    monkeypatch.setattr(run_limits_module.time, 'monotonic', lambda: elapsed_seconds)
    monitor = RunLimitMonitor(configured_limits, tmp_path, 0.0, psutil.Process())

    assert monitor.stop_reason() == expected


def test_run_limit_monitor_allows_unbounded_time_and_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_limits = limits(maximum_cost=None, maximum_wall_time_seconds=None)
    monkeypatch.setattr(run_limits_module.time, 'monotonic', lambda: 1_000_000.0)
    monkeypatch.setattr(run_limits_module, 'process_tree_open_file_counts', lambda process: (0, 0))
    monkeypatch.setattr(run_limits_module.psutil, 'virtual_memory', lambda: MemoryUsage(0.0))
    monkeypatch.setattr(run_limits_module.psutil, 'disk_usage', lambda path: DiskUsage(100 * 2**30))
    monitor = RunLimitMonitor(configured_limits, tmp_path, 0.0, psutil.Process())

    assert monitor.stop_reason() is None


def test_run_limit_monitor_honors_manual_stop_file(tmp_path: Path) -> None:
    stop_file = tmp_path / 'stop-requested'
    stop_file.touch()
    monitor = RunLimitMonitor(limits(manual_stop_file=stop_file), tmp_path, 0.0, psutil.Process())

    assert monitor.stop_reason() == 'manual stop requested'


def test_run_limit_monitor_enforces_host_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_limits_module.time, 'monotonic', lambda: 1.0)
    monkeypatch.setattr(run_limits_module, 'process_tree_open_file_counts', lambda process: (100, 100))
    monkeypatch.setattr(run_limits_module.psutil, 'virtual_memory', lambda: MemoryUsage(10.0))
    monkeypatch.setattr(run_limits_module.psutil, 'disk_usage', lambda path: DiskUsage(100 * 2**30))
    monitor = RunLimitMonitor(limits(), tmp_path, 0.0, psutil.Process())

    assert monitor.stop_reason() == 'maximum open file count reached'
