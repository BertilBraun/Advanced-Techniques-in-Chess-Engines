from __future__ import annotations

import pytest
from src.training.self_play_health import SelfPlayHealthMonitor


@pytest.mark.parametrize(
    'worker_count,expected_minimum',
    [(1, 1), (2, 1), (4, 2), (5, 3), (8, 4)],
)
def test_minimum_live_workers_is_half_the_group_rounded_up(worker_count: int, expected_minimum: int) -> None:
    monitor = SelfPlayHealthMonitor(worker_count)

    assert monitor.minimum_live_workers == expected_minimum


def test_full_capacity_reports_no_stop_reason() -> None:
    monitor = SelfPlayHealthMonitor(4, grace_seconds=0.0)

    assert monitor.stop_reason(4, now=1000.0) is None


def test_capacity_above_the_minimum_reports_no_stop_reason() -> None:
    monitor = SelfPlayHealthMonitor(4, grace_seconds=0.0)

    assert monitor.stop_reason(2, now=1000.0) is None


def test_degraded_capacity_is_tolerated_inside_the_grace_period() -> None:
    monitor = SelfPlayHealthMonitor(4, grace_seconds=600.0)
    monitor.stop_reason(1, now=1000.0)

    assert monitor.stop_reason(1, now=1500.0) is None


def test_degraded_capacity_beyond_the_grace_period_reports_a_stop_reason() -> None:
    monitor = SelfPlayHealthMonitor(4, grace_seconds=600.0)
    monitor.stop_reason(1, now=1000.0)

    assert monitor.stop_reason(1, now=1601.0) == 'self-play capacity degraded: 1 of 4 workers alive for 10 minutes'


def test_recovered_capacity_resets_the_grace_period() -> None:
    monitor = SelfPlayHealthMonitor(4, grace_seconds=600.0)
    monitor.stop_reason(0, now=1000.0)
    monitor.stop_reason(4, now=1100.0)

    assert monitor.stop_reason(1, now=1700.0) is None


def test_a_single_worker_group_stops_once_its_only_worker_stays_dead() -> None:
    monitor = SelfPlayHealthMonitor(1, grace_seconds=0.0)
    monitor.stop_reason(0, now=1000.0)

    assert monitor.stop_reason(0, now=1001.0) is not None
