from __future__ import annotations

import math

from src.util.log import log, warn

MINIMUM_LIVE_WORKER_FRACTION = 0.5
DEGRADED_CAPACITY_GRACE_SECONDS = 1800.0


class SelfPlayHealthMonitor:
    """Decides whether reduced self-play capacity is still worth running or the run should stop cleanly."""

    def __init__(
        self,
        worker_count: int,
        minimum_live_worker_fraction: float = MINIMUM_LIVE_WORKER_FRACTION,
        grace_seconds: float = DEGRADED_CAPACITY_GRACE_SECONDS,
    ) -> None:
        self.worker_count = worker_count
        self.minimum_live_workers = max(1, math.ceil(minimum_live_worker_fraction * worker_count))
        self.grace_seconds = grace_seconds
        self._degraded_since: float | None = None
        self._reported_live_workers = worker_count

    def stop_reason(self, live_worker_count: int, now: float) -> str | None:
        self._report_capacity_change(live_worker_count)
        if live_worker_count >= self.minimum_live_workers:
            self._degraded_since = None
            return None
        if self._degraded_since is None:
            self._degraded_since = now
            return None
        degraded_seconds = now - self._degraded_since
        if degraded_seconds < self.grace_seconds:
            return None
        return (
            f'self-play capacity degraded: {live_worker_count} of {self.worker_count} workers alive '
            f'for {degraded_seconds / 60:.0f} minutes'
        )

    def _report_capacity_change(self, live_worker_count: int) -> None:
        if live_worker_count == self._reported_live_workers:
            return
        self._reported_live_workers = live_worker_count
        if live_worker_count < self.worker_count:
            warn(
                f'Self-play capacity reduced to {live_worker_count} of {self.worker_count} workers '
                f'(minimum {self.minimum_live_workers}).'
            )
        else:
            log(f'Self-play capacity restored to {self.worker_count} workers.')
