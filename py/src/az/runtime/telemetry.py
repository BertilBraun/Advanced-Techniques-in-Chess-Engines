from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSample:
    monotonic_seconds: float
    process_id: int
    cpu_time_seconds: float


def sample_process_resources() -> ResourceSample:
    times = os.times()
    return ResourceSample(
        monotonic_seconds=time.monotonic(),
        process_id=os.getpid(),
        cpu_time_seconds=times.user + times.system,
    )
