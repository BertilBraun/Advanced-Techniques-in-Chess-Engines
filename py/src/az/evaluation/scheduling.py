from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointClaim:
    requested_elapsed_seconds: int
    detected_elapsed_seconds: float


class ElapsedCheckpointScheduler:
    def __init__(self, experiment_epoch_monotonic_ns: int, requested_seconds: tuple[int, ...]) -> None:
        if experiment_epoch_monotonic_ns <= 0:
            raise ValueError('Experiment epoch must be positive.')
        if tuple(sorted(set(requested_seconds))) != requested_seconds or any(value <= 0 for value in requested_seconds):
            raise ValueError('Requested checkpoint times must be positive and strictly increasing.')
        self._epoch = experiment_epoch_monotonic_ns
        self._requested = requested_seconds

    def due(self, now_monotonic_ns: int, completed_requested_seconds: frozenset[int]) -> tuple[CheckpointClaim, ...]:
        elapsed = max(0.0, (now_monotonic_ns - self._epoch) / 1_000_000_000)
        return tuple(
            CheckpointClaim(requested_elapsed_seconds=requested, detected_elapsed_seconds=elapsed)
            for requested in self._requested
            if requested <= elapsed and requested not in completed_requested_seconds
        )
