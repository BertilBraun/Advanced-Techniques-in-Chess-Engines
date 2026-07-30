from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerAssignment:
    worker_index: int
    device_index: int | None
    maximum_active_searches: int

    def __post_init__(self) -> None:
        if self.worker_index < 0:
            raise ValueError('Worker index cannot be negative.')
        if self.device_index is not None and self.device_index < 0:
            raise ValueError('Device index cannot be negative.')
        if self.maximum_active_searches <= 0:
            raise ValueError('Maximum active searches must be positive.')


@dataclass(frozen=True)
class RuntimeTopology:
    workers: tuple[WorkerAssignment, ...]
    trainer_device_indices: tuple[int, ...]
    evaluation_device_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.workers:
            raise ValueError('At least one self-play worker is required.')
        worker_indices = tuple(worker.worker_index for worker in self.workers)
        if worker_indices != tuple(range(len(self.workers))):
            raise ValueError('Worker indices must be contiguous from zero.')
        for name, devices in (
            ('Trainer', self.trainer_device_indices),
            ('Evaluation', self.evaluation_device_indices),
        ):
            if len(set(devices)) != len(devices):
                raise ValueError(f'{name} device indices must be unique.')
            if any(device < 0 for device in devices):
                raise ValueError(f'{name} device indices cannot be negative.')

    def validate_visible_cuda_devices(self, visible_device_count: int) -> None:
        if visible_device_count < 0:
            raise ValueError('Visible CUDA device count cannot be negative.')
        assigned = (
            *self.trainer_device_indices,
            *self.evaluation_device_indices,
            *(worker.device_index for worker in self.workers if worker.device_index is not None),
        )
        if assigned and max(assigned) >= visible_device_count:
            raise ValueError('Runtime topology assigns a CUDA device that is not visible.')
