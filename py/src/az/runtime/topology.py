from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerAssignment:
    worker_index: int
    device_worker_index: int
    device_index: int | None
    search_thread_count: int
    maximum_active_searches: int

    def __post_init__(self) -> None:
        if self.worker_index < 0:
            raise ValueError("Worker index cannot be negative.")
        if self.device_worker_index < 0:
            raise ValueError("Per-device worker index cannot be negative.")
        if self.device_index is not None and self.device_index < 0:
            raise ValueError("Device index cannot be negative.")
        if self.search_thread_count <= 0:
            raise ValueError("Search thread count must be positive.")
        if self.maximum_active_searches <= 0:
            raise ValueError("Maximum active searches must be positive.")


@dataclass(frozen=True)
class RuntimeTopology:
    workers: tuple[WorkerAssignment, ...]
    optimizer_active_worker_ids: tuple[int, ...]
    trainer_device_indices: tuple[int, ...]
    evaluation_device_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.workers:
            raise ValueError("At least one self-play worker is required.")
        worker_indices = tuple(worker.worker_index for worker in self.workers)
        if worker_indices != tuple(range(len(self.workers))):
            raise ValueError("Worker indices must be contiguous from zero.")
        if (
            tuple(sorted(set(self.optimizer_active_worker_ids)))
            != self.optimizer_active_worker_ids
        ):
            raise ValueError(
                "Optimizer-active worker IDs must be unique and increasing."
            )
        if not self.optimizer_active_worker_ids or self.optimizer_active_worker_ids[
            -1
        ] >= len(self.workers):
            raise ValueError(
                "Optimizer-active worker IDs must identify configured workers."
            )
        device_worker_indices: dict[int | None, list[int]] = {}
        for worker in self.workers:
            device_worker_indices.setdefault(worker.device_index, []).append(
                worker.device_worker_index
            )
        for device_index, worker_ids in device_worker_indices.items():
            if tuple(worker_ids) != tuple(range(len(worker_ids))):
                raise ValueError(
                    f"Per-device worker indices must be contiguous for device {device_index}."
                )
            active_count = sum(
                self.workers[worker_id].device_index == device_index
                for worker_id in self.optimizer_active_worker_ids
            )
            if active_count != 1:
                raise ValueError(
                    "Exactly one worker per device must remain active during optimizer quanta."
                )
        for name, devices in (
            ("Trainer", self.trainer_device_indices),
            ("Evaluation", self.evaluation_device_indices),
        ):
            if len(set(devices)) != len(devices):
                raise ValueError(f"{name} device indices must be unique.")
            if any(device < 0 for device in devices):
                raise ValueError(f"{name} device indices cannot be negative.")

    @property
    def optimizer_paused_worker_ids(self) -> tuple[int, ...]:
        active = set(self.optimizer_active_worker_ids)
        return tuple(
            worker.worker_index
            for worker in self.workers
            if worker.worker_index not in active
        )

    def validate_visible_cuda_devices(self, visible_device_count: int) -> None:
        if visible_device_count < 0:
            raise ValueError("Visible CUDA device count cannot be negative.")
        assigned = (
            *self.trainer_device_indices,
            *self.evaluation_device_indices,
            *(
                worker.device_index
                for worker in self.workers
                if worker.device_index is not None
            ),
        )
        if assigned and max(assigned) >= visible_device_count:
            raise ValueError(
                "Runtime topology assigns a CUDA device that is not visible."
            )
