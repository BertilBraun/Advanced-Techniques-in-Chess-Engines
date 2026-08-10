from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, field_validator

from src.experiment_queue.configuration import QueuedExperiment, ResourceSlot, slot_satisfies_request
from src.util.frozen_model import FrozenModel


class ResourceAssignment(FrozenModel):
    slot_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    cuda_devices: tuple[int, ...]
    cpu_affinity: tuple[int, ...] = Field(min_length=1)
    ram_limit_bytes: int = Field(gt=0)
    log_directory: Path

    @field_validator('cuda_devices', 'cpu_affinity')
    @classmethod
    def validate_device_set(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 for value in values):
            raise ValueError('Assigned device and CPU indices must be nonnegative.')
        if tuple(sorted(set(values))) != values:
            raise ValueError('Assigned device and CPU indices must be unique and strictly increasing.')
        return values


@dataclass(frozen=True)
class ScheduledExperiment:
    experiment: QueuedExperiment
    assignment: ResourceAssignment


def schedule_experiments(
    pending_experiments: tuple[QueuedExperiment, ...],
    available_slots: tuple[ResourceSlot, ...],
) -> tuple[ScheduledExperiment, ...]:
    remaining_slots = list(available_slots)
    scheduled: list[ScheduledExperiment] = []
    for experiment in pending_experiments:
        compatible_slot = next(
            (slot for slot in remaining_slots if slot_satisfies_request(slot, experiment.resources)),
            None,
        )
        if compatible_slot is None:
            continue
        assignment = create_assignment(experiment, compatible_slot)
        scheduled.append(ScheduledExperiment(experiment=experiment, assignment=assignment))
        remaining_slots.remove(compatible_slot)
    return tuple(scheduled)


def create_assignment(experiment: QueuedExperiment, slot: ResourceSlot) -> ResourceAssignment:
    if not slot_satisfies_request(slot, experiment.resources):
        raise ValueError(f'Slot {slot.slot_id!r} cannot satisfy experiment {experiment.experiment_id!r}.')
    return ResourceAssignment(
        slot_id=slot.slot_id,
        cuda_devices=slot.cuda_devices,
        cpu_affinity=slot.cpu_affinity[: experiment.resources.cpu_core_count],
        ram_limit_bytes=experiment.resources.ram_limit_bytes,
        log_directory=slot.log_directory,
    )
