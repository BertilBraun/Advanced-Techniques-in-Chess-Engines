from __future__ import annotations

import json
from collections.abc import Hashable
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from src.util.frozen_model import FrozenModel


UniqueValueT = TypeVar('UniqueValueT', bound=Hashable)


class RunnerCommand(FrozenModel):
    command: tuple[str, ...] = Field(min_length=1)
    experiment_path_argument: str = Field(default='--run-config', min_length=1)

    @field_validator('command')
    @classmethod
    def validate_command_parts(cls, command: tuple[str, ...]) -> tuple[str, ...]:
        if any(not part.strip() for part in command):
            raise ValueError('Runner command parts must not be empty.')
        return command


class ResourceRequest(FrozenModel):
    cuda_device_count: int = Field(ge=0)
    cpu_core_count: int = Field(gt=0)
    ram_limit_bytes: int = Field(gt=0)


class ResourceSlot(FrozenModel):
    slot_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    cuda_devices: tuple[int, ...] = ()
    cpu_affinity: tuple[int, ...] = Field(min_length=1)
    ram_capacity_bytes: int = Field(gt=0)
    working_directory: Path
    log_directory: Path

    @field_validator('cuda_devices', 'cpu_affinity')
    @classmethod
    def validate_device_set(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 for value in values):
            raise ValueError('Device and CPU indices must be nonnegative.')
        if tuple(sorted(set(values))) != values:
            raise ValueError('Device and CPU indices must be unique and strictly increasing.')
        return values


class QueuedExperiment(FrozenModel):
    experiment_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    experiment_file: Path
    resources: ResourceRequest


class QueueConfiguration(FrozenModel):
    schema_version: Literal[1] = 1
    runner: RunnerCommand
    slots: tuple[ResourceSlot, ...] = Field(min_length=1)
    experiments: tuple[QueuedExperiment, ...] = Field(min_length=1)
    summary_path: Path
    poll_interval_seconds: float = Field(default=0.1, gt=0.0)
    termination_grace_seconds: float = Field(default=10.0, gt=0.0)
    wait_for_updates_when_empty: bool = False

    @model_validator(mode='after')
    def validate_queue(self) -> Self:
        _require_unique_values('slot IDs', tuple(slot.slot_id for slot in self.slots))
        _require_unique_values('experiment IDs', tuple(item.experiment_id for item in self.experiments))
        _require_unique_values('experiment files', tuple(item.experiment_file for item in self.experiments))
        _validate_slot_exclusivity(self.slots)
        for experiment in self.experiments:
            if not any(slot_satisfies_request(slot, experiment.resources) for slot in self.slots):
                raise ValueError(f'Experiment {experiment.experiment_id!r} has no compatible resource slot.')
        return self


def slot_satisfies_request(slot: ResourceSlot, request: ResourceRequest) -> bool:
    return (
        len(slot.cuda_devices) == request.cuda_device_count
        and len(slot.cpu_affinity) >= request.cpu_core_count
        and slot.ram_capacity_bytes >= request.ram_limit_bytes
    )


def load_queue_configuration(path: Path) -> QueueConfiguration:
    payload = path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(payload) if path.suffix.casefold() in {'.yaml', '.yml'} else json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f'Queue file must contain a mapping: {path}')
    configuration = QueueConfiguration.model_validate(parsed)
    return _resolve_configuration_paths(configuration, path.resolve().parent)


def _resolve_configuration_paths(configuration: QueueConfiguration, base_directory: Path) -> QueueConfiguration:
    slots = tuple(
        ResourceSlot(
            slot_id=slot.slot_id,
            cuda_devices=slot.cuda_devices,
            cpu_affinity=slot.cpu_affinity,
            ram_capacity_bytes=slot.ram_capacity_bytes,
            working_directory=_resolve_path(slot.working_directory, base_directory),
            log_directory=_resolve_path(slot.log_directory, base_directory),
        )
        for slot in configuration.slots
    )
    experiments = tuple(
        QueuedExperiment(
            experiment_id=experiment.experiment_id,
            experiment_file=_resolve_path(experiment.experiment_file, base_directory),
            resources=experiment.resources,
        )
        for experiment in configuration.experiments
    )
    return QueueConfiguration(
        schema_version=configuration.schema_version,
        runner=configuration.runner,
        slots=slots,
        experiments=experiments,
        summary_path=_resolve_path(configuration.summary_path, base_directory),
        poll_interval_seconds=configuration.poll_interval_seconds,
        termination_grace_seconds=configuration.termination_grace_seconds,
        wait_for_updates_when_empty=configuration.wait_for_updates_when_empty,
    )


def _resolve_path(path: Path, base_directory: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


def _require_unique_values(label: str, values: tuple[UniqueValueT, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f'Queue {label} must be unique.')


def _validate_slot_exclusivity(slots: tuple[ResourceSlot, ...]) -> None:
    claimed_cuda_devices: set[int] = set()
    claimed_cpu_cores: set[int] = set()
    for slot in slots:
        overlapping_cuda_devices = claimed_cuda_devices.intersection(slot.cuda_devices)
        if overlapping_cuda_devices:
            raise ValueError(f'Resource slots share CUDA devices: {sorted(overlapping_cuda_devices)}.')
        overlapping_cpu_cores = claimed_cpu_cores.intersection(slot.cpu_affinity)
        if overlapping_cpu_cores:
            raise ValueError(f'Resource slots share CPU cores: {sorted(overlapping_cpu_cores)}.')
        claimed_cuda_devices.update(slot.cuda_devices)
        claimed_cpu_cores.update(slot.cpu_affinity)
