from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator
from typing_extensions import Self

from src.experiment_queue.scheduler import ResourceAssignment
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class ExecutionIdentity(FrozenModel):
    assignment: ResourceAssignment
    started_at: AwareDatetime
    pid: int = Field(gt=0)
    process_group_id: int = Field(gt=0)
    stdout_log: Path
    stderr_log: Path


class PendingExperimentStatus(FrozenModel):
    status: Literal['pending'] = 'pending'
    experiment_id: str
    queued_at: AwareDatetime


class RunningExperimentStatus(FrozenModel):
    status: Literal['running'] = 'running'
    experiment_id: str
    execution: ExecutionIdentity


class CompletedExperimentStatus(FrozenModel):
    status: Literal['completed'] = 'completed'
    experiment_id: str
    execution: ExecutionIdentity
    finished_at: AwareDatetime
    exit_code: Literal[0] = 0


class FailedExperimentStatus(FrozenModel):
    status: Literal['failed'] = 'failed'
    experiment_id: str
    execution: ExecutionIdentity
    finished_at: AwareDatetime
    exit_code: int | None
    reason: str = Field(min_length=1)


ExperimentStatus: TypeAlias = Annotated[
    PendingExperimentStatus | RunningExperimentStatus | CompletedExperimentStatus | FailedExperimentStatus,
    Field(discriminator='status'),
]


class QueueSummary(FrozenModel):
    schema_version: Literal[1] = 1
    queue_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    created_at: AwareDatetime
    updated_at: AwareDatetime
    experiments: tuple[ExperimentStatus, ...]

    @model_validator(mode='after')
    def validate_summary(self) -> Self:
        experiment_ids = tuple(experiment.experiment_id for experiment in self.experiments)
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError('Queue summary experiment IDs must be unique.')
        if self.updated_at < self.created_at:
            raise ValueError('Queue summary update time cannot precede its creation time.')
        return self


def load_queue_summary(path: Path) -> QueueSummary:
    parsed = json.loads(path.read_text(encoding='utf-8'))
    return TypeAdapter(QueueSummary).validate_python(parsed)


def write_queue_summary(path: Path, summary: QueueSummary) -> None:
    write_text_atomically(path, summary.model_dump_json(indent=2) + '\n')
