from __future__ import annotations

from typing import Annotated, Literal
from base64 import b64decode, b64encode

from pydantic import Field

from src.az.config.base import FrozenModel
from src.az.replay.envelope import ReplayEnvelope, ReplayRecord


class IpcReplayRecord(FrozenModel):
    envelope: ReplayEnvelope
    payload_base64: str = Field(min_length=1)

    @classmethod
    def from_record(cls, record: ReplayRecord) -> IpcReplayRecord:
        return cls(
            envelope=record.envelope,
            payload_base64=b64encode(record.payload).decode('ascii'),
        )

    def to_record(self) -> ReplayRecord:
        try:
            payload = b64decode(self.payload_base64, validate=True)
        except ValueError as error:
            raise ValueError('IPC replay payload is not canonical base64.') from error
        if b64encode(payload).decode('ascii') != self.payload_base64:
            raise ValueError('IPC replay payload must use canonical base64.')
        return ReplayRecord(envelope=self.envelope, payload=payload)


class WorkerReady(FrozenModel):
    kind: Literal['worker_ready']
    worker_index: int = Field(ge=0)
    process_id: int = Field(gt=0)
    model_version: int = Field(ge=0)


class WorkerProgress(FrozenModel):
    kind: Literal['worker_progress']
    worker_index: int = Field(ge=0)
    completed_games_total: int = Field(ge=0)
    emitted_positions_total: int = Field(ge=0)
    model_version: int = Field(ge=0)
    interval_inference_batches: int = Field(ge=0)
    interval_inference_requests: int = Field(ge=0)
    interval_maximum_inference_batch_size: int = Field(ge=0)
    interval_total_inference_wait_microseconds: int = Field(ge=0)
    interval_inference_cache_hits: int = Field(ge=0)
    monotonic_seconds: float = Field(ge=0)


class WorkerResourceSample(FrozenModel):
    kind: Literal['worker_resource_sample']
    worker_index: int = Field(ge=0)
    monotonic_seconds: float = Field(ge=0)
    cpu_time_seconds: float = Field(ge=0)
    device_memory_bytes: int = Field(ge=0)


class WorkerRecords(FrozenModel):
    kind: Literal['worker_records']
    worker_index: int = Field(ge=0)
    records: tuple[IpcReplayRecord, ...] = Field(min_length=1)


class WorkerPublished(FrozenModel):
    kind: Literal['worker_published']
    worker_index: int = Field(ge=0)
    committed_games: int = Field(gt=0)
    committed_positions: int = Field(gt=0)
    partial_shard: bool
    shard_sequence: int = Field(ge=0)


class WorkerPublicationAborted(FrozenModel):
    kind: Literal['worker_publication_aborted']
    worker_index: int = Field(ge=0)
    discarded_positions: int = Field(gt=0)


class WorkerModelRefreshed(FrozenModel):
    kind: Literal['worker_model_refreshed']
    worker_index: int = Field(ge=0)
    previous_model_version: int = Field(ge=0)
    model_version: int = Field(gt=0)
    checkpoint_id: str = Field(min_length=1)


class WorkerStopped(FrozenModel):
    kind: Literal['worker_stopped']
    worker_index: int = Field(ge=0)
    completed_games: int = Field(ge=0)
    emitted_positions: int = Field(ge=0)


class WorkerFailure(FrozenModel):
    kind: Literal['worker_failure']
    worker_index: int = Field(ge=0)
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RuntimeFailure(FrozenModel):
    kind: Literal['runtime_failure']
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


RuntimeMessage = Annotated[
    WorkerReady
    | WorkerProgress
    | WorkerResourceSample
    | WorkerRecords
    | WorkerPublished
    | WorkerPublicationAborted
    | WorkerModelRefreshed
    | WorkerStopped
    | WorkerFailure
    | RuntimeFailure,
    Field(discriminator='kind'),
]
