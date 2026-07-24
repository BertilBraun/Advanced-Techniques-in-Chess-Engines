from __future__ import annotations

from decimal import Decimal
from enum import Enum
import os
from pathlib import Path
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.train.CreditTrainingLedger import CreditTrainingProgress
from src.train.TrainingStats import TrainingStats
from src.util.tensorboard import log_scalar


class CreditEvaluationTelemetryStatus(str, Enum):
    IDLE = 'idle'
    ACTIVE = 'active'
    RETRY_PENDING = 'retry_pending'
    SUCCEEDED = 'succeeded'
    PERMANENT_FAILURE = 'permanent_failure'
    INTERRUPTED = 'interrupted'
    COALESCED = 'coalesced'


class CreditTrainingTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    trained_position_presentations: int = Field(ge=0)
    optimizer_step: int = Field(ge=0)
    training_quantum: int = Field(ge=0)
    model_version: int = Field(ge=0)
    generated_unique_positions: int = Field(ge=0)
    credited_unique_positions: int = Field(ge=0)
    earned_position_credits: Decimal = Field(ge=0)
    consumed_position_credits: Decimal = Field(ge=0)
    available_position_credits: Decimal = Field(ge=0)
    instantaneous_replay_ratio: float = Field(ge=0)
    cumulative_replay_ratio: float = Field(ge=0)
    live_replay_positions: int = Field(ge=0)
    optimizer_seconds: float = Field(ge=0)
    optimizer_samples_per_second: float = Field(ge=0)
    newly_credited_unique_positions_per_second: float = Field(ge=0)
    credit_observation_seconds: float = Field(ge=0)
    decode_seconds: float = Field(ge=0)
    loader_wait_seconds: float = Field(ge=0)
    replay_payload_open_count: int = Field(ge=0)
    replay_selected_rows: int = Field(ge=0)
    replay_rows_read: int = Field(ge=0)
    replay_selected_bytes: int = Field(ge=0)
    replay_bytes_read: int = Field(ge=0)
    replay_oldest_source_model_version: int | None = Field(default=None, ge=0)
    replay_newest_source_model_version: int | None = Field(default=None, ge=0)
    replay_weighted_mean_source_model_version_midpoint: float | None = Field(default=None, ge=0)
    replay_oldest_source_model_staleness: int | None = None
    replay_newest_source_model_staleness: int | None = None
    replay_weighted_mean_source_model_staleness: float | None = None
    replay_oldest_position_age_seconds: float | None = Field(default=None, ge=0)
    replay_weighted_mean_position_age_seconds: float | None = Field(default=None, ge=0)
    publication_seconds: float = Field(ge=0)
    acknowledgement_seconds: float = Field(ge=0)
    evaluation_source_model_version: int | None = Field(default=None, ge=0)
    evaluation_status: CreditEvaluationTelemetryStatus

    def console_summary(self) -> str:
        return (
            f'Training update: model={self.model_version} '
            f'optimizer_steps={self.optimizer_step} '
            f'trained_samples={self.trained_position_presentations} '
            f'replay_positions={self.live_replay_positions} '
            f'available_credits={self.available_position_credits} '
            f'consumed_credits={self.consumed_position_credits} '
            f'last_training_seconds={self.optimizer_seconds:.2f} '
            f'since_previous_training_seconds={self.loader_wait_seconds:.2f} '
            f'generated_positions_per_second={self.newly_credited_unique_positions_per_second:.2f}'
        )

    @model_validator(mode='after')
    def validate_axes(self) -> CreditTrainingTelemetry:
        if self.training_quantum != self.model_version:
            raise ValueError('Credit telemetry quantum and model version must agree.')
        replay_statistics = (
            self.replay_oldest_source_model_version,
            self.replay_newest_source_model_version,
            self.replay_weighted_mean_source_model_version_midpoint,
            self.replay_oldest_source_model_staleness,
            self.replay_newest_source_model_staleness,
            self.replay_weighted_mean_source_model_staleness,
            self.replay_oldest_position_age_seconds,
            self.replay_weighted_mean_position_age_seconds,
        )
        if any(value is None for value in replay_statistics) and any(value is not None for value in replay_statistics):
            raise ValueError('Replay age and model-version statistics must be present together.')
        return self

    def log_to_tensorboard(self, training_stats: TrainingStats) -> None:
        step = self.trained_position_presentations
        training_stats.log_to_tensorboard(step, 'train')
        scalar_values = (
            ('credit/optimizer_step', self.optimizer_step),
            ('credit/training_quantum', self.training_quantum),
            ('credit/model_version', self.model_version),
            ('credit/generated_unique_positions', self.generated_unique_positions),
            ('credit/credited_unique_positions', self.credited_unique_positions),
            ('credit/earned_position_credits', float(self.earned_position_credits)),
            ('credit/consumed_position_credits', float(self.consumed_position_credits)),
            ('credit/available_position_credits', float(self.available_position_credits)),
            ('credit/instantaneous_replay_ratio', self.instantaneous_replay_ratio),
            ('credit/cumulative_replay_ratio', self.cumulative_replay_ratio),
            ('credit/live_replay_positions', self.live_replay_positions),
            ('credit/optimizer_seconds', self.optimizer_seconds),
            ('credit/optimizer_samples_per_second', self.optimizer_samples_per_second),
            (
                'credit/newly_credited_unique_positions_per_second',
                self.newly_credited_unique_positions_per_second,
            ),
            ('credit/credit_observation_seconds', self.credit_observation_seconds),
            ('credit/decode_seconds', self.decode_seconds),
            ('credit/loader_wait_seconds', self.loader_wait_seconds),
            ('credit/replay_payload_open_count', self.replay_payload_open_count),
            ('credit/replay_row_read_amplification', self.replay_row_read_amplification),
            ('credit/replay_byte_read_amplification', self.replay_byte_read_amplification),
            ('credit/publication_seconds', self.publication_seconds),
            ('credit/acknowledgement_seconds', self.acknowledgement_seconds),
        )
        for name, value in scalar_values:
            log_scalar(name, value, step)
        optional_scalar_values = (
            ('credit/replay_oldest_source_model_version', self.replay_oldest_source_model_version),
            ('credit/replay_newest_source_model_version', self.replay_newest_source_model_version),
            (
                'credit/replay_weighted_mean_source_model_version_midpoint',
                self.replay_weighted_mean_source_model_version_midpoint,
            ),
            ('credit/replay_oldest_source_model_staleness', self.replay_oldest_source_model_staleness),
            ('credit/replay_newest_source_model_staleness', self.replay_newest_source_model_staleness),
            (
                'credit/replay_weighted_mean_source_model_staleness',
                self.replay_weighted_mean_source_model_staleness,
            ),
            ('credit/replay_oldest_position_age_seconds', self.replay_oldest_position_age_seconds),
            (
                'credit/replay_weighted_mean_position_age_seconds',
                self.replay_weighted_mean_position_age_seconds,
            ),
        )
        for name, value in optional_scalar_values:
            if value is not None:
                log_scalar(name, value, step)
        if self.evaluation_source_model_version is not None:
            log_scalar(
                'credit/evaluation_source_model_version',
                self.evaluation_source_model_version,
                step,
            )
        log_scalar(
            'credit/evaluation_status',
            _evaluation_status_code(self.evaluation_status),
            step,
        )

    @property
    def replay_row_read_amplification(self) -> float:
        return self.replay_rows_read / self.replay_selected_rows if self.replay_selected_rows else 0.0

    @property
    def replay_byte_read_amplification(self) -> float:
        return self.replay_bytes_read / self.replay_selected_bytes if self.replay_selected_bytes else 0.0


def build_credit_training_telemetry(
    previous_progress: CreditTrainingProgress,
    progress: CreditTrainingProgress,
    live_replay_positions: int,
    optimizer_seconds: float,
    decode_seconds: float,
    loader_wait_seconds: float,
    credit_observation_seconds: float,
    replay_payload_open_count: int,
    replay_selected_rows: int,
    replay_rows_read: int,
    replay_selected_bytes: int,
    replay_bytes_read: int,
    replay_oldest_source_model_version: int | None,
    replay_newest_source_model_version: int | None,
    replay_weighted_mean_source_model_version_midpoint: float | None,
    replay_oldest_position_age_seconds: float | None,
    replay_weighted_mean_position_age_seconds: float | None,
    publication_seconds: float,
    acknowledgement_seconds: float,
    global_batch_size: int,
    evaluation_source_model_version: int | None,
    evaluation_status: CreditEvaluationTelemetryStatus,
) -> CreditTrainingTelemetry:
    replay_source_model_versions = (
        replay_oldest_source_model_version,
        replay_newest_source_model_version,
        replay_weighted_mean_source_model_version_midpoint,
    )
    if any(
        source_model_version is not None and source_model_version > progress.model_version
        for source_model_version in replay_source_model_versions
    ):
        raise ValueError('Replay source model version cannot be newer than the trained model version.')
    presentations = progress.completed_optimizer_steps * global_batch_size
    newly_credited = progress.credited_unique_samples - previous_progress.credited_unique_samples
    newly_consumed = progress.consumed_position_credits - previous_progress.consumed_position_credits
    instantaneous_replay_ratio = float(newly_consumed / newly_credited) if newly_credited else 0.0
    cumulative_replay_ratio = (
        float(progress.consumed_position_credits / progress.credited_unique_samples)
        if progress.credited_unique_samples
        else 0.0
    )
    return CreditTrainingTelemetry(
        trained_position_presentations=presentations,
        optimizer_step=progress.completed_optimizer_steps,
        training_quantum=progress.completed_training_quanta,
        model_version=progress.model_version,
        generated_unique_positions=progress.credited_unique_samples,
        credited_unique_positions=progress.credited_unique_samples,
        earned_position_credits=progress.earned_position_credits,
        consumed_position_credits=progress.consumed_position_credits,
        available_position_credits=progress.available_position_credits,
        instantaneous_replay_ratio=instantaneous_replay_ratio,
        cumulative_replay_ratio=cumulative_replay_ratio,
        live_replay_positions=live_replay_positions,
        optimizer_seconds=optimizer_seconds,
        optimizer_samples_per_second=float(newly_consumed) / optimizer_seconds if optimizer_seconds else 0.0,
        newly_credited_unique_positions_per_second=(
            newly_credited / credit_observation_seconds if credit_observation_seconds else 0.0
        ),
        credit_observation_seconds=credit_observation_seconds,
        decode_seconds=decode_seconds,
        loader_wait_seconds=loader_wait_seconds,
        replay_payload_open_count=replay_payload_open_count,
        replay_selected_rows=replay_selected_rows,
        replay_rows_read=replay_rows_read,
        replay_selected_bytes=replay_selected_bytes,
        replay_bytes_read=replay_bytes_read,
        replay_oldest_source_model_version=replay_oldest_source_model_version,
        replay_newest_source_model_version=replay_newest_source_model_version,
        replay_weighted_mean_source_model_version_midpoint=replay_weighted_mean_source_model_version_midpoint,
        replay_oldest_source_model_staleness=(
            progress.model_version - replay_oldest_source_model_version
            if replay_oldest_source_model_version is not None
            else None
        ),
        replay_newest_source_model_staleness=(
            progress.model_version - replay_newest_source_model_version
            if replay_newest_source_model_version is not None
            else None
        ),
        replay_weighted_mean_source_model_staleness=(
            progress.model_version - replay_weighted_mean_source_model_version_midpoint
            if replay_weighted_mean_source_model_version_midpoint is not None
            else None
        ),
        replay_oldest_position_age_seconds=replay_oldest_position_age_seconds,
        replay_weighted_mean_position_age_seconds=replay_weighted_mean_position_age_seconds,
        publication_seconds=publication_seconds,
        acknowledgement_seconds=acknowledgement_seconds,
        evaluation_source_model_version=evaluation_source_model_version,
        evaluation_status=evaluation_status,
    )


def append_credit_training_telemetry(
    path: Path,
    telemetry: CreditTrainingTelemetry,
) -> None:
    _repair_torn_jsonl_tail(path)
    previous = load_last_credit_training_telemetry(path)
    if previous is not None:
        if telemetry.trained_position_presentations <= previous.trained_position_presentations:
            raise ValueError('Trained-position telemetry axis must increase across restart.')
        if telemetry.optimizer_step <= previous.optimizer_step:
            raise ValueError('Optimizer-step telemetry axis must increase across restart.')
        if telemetry.training_quantum <= previous.training_quantum:
            raise ValueError('Training-quantum telemetry axis must increase across restart.')
        if telemetry.model_version <= previous.model_version:
            raise ValueError('Model-version telemetry axis must increase across restart.')
        if telemetry.credited_unique_positions < previous.credited_unique_positions:
            raise ValueError('Credited unique positions cannot decrease across restart.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as output:
        output.write(telemetry.model_dump_json() + '\n')
        output.flush()
        os.fsync(output.fileno())


def load_last_credit_training_telemetry(path: Path) -> CreditTrainingTelemetry | None:
    if not path.exists():
        return None
    last_telemetry: CreditTrainingTelemetry | None = None
    with path.open('r', encoding='utf-8') as source:
        for line in source:
            if line.strip():
                last_telemetry = CreditTrainingTelemetry.model_validate_json(line)
    return last_telemetry


def _evaluation_status_code(status: CreditEvaluationTelemetryStatus) -> float:
    match status:
        case CreditEvaluationTelemetryStatus.IDLE:
            return 0.0
        case CreditEvaluationTelemetryStatus.ACTIVE:
            return 1.0
        case CreditEvaluationTelemetryStatus.RETRY_PENDING:
            return 2.0
        case CreditEvaluationTelemetryStatus.SUCCEEDED:
            return 3.0
        case CreditEvaluationTelemetryStatus.PERMANENT_FAILURE:
            return 4.0
        case CreditEvaluationTelemetryStatus.INTERRUPTED:
            return 5.0
        case CreditEvaluationTelemetryStatus.COALESCED:
            return 6.0


def _repair_torn_jsonl_tail(path: Path) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding='utf-8').splitlines()
    valid_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            CreditTrainingTelemetry.model_validate_json(line)
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError('Credit telemetry contains corruption before its final record.')
            temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
            temporary_path.write_text(
                ''.join(f'{valid_line}\n' for valid_line in valid_lines),
                encoding='utf-8',
            )
            os.replace(temporary_path, path)
            return
        valid_lines.append(line)
