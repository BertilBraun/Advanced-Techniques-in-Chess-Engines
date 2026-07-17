from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import monotonic

from pydantic import BaseModel, ConfigDict


class IterationTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    iteration: int
    games_at_wait_start: int
    games_at_wait_end: int
    games_generated_while_waiting: int
    wait_seconds: float
    games_per_wait_second: float
    replay_samples_loaded: int
    replay_games_loaded: int
    training_seconds: float
    elapsed_seconds: float
    estimated_cost_eur: float
    maximum_process_open_file_count: int
    total_open_file_count: int


class RunOutcomeStatus(str, Enum):
    COMPLETED = 'completed'
    STOPPED = 'stopped'
    FAILED = 'failed'


class RunOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    status: RunOutcomeStatus
    reason: str | None
    completed_at_utc: datetime
    elapsed_seconds: float
    estimated_cost_eur: float
    latest_checkpoint_iteration: int


def append_iteration_telemetry(path: Path, telemetry: IterationTelemetry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as telemetry_file:
        telemetry_file.write(telemetry.model_dump_json() + '\n')
        telemetry_file.flush()


def write_run_outcome(
    path: Path,
    status: RunOutcomeStatus,
    reason: str | None,
    started_at: float,
    hourly_price_eur: float,
    latest_checkpoint_iteration: int,
) -> None:
    elapsed_seconds = monotonic() - started_at
    outcome = RunOutcome(
        status=status,
        reason=reason,
        completed_at_utc=datetime.now(timezone.utc),
        elapsed_seconds=elapsed_seconds,
        estimated_cost_eur=hourly_price_eur * elapsed_seconds / 3600,
        latest_checkpoint_iteration=latest_checkpoint_iteration,
    )
    temporary_path = path.with_name(f'.{path.name}.tmp')
    temporary_path.write_text(
        outcome.model_dump_json(indent=2) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(path)
