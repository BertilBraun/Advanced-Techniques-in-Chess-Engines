from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import monotonic

from pydantic import BaseModel, ConfigDict

from src.experiment.cost_accounting import CostCurrency, estimated_cost
from src.util.atomic_file import write_text_atomically


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
    cost_currency: CostCurrency
    estimated_cost: float
    latest_checkpoint_model_version: int


def write_run_outcome(
    path: Path,
    status: RunOutcomeStatus,
    reason: str | None,
    started_at: float,
    cost_currency: CostCurrency,
    hourly_price: float,
    latest_checkpoint_model_version: int,
) -> None:
    elapsed_seconds = monotonic() - started_at
    outcome = RunOutcome(
        status=status,
        reason=reason,
        completed_at_utc=datetime.now(timezone.utc),
        elapsed_seconds=elapsed_seconds,
        cost_currency=cost_currency,
        estimated_cost=estimated_cost(hourly_price, elapsed_seconds),
        latest_checkpoint_model_version=latest_checkpoint_model_version,
    )
    write_text_atomically(
        path,
        outcome.model_dump_json(indent=2) + '\n',
    )
