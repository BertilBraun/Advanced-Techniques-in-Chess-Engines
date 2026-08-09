from dataclasses import dataclass
from decimal import Decimal

from src.replay.manager import ReplayDescription
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedgerState


@dataclass(frozen=True)
class TrainingLifecycleTelemetry:
    configured_replay_ratio: float
    observed_replay_ratio: float
    materialized_samples: int
    consumed_presentations: float
    available_presentations: float
    required_presentations_per_quantum: int
    available_quantum_fraction: float
    live_replay_rows: int
    logical_replay_capacity: int
    replay_fill_fraction: float


def training_lifecycle_telemetry(
    state: CreditLedgerState,
    credit: CreditTrainingParams,
    replay: ReplayDescription,
    global_batch_size: int,
) -> TrainingLifecycleTelemetry:
    materialized_samples = state.earned_credits / credit.replay_ratio
    assert materialized_samples == materialized_samples.to_integral_value()
    materialized_sample_count = int(materialized_samples)
    observed_replay_ratio = (
        float(state.consumed_credits / materialized_samples) if materialized_samples > Decimal(0) else 0.0
    )
    required_presentations = credit.presentation_credits_per_quantum(global_batch_size)
    return TrainingLifecycleTelemetry(
        configured_replay_ratio=float(credit.replay_ratio),
        observed_replay_ratio=observed_replay_ratio,
        materialized_samples=materialized_sample_count,
        consumed_presentations=float(state.consumed_credits),
        available_presentations=float(state.available_credits),
        required_presentations_per_quantum=required_presentations,
        available_quantum_fraction=float(state.available_credits / Decimal(required_presentations)),
        live_replay_rows=replay.size,
        logical_replay_capacity=replay.logical_capacity,
        replay_fill_fraction=replay.size / replay.logical_capacity,
    )
