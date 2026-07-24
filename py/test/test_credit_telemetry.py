from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.experiment.credit_telemetry import (
    CreditEvaluationTelemetryStatus,
    CreditTrainingTelemetry,
    append_credit_training_telemetry,
    build_credit_training_telemetry,
    load_last_credit_training_telemetry,
)
from src.train.CreditTrainingLedger import CreditTrainingProgress


def _progress(
    *,
    optimizer_steps: int,
    quanta: int,
    credited_unique_samples: int,
    consumed_credits: int,
) -> CreditTrainingProgress:
    earned = Decimal(credited_unique_samples * 4)
    return CreditTrainingProgress(
        credited_unique_samples=credited_unique_samples,
        earned_position_credits=earned,
        consumed_position_credits=Decimal(consumed_credits),
        available_position_credits=earned - Decimal(consumed_credits),
        completed_optimizer_steps=optimizer_steps,
        completed_training_quanta=quanta,
        model_version=quanta,
        sampler_global_step=optimizer_steps,
    )


def _telemetry(
    previous: CreditTrainingProgress,
    current: CreditTrainingProgress,
    newest_source_model_version: int = 1,
) -> CreditTrainingTelemetry:
    return build_credit_training_telemetry(
        previous_progress=previous,
        progress=current,
        previous_credited_completed_searches=100_000,
        credited_completed_searches=900_000,
        live_replay_positions=current.credited_unique_samples,
        replay_capacity_unique_positions=100_000,
        optimizer_seconds=2,
        decode_seconds=1,
        loader_wait_seconds=3,
        credit_observation_seconds=4,
        replay_payload_open_count=4,
        replay_selected_rows=51_200,
        replay_rows_read=102_400,
        replay_selected_bytes=10,
        replay_bytes_read=30,
        replay_oldest_source_model_version=0,
        replay_newest_source_model_version=newest_source_model_version,
        replay_weighted_mean_source_model_version_midpoint=0.5,
        replay_oldest_position_age_seconds=100,
        replay_weighted_mean_position_age_seconds=50,
        publication_seconds=0.5,
        acknowledgement_seconds=0.25,
        global_batch_size=1_024,
        evaluation_source_model_version=20,
        evaluation_status=CreditEvaluationTelemetryStatus.ACTIVE,
    )


def test_credit_telemetry_axes_replay_ratios_and_io_statistics_are_exact() -> None:
    previous = _progress(
        optimizer_steps=0,
        quanta=0,
        credited_unique_samples=0,
        consumed_credits=0,
    )
    current = _progress(
        optimizer_steps=50,
        quanta=1,
        credited_unique_samples=12_800,
        consumed_credits=51_200,
    )

    telemetry = _telemetry(previous, current)

    assert telemetry.trained_position_presentations == 51_200
    assert telemetry.optimizer_step == 50
    assert telemetry.instantaneous_replay_ratio == 4
    assert telemetry.cumulative_replay_ratio == 4
    assert telemetry.replay_capacity_unique_positions == 100_000
    assert telemetry.optimizer_samples_per_second == 25_600
    assert telemetry.newly_credited_unique_positions_per_second == pytest.approx(12_800 / 4)
    assert telemetry.credited_completed_searches == 900_000
    assert telemetry.newly_credited_completed_searches_per_second == 200_000
    assert telemetry.credit_observation_seconds == 4
    assert telemetry.replay_row_read_amplification == 2
    assert telemetry.replay_byte_read_amplification == 3
    assert telemetry.replay_oldest_source_model_version == 0
    assert telemetry.replay_newest_source_model_version == 1
    assert telemetry.replay_weighted_mean_source_model_version_midpoint == 0.5
    assert telemetry.replay_oldest_source_model_staleness == 1
    assert telemetry.replay_newest_source_model_staleness == 0
    assert telemetry.replay_weighted_mean_source_model_staleness == 0.5
    assert telemetry.replay_oldest_position_age_seconds == 100
    assert telemetry.replay_weighted_mean_position_age_seconds == 50
    assert telemetry.console_summary() == (
        'Training update: model=1 optimizer_steps=50 trained_samples=51200 '
        'replay_positions=12800 replay_capacity=100000 available_credits=0 consumed_credits=51200 '
        'last_training_seconds=2.00 since_previous_training_seconds=3.00 '
        'generated_positions_per_second=3200.00 searches_per_second=200000.00'
    )


def test_credit_telemetry_rejects_replay_from_a_future_model_version() -> None:
    previous = CreditTrainingProgress.initial()
    current = _progress(
        optimizer_steps=50,
        quanta=1,
        credited_unique_samples=12_800,
        consumed_credits=51_200,
    )

    with pytest.raises(ValueError, match='cannot be newer'):
        _telemetry(previous, current, newest_source_model_version=2)


def test_credit_telemetry_restart_is_monotonic_and_repairs_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / 'credit-training-telemetry.jsonl'
    initial = CreditTrainingProgress.initial()
    first_progress = _progress(
        optimizer_steps=50,
        quanta=1,
        credited_unique_samples=12_800,
        consumed_credits=51_200,
    )
    second_progress = _progress(
        optimizer_steps=100,
        quanta=2,
        credited_unique_samples=25_600,
        consumed_credits=102_400,
    )
    first = _telemetry(initial, first_progress)
    second = _telemetry(first_progress, second_progress)

    append_credit_training_telemetry(path, first)
    with path.open('a', encoding='utf-8') as output:
        output.write('{"torn":')
    append_credit_training_telemetry(path, second)

    assert load_last_credit_training_telemetry(path) == second
    assert len(path.read_text(encoding='utf-8').splitlines()) == 2

    with pytest.raises(ValueError, match='axis must increase'):
        append_credit_training_telemetry(path, first)
