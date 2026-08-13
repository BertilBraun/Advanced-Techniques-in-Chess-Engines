from decimal import Decimal
from pathlib import Path

import pytest

from src.games.representation import PackedPlaneLayout
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayDescription
from src.training.checkpoint import CheckpointReference
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedgerState
from src.training.telemetry import training_lifecycle_telemetry
from src.training.targets import TrainingTargetLayout


def test_training_lifecycle_telemetry_reports_credit_backlog_and_observed_ratio() -> None:
    checkpoint = CheckpointReference(
        generation=3,
        manifest_path=Path('checkpoint_3.json'),
        model_path=Path('model_3.pt'),
        optimizer_path=Path('optimizer_3.pt'),
        inference_model_path=Path('model_3.jit.pt'),
        inference_model_sha256='0' * 64,
    )
    state = CreditLedgerState(
        completed_optimizer_steps=1500,
        earned_credits=Decimal(600),
        consumed_credits=Decimal(480),
        active_checkpoint=checkpoint,
    )
    credit = CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=500,
        maximum_optimizer_steps=1_000_000,
        retained_checkpoint_interval_generations=1,
        self_play_backpressure_quanta=5,
    )
    replay = ReplayDescription(
        path=Path('replay.bin'),
        head=0,
        size=100,
        logical_capacity=250,
        maximum_capacity=1000,
        layout=ReplayLayout(
            packed_planes=PackedPlaneLayout(board_size=7, binary_plane_count=1, scalar_count=1),
            targets=TrainingTargetLayout(action_size=50, wdl_size=3, auxiliary_heads=()),
            maximum_policy_entries=50,
        ),
    )

    telemetry = training_lifecycle_telemetry(state, credit, replay, global_batch_size=2)

    assert telemetry.configured_replay_ratio == 4.0
    assert telemetry.observed_replay_ratio == 3.2
    assert telemetry.materialized_samples == 150
    assert telemetry.consumed_presentations == 480.0
    assert telemetry.available_presentations == 120.0
    assert telemetry.required_presentations_per_quantum == 1000
    assert telemetry.available_quantum_fraction == 0.12
    assert telemetry.live_replay_rows == 100
    assert telemetry.logical_replay_capacity == 250
    assert telemetry.replay_fill_fraction == 0.4


@pytest.mark.parametrize(
    ('available_credits', 'expected'),
    ((5_120_000, False), (5_120_001, True)),
)
def test_self_play_backpressure_starts_above_configured_quantum_surplus(
    available_credits: int,
    expected: bool,
) -> None:
    credit = CreditTrainingParams(
        replay_ratio=Decimal(8),
        optimizer_steps_per_quantum=500,
        maximum_optimizer_steps=1_000_000,
        retained_checkpoint_interval_generations=1,
        self_play_backpressure_quanta=5,
    )

    assert credit.requires_self_play_backpressure(Decimal(available_credits), global_batch_size=2048) is expected
