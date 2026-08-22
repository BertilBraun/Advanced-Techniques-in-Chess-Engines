from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from src.training.checkpoint import CheckpointReference
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedger
from src.training.distributions import PolicyTrainingDistribution, TrainingDistributionSnapshot
from src.training.trainer import TrainingQuantumResult, TrainingStatistics
from test_helpers.checkpoints import checkpoint_reference, materialized_checkpoint


def _empty_distributions() -> TrainingDistributionSnapshot:
    policy = PolicyTrainingDistribution(
        loss=(),
        target_top1_mass=(),
        target_top2_mass=(),
        target_top3_mass=(),
        target_entropy=(),
        prediction_entropy=(),
    )
    return TrainingDistributionSnapshot(
        policy=policy,
        wdl_loss=(),
        root_value=(),
        terminal_value=(),
        predicted_value=(),
        value_absolute_error=(),
        sample_weight=(),
        replay_generation_age=(),
        replay_age_seconds=(),
        auxiliary=(),
    )


def _checkpoint(tmp_path: Path, generation: int) -> CheckpointReference:
    return checkpoint_reference(tmp_path, generation)


def _parameters() -> CreditTrainingParams:
    return CreditTrainingParams(
        replay_ratio=Decimal(2),
        optimizer_steps_per_quantum=4,
        maximum_optimizer_steps=8,
        retained_checkpoint_interval_generations=1,
        self_play_backpressure_quanta=5,
    )


def test_credit_ledger_persists_only_approximate_counters_and_active_checkpoint(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_materialized_samples(16)

    assert ledger.can_train_quantum(live_samples=8)
    result = TrainingQuantumResult(
        completed_optimizer_steps=4,
        checkpoint=_checkpoint(tmp_path, 1),
        statistics=TrainingStatistics(
            policy_loss=1.0,
            wdl_loss=1.0,
            auxiliary_losses=(),
            total_loss=2.0,
            gradient_norm=0.5,
            training_samples_per_second=1.0,
            elapsed_seconds=1.0,
            distributions=_empty_distributions(),
        ),
    )
    ledger.commit_quantum(result)

    restarted = CreditLedger(
        tmp_path,
        _parameters(),
        global_batch_size=8,
        starting_checkpoint=_checkpoint(tmp_path, 0),
    )
    assert restarted.model_generation == 1
    assert restarted.state.available_credits == 0
    assert restarted.state.active_checkpoint == result.checkpoint


def test_credit_ledger_without_optimizer_limit_never_completes(tmp_path: Path) -> None:
    parameters = _parameters().model_copy(update={'maximum_optimizer_steps': None})
    ledger = CreditLedger(tmp_path, parameters, global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))

    assert not ledger.training_complete


def test_credit_ledger_separates_credit_gate_from_live_sample_gate(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))

    assert not ledger.has_quantum_credits
    ledger.add_materialized_samples(16)
    assert ledger.has_quantum_credits
    assert not ledger.can_train_quantum(live_samples=7)
    assert ledger.can_train_quantum(live_samples=8)


def test_credit_ledger_rejects_negative_materialized_sample_count(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))

    with pytest.raises(ValueError, match='Materialized sample count'):
        ledger.add_materialized_samples(-1)


def test_credit_ledger_reconciliation_recovers_credits_lost_in_a_crash(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_materialized_samples(5)

    ledger.reconcile_materialized_samples(9)

    assert ledger.state.earned_credits == Decimal(18)


def test_credit_ledger_reconciliation_is_a_no_op_when_credits_match(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_materialized_samples(5)

    ledger.reconcile_materialized_samples(5)

    assert ledger.state.earned_credits == Decimal(10)


def test_credit_ledger_reconciliation_rejects_credits_beyond_materialized_data(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_materialized_samples(5)

    with pytest.raises(ValueError, match='earned more credits'):
        ledger.reconcile_materialized_samples(4)


def test_credit_ledger_initializes_progress_from_checkpoint_generation(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 2))

    assert ledger.model_generation == 2
    assert ledger.progress.completed_optimizer_steps == 8
    assert ledger.state.active_checkpoint.generation == 2
    assert ledger.state.available_credits == 0


def test_credit_ledger_adopts_one_complete_checkpoint_after_interrupted_commit(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_materialized_samples(16)
    completed = materialized_checkpoint(tmp_path, 1)

    restarted = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))

    assert restarted.model_generation == 1
    assert restarted.state.active_checkpoint == completed
    assert restarted.state.available_credits == 0
