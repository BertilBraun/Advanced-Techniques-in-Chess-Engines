from decimal import Decimal
from pathlib import Path

import pytest

from src.train.CreditTrainingLedger import CreditTrainingLedger, CreditTrainingProgress
from src.train.TrainingArgs import CreditTrainingParams


def _parameters(replay_ratio: str = '4') -> CreditTrainingParams:
    return CreditTrainingParams(
        replay_ratio=Decimal(replay_ratio),
        optimizer_steps_per_quantum=50,
        maximum_optimizer_steps=500_000,
        initial_replay_capacity_unique_positions=100_000,
        maximum_replay_capacity_unique_positions=2_500_000,
        replay_capacity_ramp_model_versions=1_000,
        retained_checkpoint_interval_steps=1_000,
    )


def _checkpoint(run_path: Path, version: int) -> Path:
    checkpoint_path = run_path / f'checkpoint_{version}.json'
    checkpoint_path.write_text(f'checkpoint {version}\n', encoding='utf-8')
    return checkpoint_path


def test_exactly_12_800_unique_samples_enable_one_quantum() -> None:
    parameters = _parameters()
    assert parameters.presentation_credits_per_quantum(1_024) == 51_200
    assert parameters.unique_samples_per_quantum(1_024) == 12_800

    insufficient = CreditTrainingProgress.initial().reconcile_credited_samples(12_799, parameters.replay_ratio)
    sufficient = insufficient.reconcile_credited_samples(12_800, parameters.replay_ratio)

    assert not insufficient.can_train(51_200)
    assert sufficient.can_train(51_200)


def test_fractional_and_surplus_credits_carry_forward() -> None:
    parameters = _parameters('1.5')
    progress = CreditTrainingProgress.initial().reconcile_credited_samples(40_000, parameters.replay_ratio)
    consumed = progress.consume_quantum(parameters, global_batch_size=1_024)

    assert consumed.earned_position_credits == Decimal('60000.0')
    assert consumed.consumed_position_credits == Decimal(51_200)
    assert consumed.available_position_credits == Decimal('8800.0')


def test_failed_publication_leaves_prepared_quantum_without_consuming_credits(tmp_path: Path) -> None:
    ledger = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    ledger.reconcile_credited_samples(12_800)
    prepared = ledger.prepare_quantum(_checkpoint(tmp_path, 1))

    restarted = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)

    assert restarted.progress.consumed_position_credits == 0
    assert restarted.progress.completed_optimizer_steps == 0
    assert restarted.prepared_quantum == prepared


def test_commit_is_restart_safe_and_does_not_consume_twice(tmp_path: Path) -> None:
    ledger = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    ledger.reconcile_credited_samples(20_000)
    ledger.prepare_quantum(_checkpoint(tmp_path, 1))
    committed = ledger.commit_prepared_quantum()

    restarted = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    restarted.reconcile_credited_samples(20_000)

    assert restarted.progress == committed
    assert restarted.progress.consumed_position_credits == 51_200
    assert restarted.progress.available_position_credits == 28_800
    with pytest.raises(RuntimeError, match='No training quantum'):
        restarted.commit_prepared_quantum()


def test_restart_recovers_commit_written_before_prepared_cleanup(tmp_path: Path) -> None:
    ledger = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    ledger.reconcile_credited_samples(12_800)
    prepared = ledger.prepare_quantum(_checkpoint(tmp_path, 1))
    ledger.commit_prepared_quantum()
    ledger.prepared_path.write_text(prepared.model_dump_json(indent=2), encoding='utf-8')

    restarted = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)

    assert restarted.progress.completed_training_quanta == 1
    assert not restarted.prepared_path.exists()


def test_replay_credits_cannot_change_while_publication_is_pending(tmp_path: Path) -> None:
    ledger = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    ledger.reconcile_credited_samples(12_800)
    ledger.prepare_quantum(_checkpoint(tmp_path, 1))

    with pytest.raises(RuntimeError, match='while a training quantum is prepared'):
        ledger.reconcile_credited_samples(13_000)


def test_replay_reconciliation_recovers_credit_after_ledger_write_loss(tmp_path: Path) -> None:
    ledger = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    ledger.reconcile_credited_samples(3_000)
    ledger.progress_path.unlink()

    restarted = CreditTrainingLedger(tmp_path, _parameters(), global_batch_size=1_024)
    recovered = restarted.reconcile_credited_samples(3_000)

    assert recovered.credited_unique_samples == 3_000
    assert recovered.earned_position_credits == 12_000
