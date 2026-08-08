from decimal import Decimal
from pathlib import Path

import pytest

from src.training.checkpoint import CheckpointReference
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedger
from src.training.trainer_group import TrainingQuantumResult, TrainingStatistics


def _checkpoint(tmp_path: Path, generation: int) -> CheckpointReference:
    return CheckpointReference(
        generation=generation,
        manifest_path=tmp_path / f'checkpoint_{generation}.json',
        model_path=tmp_path / f'model_{generation}.pt',
        optimizer_path=tmp_path / f'optimizer_{generation}.pt',
        inference_model_path=tmp_path / f'model_{generation}.jit.pt',
        inference_model_sha256=f'{generation:064x}',
    )


def _parameters() -> CreditTrainingParams:
    return CreditTrainingParams(
        replay_ratio=Decimal(2),
        optimizer_steps_per_quantum=4,
        maximum_optimizer_steps=8,
        retained_checkpoint_interval_generations=1,
    )


def test_credit_ledger_persists_only_approximate_counters_and_active_checkpoint(tmp_path: Path) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_samples(16, model_generation=0)

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
            elapsed_seconds=1.0,
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


def test_credit_ledger_adopts_one_complete_checkpoint_after_interrupted_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))
    ledger.add_samples(16, model_generation=0)
    completed = _checkpoint(tmp_path, 1)
    completed.manifest_path.write_text('{}', encoding='utf-8')

    def load_checkpoint(
        cls: type[CheckpointReference],
        run_path: Path,
        generation: int,
    ) -> CheckpointReference:
        del cls
        assert run_path == tmp_path
        assert generation == 1
        return completed

    monkeypatch.setattr(CheckpointReference, 'load', classmethod(load_checkpoint))

    restarted = CreditLedger(tmp_path, _parameters(), global_batch_size=8, starting_checkpoint=_checkpoint(tmp_path, 0))

    assert restarted.model_generation == 1
    assert restarted.state.active_checkpoint == completed
    assert restarted.state.available_credits == 0
