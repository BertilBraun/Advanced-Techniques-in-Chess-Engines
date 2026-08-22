from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest
from src.replay.description import ReplayDescription
from src.replay.manager import IngestedCompletedGame, ReplayIngestionReceipt
from src.replay.parallel_materialization import SealedReplayShard
from src.self_play.completed_game import TerminationReason
from src.training.coordinator import Coordinator, _PendingReplayReporting
from src.training.credit_ledger import CreditLedgerState
from src.training.reporting import ReplayIngestionTelemetry
from src.training.session import TrainingSessionResult


def _receipt(identity_digit: str = '1') -> ReplayIngestionReceipt:
    return ReplayIngestionReceipt(
        receipt_identity=identity_digit * 64,
        model_generation=2,
        shard_identities=('2' * 64,),
        append_sequence_after=1,
        games_ingested=1,
        samples_added=3,
        live_samples=3,
        evicted_samples=0,
        policies_truncated=1,
        retained_visit_mass=7,
        discarded_visit_mass=2,
        elapsed_seconds=0.25,
        completed_games=(IngestedCompletedGame(4, TerminationReason.NATURAL),),
    )


@dataclass
class _FakeLedger:
    reconciled_total: int = 0

    @property
    def state(self) -> CreditLedgerState:
        return cast(CreditLedgerState, None)

    def reconcile_materialized_samples(self, total_materialized_samples: int) -> None:
        self.reconciled_total = total_materialized_samples


@dataclass
class _FakeReplayManager:
    total_materialized: int = 0
    receipts: tuple[ReplayIngestionReceipt, ...] = ()
    acknowledged: list[tuple[str, ...]] = field(default_factory=list)

    def total_materialized_samples(self) -> int:
        return self.total_materialized

    def pending_ingestion_receipts(self) -> tuple[ReplayIngestionReceipt, ...]:
        return self.receipts

    def acknowledge_ingestion_receipts(self, receipt_identities: tuple[str, ...]) -> None:
        self.acknowledged.append(receipt_identities)

    def description(self) -> ReplayDescription:
        return cast(ReplayDescription, None)

    @property
    def inbox_depth(self) -> int:
        return 0

    @property
    def staging_depth(self) -> int:
        return 0

    @property
    def materialization_failures(self) -> int:
        return 0


@dataclass
class _FakeReporter:
    replay_manager: _FakeReplayManager
    fail: bool = False
    completed_games: tuple[IngestedCompletedGame, ...] = ()
    ingestion: ReplayIngestionTelemetry | None = None

    def record_training_outcome(
        self,
        outcome: TrainingSessionResult,
        credit_wait_seconds: float,
        ledger_state: CreditLedgerState,
        replay: ReplayDescription,
        completed_games: tuple[IngestedCompletedGame, ...],
        ingestion: ReplayIngestionTelemetry,
    ) -> None:
        del outcome, credit_wait_seconds, ledger_state, replay
        assert self.replay_manager.acknowledged == []
        self.completed_games = completed_games
        self.ingestion = ingestion
        if self.fail:
            raise RuntimeError('report failed')


def _coordinator_for_replay_helpers(
    replay_manager: _FakeReplayManager,
    reporter: _FakeReporter | None = None,
) -> Coordinator:
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.replay_manager = replay_manager  # type: ignore[assignment]
    coordinator.ledger = _FakeLedger()  # type: ignore[assignment]
    coordinator.reporter = reporter or _FakeReporter(replay_manager)  # type: ignore[assignment]
    coordinator._pending_replay_reporting = _PendingReplayReporting()
    return coordinator


def test_restarted_sealed_callback_reconciles_absolute_credit_without_double_counting() -> None:
    replay_manager = _FakeReplayManager(total_materialized=12)
    coordinator = _coordinator_for_replay_helpers(replay_manager)
    sealed = SealedReplayShard(sequence=0, shard_identity='3' * 64, row_count=12, game_count=1)

    coordinator._reconcile_materialized_shard(sealed)
    coordinator._reconcile_materialized_shard(sealed)

    ledger = cast(_FakeLedger, coordinator.ledger)
    assert ledger.reconciled_total == 12


def test_pending_receipt_collection_replays_and_deduplicates_restart_receipts() -> None:
    receipt = _receipt()
    replay_manager = _FakeReplayManager(receipts=(receipt, receipt))
    coordinator = _coordinator_for_replay_helpers(replay_manager)

    coordinator._collect_pending_ingestion_receipts()
    coordinator._collect_pending_ingestion_receipts()

    pending = coordinator._pending_replay_reporting
    assert pending.receipt_identities == [receipt.receipt_identity]
    assert pending.completed_games == list(receipt.completed_games)
    assert pending.ingest_seconds == receipt.elapsed_seconds


def test_successful_report_acknowledges_receipts_then_clears_accumulator() -> None:
    receipt = _receipt()
    replay_manager = _FakeReplayManager(receipts=(receipt,))
    reporter = _FakeReporter(replay_manager)
    coordinator = _coordinator_for_replay_helpers(replay_manager, reporter)
    coordinator._collect_pending_ingestion_receipts()

    coordinator._report_training_outcome(cast(TrainingSessionResult, None), 1.5)

    assert replay_manager.acknowledged == [(receipt.receipt_identity,)]
    assert reporter.completed_games == receipt.completed_games
    assert reporter.ingestion is not None
    assert reporter.ingestion.ingest_seconds == receipt.elapsed_seconds
    assert coordinator._pending_replay_reporting.receipt_identities == []


def test_report_failure_retains_receipt_and_does_not_acknowledge() -> None:
    receipt = _receipt()
    replay_manager = _FakeReplayManager(receipts=(receipt,))
    reporter = _FakeReporter(replay_manager, fail=True)
    coordinator = _coordinator_for_replay_helpers(replay_manager, reporter)
    coordinator._collect_pending_ingestion_receipts()

    with pytest.raises(RuntimeError, match='report failed'):
        coordinator._report_training_outcome(cast(TrainingSessionResult, None), 1.5)

    assert replay_manager.acknowledged == []
    assert coordinator._pending_replay_reporting.receipt_identities == [receipt.receipt_identity]


def test_acknowledgment_failure_retains_pending_accumulator() -> None:
    class _FailingAcknowledgeReplayManager(_FakeReplayManager):
        def acknowledge_ingestion_receipts(self, receipt_identities: tuple[str, ...]) -> None:
            del receipt_identities
            raise OSError('receipt directory unavailable')

    receipt = _receipt()
    replay_manager = _FailingAcknowledgeReplayManager(receipts=(receipt,))
    reporter = _FakeReporter(replay_manager)
    coordinator = _coordinator_for_replay_helpers(replay_manager, reporter)
    coordinator._collect_pending_ingestion_receipts()

    with pytest.raises(OSError, match='receipt directory unavailable'):
        coordinator._report_training_outcome(cast(TrainingSessionResult, None), 1.5)

    assert coordinator._pending_replay_reporting.receipt_identities == [receipt.receipt_identity]
