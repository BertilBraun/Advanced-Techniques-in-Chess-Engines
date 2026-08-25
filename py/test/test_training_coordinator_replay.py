from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from src.replay.description import ReplayDescription
from src.replay.manager import IngestedCompletedGame, ReplayIngestion
from src.self_play.completed_game import TerminationReason
from src.training.coordinator import Coordinator
from src.training.credit_ledger import CreditLedgerState
from src.training.reporting import ReplayIngestionTelemetry
from src.training.session import TrainingSessionResult


@dataclass
class _FakeLedger:
    model_generation: int = 2
    reconciled_total: int = 0

    @property
    def state(self) -> CreditLedgerState:
        return cast(CreditLedgerState, None)

    def reconcile_materialized_samples(self, total_materialized_samples: int) -> None:
        self.reconciled_total = total_materialized_samples


@dataclass
class _FakeReplayManager:
    total_materialized: int
    ingestion: ReplayIngestion

    def total_materialized_samples(self) -> int:
        return self.total_materialized

    def append_staged_games(self, model_generation: int) -> ReplayIngestion:
        assert model_generation == 2
        return self.ingestion

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

    @property
    def rejection_rate(self) -> float:
        return 0.0


@dataclass
class _FakeReporter:
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
        self.completed_games = completed_games
        self.ingestion = ingestion


def _ingestion() -> ReplayIngestion:
    return ReplayIngestion(
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


def _coordinator(replay_manager: _FakeReplayManager) -> Coordinator:
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.replay_manager = replay_manager  # type: ignore[assignment]
    coordinator.ledger = _FakeLedger()  # type: ignore[assignment]
    coordinator.reporter = _FakeReporter()  # type: ignore[assignment]
    coordinator.resignation_calibrator = None
    coordinator._completed_games_since_last_quantum = []
    coordinator._ingest_seconds_since_last_quantum = 0.0
    return coordinator


def test_append_credits_the_absolute_appended_total_without_double_counting() -> None:
    replay_manager = _FakeReplayManager(total_materialized=12, ingestion=_ingestion())
    coordinator = _coordinator(replay_manager)

    coordinator._append_staged_games()
    coordinator._append_staged_games()

    ledger = cast(_FakeLedger, coordinator.ledger)
    assert ledger.reconciled_total == 12


def test_append_aggregates_in_memory_reporting_and_successful_report_clears_it() -> None:
    replay_manager = _FakeReplayManager(total_materialized=3, ingestion=_ingestion())
    coordinator = _coordinator(replay_manager)

    coordinator._append_staged_games()
    coordinator._append_staged_games()
    coordinator._report_training_outcome(cast(TrainingSessionResult, None), 1.5)

    reporter = cast(_FakeReporter, coordinator.reporter)
    assert reporter.completed_games == 2 * _ingestion().completed_games
    assert reporter.ingestion is not None
    assert reporter.ingestion.ingest_seconds == 0.5
    assert coordinator._completed_games_since_last_quantum == []
    assert coordinator._ingest_seconds_since_last_quantum == 0.0
