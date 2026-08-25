from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
import src.training.coordinator as coordinator_module
from src.replay.manager import ReplayIngestion
from src.self_play.protocol import (
    RunningSelfPlayState,
    RunningSelfPlayStateApplied,
    SelfPlayStateApplied,
    StoppedSelfPlayState,
)
from src.self_play.resignation import PublishedResignationPolicy
from src.training.checkpoint import CheckpointReference
from src.training.coordinator import Coordinator
from src.training.self_play_group import SelfPlaySupervision
from src.training.self_play_health import SelfPlayHealthMonitor
from test_helpers.checkpoints import checkpoint_reference

WORKER_COUNT = 4


@dataclass
class _LedgerState:
    active_checkpoint: CheckpointReference
    available_credits: int = 0


@dataclass
class _Ledger:
    state: _LedgerState
    maximum_loops: int
    loops: int = 0
    model_generation: int = 3
    has_quantum_credits: bool = False
    saved: bool = False

    @property
    def training_complete(self) -> bool:
        return self.loops >= self.maximum_loops

    def save(self) -> None:
        self.saved = True

    def reconcile_materialized_samples(self, total_materialized_samples: int) -> None:
        del total_materialized_samples


@dataclass
class _ReplayManager:
    ledger: _Ledger
    append_calls: int = 0
    closed: bool = False
    live_samples: int = 0

    def start_materialization(self) -> None:
        pass

    def total_materialized_samples(self) -> int:
        return 0

    def raise_if_materialization_failed(self) -> None:
        pass

    def append_staged_games(self, model_generation: int) -> ReplayIngestion:
        del model_generation
        self.append_calls += 1
        self.ledger.loops += 1
        return ReplayIngestion(
            games_ingested=0,
            samples_added=0,
            live_samples=0,
            evicted_samples=0,
            policies_truncated=0,
            retained_visit_mass=0,
            discarded_visit_mass=0,
            elapsed_seconds=0.0,
            completed_games=(),
        )

    def close(self) -> None:
        self.closed = True


@dataclass
class _EvaluationManager:
    def start(self) -> None:
        pass

    def collect_completed_jobs(self) -> None:
        pass

    def schedule_due_jobs(self, checkpoint: CheckpointReference) -> None:
        del checkpoint

    def close(self) -> None:
        pass


@dataclass
class _RunLimitMonitor:
    reason: str | None = None

    def stop_reason(self) -> str | None:
        return self.reason


@dataclass
class _SelfPlayGroup:
    checkpoint: CheckpointReference
    live_worker_counts: list[int]
    supervisions: list[SelfPlaySupervision]
    supervise_calls: int = 0
    closed: bool = False
    worker_count: int = WORKER_COUNT

    @property
    def live_worker_count(self) -> int:
        index = min(self.supervise_calls, len(self.live_worker_counts)) - 1
        return self.worker_count if index < 0 else self.live_worker_counts[index]

    def supervise(
        self,
        checkpoint: CheckpointReference,
        resignation_policy: PublishedResignationPolicy,
    ) -> SelfPlaySupervision:
        del checkpoint, resignation_policy
        index = min(self.supervise_calls, len(self.supervisions) - 1)
        self.supervise_calls += 1
        return self.supervisions[index]

    def apply(
        self,
        desired_states: tuple[RunningSelfPlayState | StoppedSelfPlayState, ...],
    ) -> tuple[SelfPlayStateApplied, ...]:
        return tuple(
            cast(
                SelfPlayStateApplied,
                RunningSelfPlayStateApplied(
                    worker_id=worker_id,
                    loaded_generation=self.checkpoint.generation,
                    loaded_inference_model_sha256=self.checkpoint.inference_model_sha256,
                    completed_generation_statistics=None,
                ),
            )
            for worker_id in range(len(desired_states))
        )

    def request_pause(self, worker_ids: tuple[int, ...]) -> None:
        del worker_ids

    def close(self) -> None:
        self.closed = True


@dataclass
class _TrainingSession:
    has_pending_quantum: bool = False

    def close(self) -> None:
        pass


class _Credit:
    def requires_self_play_backpressure(self, available_credits: int, global_batch_size: int) -> bool:
        del available_credits, global_batch_size
        return False


@dataclass(frozen=True)
class _Lifecycle:
    credit: _Credit = field(default_factory=_Credit)


@dataclass(frozen=True)
class _Trainer:
    global_batch_size: int = 128


@dataclass(frozen=True)
class _Training:
    lifecycle: _Lifecycle = field(default_factory=_Lifecycle)
    trainer: _Trainer = field(default_factory=_Trainer)


@dataclass(frozen=True)
class _Configuration:
    training: _Training = field(default_factory=_Training)


class _Game:
    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _Harness:
    coordinator: Coordinator
    replay_manager: _ReplayManager
    self_play_group: _SelfPlayGroup


def _harness(
    tmp_path: Path,
    live_worker_counts: list[int],
    supervisions: list[SelfPlaySupervision],
    maximum_loops: int = 3,
    grace_seconds: float = 0.0,
    run_limit_reason: str | None = None,
) -> _Harness:
    checkpoint = checkpoint_reference(tmp_path, 3, write_inference_model=True)
    ledger = _Ledger(state=_LedgerState(active_checkpoint=checkpoint), maximum_loops=maximum_loops)
    replay_manager = _ReplayManager(ledger)
    self_play_group = _SelfPlayGroup(checkpoint, live_worker_counts, supervisions)
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.game = _Game()  # type: ignore[assignment]
    coordinator.configuration = _Configuration()  # type: ignore[assignment]
    coordinator.ledger = ledger  # type: ignore[assignment]
    coordinator.replay_manager = replay_manager  # type: ignore[assignment]
    coordinator.self_play_group = self_play_group  # type: ignore[assignment]
    coordinator.self_play_health = SelfPlayHealthMonitor(WORKER_COUNT, grace_seconds=grace_seconds)
    coordinator.evaluation_manager = _EvaluationManager()  # type: ignore[assignment]
    coordinator.training_session = _TrainingSession()  # type: ignore[assignment]
    coordinator.run_limit_monitor = _RunLimitMonitor(run_limit_reason)  # type: ignore[assignment]
    coordinator.resignation_calibrator = None
    coordinator.final_stop_reason = None
    coordinator._backpressure_pause_requested = False
    coordinator._credit_wait_started_at = 0.0
    coordinator._completed_games_since_last_quantum = []
    coordinator._ingest_seconds_since_last_quantum = 0.0
    return _Harness(coordinator, replay_manager, self_play_group)


@pytest.fixture(autouse=True)
def _instant_idle_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coordinator_module, 'IDLE_WAIT_SECONDS', 0.0)


def test_ingestion_runs_every_loop_while_a_worker_restart_is_still_pending(tmp_path: Path) -> None:
    harness = _harness(tmp_path, [WORKER_COUNT - 1], [SelfPlaySupervision((), ())], grace_seconds=1e6)

    harness.coordinator.run()

    assert harness.replay_manager.append_calls == 3


def test_a_pending_restart_does_not_stop_the_run_inside_the_grace_period(tmp_path: Path) -> None:
    harness = _harness(tmp_path, [1], [SelfPlaySupervision((), ())], grace_seconds=1e6)

    harness.coordinator.run()

    assert harness.coordinator.final_stop_reason is None


def test_repeated_restart_failures_do_not_raise_out_of_the_run_loop(tmp_path: Path) -> None:
    harness = _harness(tmp_path, [WORKER_COUNT - 1], [SelfPlaySupervision((), (3,))], grace_seconds=1e6)

    harness.coordinator.run()

    assert harness.self_play_group.supervise_calls == 3


def test_a_worker_that_recovers_leaves_the_run_without_a_stop_reason(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        [0, WORKER_COUNT],
        [SelfPlaySupervision((), (3,)), SelfPlaySupervision((3,), ())],
        grace_seconds=0.0,
    )

    harness.coordinator.run()

    assert harness.coordinator.final_stop_reason is None


def test_a_persistently_dead_self_play_group_stops_the_run_with_a_reason(tmp_path: Path) -> None:
    harness = _harness(tmp_path, [0], [SelfPlaySupervision((), (0, 1, 2, 3))], maximum_loops=100)

    harness.coordinator.run()

    assert harness.coordinator.final_stop_reason is not None
    assert harness.coordinator.final_stop_reason.startswith('self-play capacity degraded')


def test_a_persistently_dead_self_play_group_still_ingests_before_stopping(tmp_path: Path) -> None:
    harness = _harness(tmp_path, [0], [SelfPlaySupervision((), (0, 1, 2, 3))], maximum_loops=100)

    harness.coordinator.run()

    assert harness.replay_manager.append_calls == 2


def test_a_run_limit_stop_reason_takes_precedence_over_self_play_health(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        [0],
        [SelfPlaySupervision((), ())],
        maximum_loops=100,
        run_limit_reason='manual stop requested',
    )

    harness.coordinator.run()

    assert harness.coordinator.final_stop_reason == 'manual stop requested'


def test_the_run_closes_the_self_play_group_after_a_degraded_stop(tmp_path: Path) -> None:
    harness = _harness(tmp_path, [0], [SelfPlaySupervision((), ())], maximum_loops=100)

    harness.coordinator.run()

    assert harness.self_play_group.closed
