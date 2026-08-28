from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from src.games.contracts import WdlTarget
from src.replay.contracts import ReplaySample
from src.replay.shard import ReplayShardGameMetadata, ReplayShardSourceGame
from src.search_budget.artifacts import LabelShardManifest, LabelShardPhase
from src.search_budget.calibration import CurveDecisionReason
from src.search_budget.configuration import DeepLabelingConfiguration, SearchBudgetConfiguration
from src.search_budget.curve import analytic_initial_curve, flat_curve
from src.search_budget.labeling import LabelGenerationSource, LabelPositionSource
from src.search_budget.manager import (
    InvalidLabelComputeError,
    LabelGenerationJob,
    LabelJobStatus,
    LabelManagerState,
    SearchBudgetLabelManager,
    SkippedLabelJobReport,
    _failure_decision_reason,
)
from src.search_budget.sampling import LabelPositionIdentity
from src.search_budget.worker import DeepSearchShardTask, LabelWorkerRuntime, PredictionShardTask
from src.self_play.completed_game import (
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
)
from src.training.checkpoint import CheckpointReference


@dataclass(frozen=True)
class _Writeback:
    row_count: int
    applied: bool


def _checkpoint(path: Path) -> CheckpointReference:
    return CheckpointReference(
        generation=1,
        manifest_path=path / 'checkpoint.json',
        model_path=path / 'model.pt',
        optimizer_path=path / 'optimizer.pt',
        inference_model_path=path / 'inference.pt',
        inference_model_sha256='0' * 64,
    )


def _position() -> LabelPositionSource:
    observation = SearchObservation(
        ply=0,
        model_generation=1,
        policy_target_visits=SearchVisitCounts(action_ids=(0,), visit_counts=(10,)),
        root_value=0.0,
        highest_visited_child_action_id=0,
        highest_visited_child_visit_count=10,
        highest_visited_child_q=0.0,
        selected_action_id=0,
        sample_weight=1.0,
        baseline_visits=10,
        network_root_value=0.0,
        policy_correction=0.0,
        value_correction=0.0,
        search_budget_logit=0.0,
        predicted_search_budget=0.5,
        assigned_additional_visits=10,
        parallel_searches=1,
        spend_residual=0,
        starting_visits=0,
        final_visits=10,
        stop_reason=SearchStopReason.FIXED_LIMIT,
    )
    game_identity = GameIdentity(
        worker_id=1,
        process_instance_id=UUID('c91531fb-1409-4e63-abab-660095b928cc'),
        game_number=1,
    )
    game = ReplayShardGameMetadata(
        source=ReplayShardSourceGame(identity=game_identity, counter=1),
        created_at_seconds=1.0,
        generation_seconds=1.0,
        action_ids=(0,),
        row_start=0,
        row_count=1,
        length_plies=1,
        termination_reason=TerminationReason.NATURAL,
        is_resignation_continuation=False,
        resignation_threshold=None,
        final_wdl=WdlTarget(win=1.0, draw=0.0, loss=0.0),
        observations=(observation,),
        policies_truncated=0,
        retained_visit_mass=10,
        discarded_visit_mass=0,
    )
    return LabelPositionSource(
        identity=LabelPositionIdentity(
            source_generation=1,
            game_identity=game_identity.archive_key,
            ply=0,
        ),
        game=game,
        observation_index=0,
    )


def _unused_runtime_factory(device_id: int) -> LabelWorkerRuntime:
    raise AssertionError(f'Injected executor must not initialize GPU {device_id}.')


class _UnusedSampleProvider:
    def __call__(self, source: LabelPositionSource) -> ReplaySample:
        raise AssertionError(f'No sample should be requested for {source.identity}.')

    def close(self) -> None:
        return None


def _unused_replay_writer(source_generation: int, samples: tuple[ReplaySample, ...]) -> _Writeback:
    raise AssertionError(f'No generation {source_generation} write of {len(samples)} rows is expected.')


def _manager(path: Path, executor: ThreadPoolExecutor) -> SearchBudgetLabelManager:
    return SearchBudgetLabelManager(
        run_path=path,
        configuration_sha256='1' * 64,
        device_ids=(0,),
        runtime_factory=_unused_runtime_factory,
        action_size=2,
        maximum_policy_entries=2,
        sample_provider=_UnusedSampleProvider(),
        replay_writer=_unused_replay_writer,
        initial_first_unstarted_production_generation=2,
        configuration=SearchBudgetConfiguration(),
        executor=executor,
    )


def _task(path: Path) -> PredictionShardTask:
    return PredictionShardTask(
        source_generation=1,
        shard_index=0,
        attempt=1,
        checkpoint=_checkpoint(path),
        positions=(_position(),),
        artifact_path=path / 'artifact-attempt-1.json',
        manifest_path=path / 'attempt-1.json',
    )


def test_deep_label_search_parallelism_matches_two_outstanding_batches(tmp_path: Path) -> None:
    configuration = DeepLabelingConfiguration()
    task = DeepSearchShardTask(
        source_generation=1,
        shard_index=0,
        attempt=1,
        checkpoint=_checkpoint(tmp_path),
        positions=(_position(),),
        checkpoint_visits=((10, 80),),
        deep_visit_limit=80,
        artifact_path=tmp_path / 'deep-artifact.json',
        manifest_path=tmp_path / 'deep-manifest.json',
    )

    assert configuration.parallel_searches == 2
    assert task.parallel_searches == 2
    with pytest.raises(ValueError, match='parallelism must remain two'):
        DeepLabelingConfiguration(parallel_searches=1)


def _successful_manifest(task: PredictionShardTask) -> LabelShardManifest:
    content = f'attempt {task.attempt}'.encode()
    task.artifact_path.write_bytes(content)
    return LabelShardManifest(
        phase=LabelShardPhase.PREDICTION,
        source_generation=task.source_generation,
        shard_index=task.shard_index,
        attempt=task.attempt,
        device_id=0,
        position_identities=tuple(position.identity for position in task.positions),
        position_count=len(task.positions),
        duration_seconds=0.0,
        artifact_path=task.artifact_path,
        artifact_sha256=hashlib.sha256(content).hexdigest(),
        artifact_size_bytes=len(content),
        checkpoint_sha256=task.checkpoint.inference_model_sha256,
    )


def test_shard_retry_succeeds_on_the_third_attempt(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = _manager(tmp_path, executor)
    attempts: list[int] = []

    def flaky_worker(task: PredictionShardTask) -> LabelShardManifest:
        attempts.append(task.attempt)
        if task.attempt < 3:
            raise RuntimeError('transient shard failure')
        return _successful_manifest(task)

    manifests = manager._execute_with_retry((_task(tmp_path),), flaky_worker)

    assert attempts == [1, 2, 3]
    assert manifests[0].attempt == 3
    assert manifests[0].artifact_path.name == 'artifact-attempt-3.json'
    manager.close()


def test_shard_retry_fails_terminally_after_three_attempts(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = _manager(tmp_path, executor)
    attempts: list[int] = []

    def failed_worker(task: PredictionShardTask) -> LabelShardManifest:
        attempts.append(task.attempt)
        raise RuntimeError('persistent shard failure')

    with pytest.raises(RuntimeError, match='persistent shard failure'):
        manager._execute_with_retry((_task(tmp_path),), failed_worker)

    assert attempts == [1, 2, 3]
    manager.close()


@pytest.mark.parametrize('games', ((), (_position().game,)))
def test_zero_or_under_fifty_position_cohort_is_durably_skipped(
    tmp_path: Path,
    games: tuple[ReplayShardGameMetadata, ...],
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = _manager(tmp_path, executor)

    result = manager.enqueue_replay_generation(1, games, _checkpoint(tmp_path), 10, 7)
    duplicate = manager.enqueue_replay_generation(1, games, _checkpoint(tmp_path), 10, 7)
    events = manager.poll()

    assert result.skipped and not result.accepted
    assert duplicate.skipped and not duplicate.accepted
    assert manager.accounted_source_generations == (1,)
    assert len(events) == 1
    assert isinstance(events[0], SkippedLabelJobReport)
    assert events[0].population_position_count == len(games)
    manager.close()


def test_unreadable_manager_state_constructs_fail_closed_and_preserves_evidence(tmp_path: Path) -> None:
    jobs_path = tmp_path / 'search-budget-labels'
    jobs_path.mkdir(parents=True)
    (jobs_path / 'manager-state.json').write_text('{not-json', encoding='utf-8')
    executor = ThreadPoolExecutor(max_workers=1)

    manager = _manager(tmp_path, executor)

    publication = manager.publication_for_generation(2)
    assert publication.curve == flat_curve()
    assert publication.decision_reason is CurveDecisionReason.UNREADABLE_STATE
    assert (jobs_path / 'manager-state-recovery.json').is_file()
    assert len(tuple(jobs_path.glob('manager-state-invalid-*.json'))) == 1
    manager.close()


def test_queued_job_is_rechecked_for_lag_when_it_reaches_the_front(tmp_path: Path) -> None:
    jobs_path = tmp_path / 'search-budget-labels'
    source_path = jobs_path / 'generation-00000001' / 'source.json'
    source_path.parent.mkdir(parents=True)
    source = LabelGenerationSource(
        source_generation=1,
        population_position_count=50,
        baseline_new_visits=10,
        checkpoint=_checkpoint(tmp_path),
        selected_positions=(_position(),),
    )
    source_path.write_text(source.model_dump_json(), encoding='utf-8')
    state = LabelManagerState(
        configuration_sha256='1' * 64,
        highest_started_production_generation=4,
        jobs=(
            LabelGenerationJob(
                source_generation=1,
                source_path=source_path,
                status=LabelJobStatus.QUEUED,
            ),
        ),
    )
    (jobs_path / 'manager-state.json').write_text(state.model_dump_json(), encoding='utf-8')
    executor = ThreadPoolExecutor(max_workers=1)
    manager = SearchBudgetLabelManager(
        run_path=tmp_path,
        configuration_sha256='1' * 64,
        device_ids=(0,),
        runtime_factory=_unused_runtime_factory,
        action_size=2,
        maximum_policy_entries=2,
        sample_provider=_UnusedSampleProvider(),
        replay_writer=_unused_replay_writer,
        initial_first_unstarted_production_generation=5,
        configuration=SearchBudgetConfiguration(),
        executor=executor,
    )
    deadline = time.monotonic() + 2.0
    events = ()
    while not events and time.monotonic() < deadline:
        events = manager.poll()
        time.sleep(0.01)

    assert len(events) == 1
    assert isinstance(events[0], SkippedLabelJobReport)
    assert 'at job start' in events[0].reason
    manager.close()


def test_starting_generation_boundary_is_monotonic_durable_and_prevents_late_publication(
    tmp_path: Path,
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = SearchBudgetLabelManager(
        run_path=tmp_path,
        configuration_sha256='1' * 64,
        device_ids=(0,),
        runtime_factory=_unused_runtime_factory,
        action_size=2,
        maximum_policy_entries=2,
        sample_provider=_UnusedSampleProvider(),
        replay_writer=_unused_replay_writer,
        initial_first_unstarted_production_generation=1,
        configuration=SearchBudgetConfiguration(),
        executor=executor,
    )

    started = manager.publication_for_starting_generation(1)
    manager._calibration = manager._calibration.model_copy(
        update={
            'previous_published_curve': flat_curve(),
            'published_curve': analytic_initial_curve(),
            'application_generation': manager._first_unstarted_production_generation,
        }
    )

    assert started.curve == flat_curve()
    assert manager.publication_for_generation(1).curve == flat_curve()
    assert manager.publication_for_generation(2).curve == analytic_initial_curve()
    assert manager.publication_for_starting_generation(1).curve == flat_curve()
    with pytest.raises(ValueError, match='without gaps'):
        manager.publication_for_starting_generation(3)
    manager.close()

    restarted_executor = ThreadPoolExecutor(max_workers=1)
    restarted = SearchBudgetLabelManager(
        run_path=tmp_path,
        configuration_sha256='1' * 64,
        device_ids=(0,),
        runtime_factory=_unused_runtime_factory,
        action_size=2,
        maximum_policy_entries=2,
        sample_provider=_UnusedSampleProvider(),
        replay_writer=_unused_replay_writer,
        initial_first_unstarted_production_generation=1,
        configuration=SearchBudgetConfiguration(),
        executor=restarted_executor,
    )
    assert restarted.publication_for_starting_generation(1).curve == flat_curve()
    restarted.close()


def test_invalid_reconstructed_compute_is_distinct_from_terminal_worker_failure() -> None:
    assert _failure_decision_reason(InvalidLabelComputeError('bad checksum')) is CurveDecisionReason.INVALID_COMPUTE
    assert _failure_decision_reason(RuntimeError('worker died')) is CurveDecisionReason.TERMINAL_FAILURE
