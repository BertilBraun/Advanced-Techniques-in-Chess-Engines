from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pytest
from src.games.contracts import GameStateContract, Player, TerminalOracle, WdlTarget
from src.games.representation import PackedPlaneLayout, PackedPlanePayload, RepresentationDimensions
from src.replay.configuration import ReplayConfiguration
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    IneligibleNextPolicyTarget,
    IneligibleRemainingGameLengthTarget,
    IneligibleScalarAuxiliaryTarget,
    IneligibleSearchBudgetTarget,
    ReplaySample,
)
from src.replay.dispatch import parse_worker_source_file_name, worker_source_file_names
from src.replay.encoding import encode_replay_rows
from src.replay.layout import ReplayLayout
from src.replay.manager import (
    LabelledReplayWritebackReceipt,
    LabelledReplayWritebackState,
    ReplayManager,
)
from src.replay.materialization import materialize_completed_game
from src.replay.shard import MANIFEST_SUFFIX, SealedReplayShardManifest, replay_shard_manifest_path
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
    publish_completed_self_play_game,
)
from src.self_play.resignation import CalibratedResignationConfiguration, ResignationCalibrator
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchBudgetHeadLayout,
    TrainingTargetLayout,
)
from src.util.atomic_file import write_text_atomically
from src.util.generation_schedule import ConstantSchedule


@dataclass(frozen=True)
class LinearPosition:
    action_ids: tuple[int, ...] = ()


class LinearStateContract(GameStateContract[LinearPosition]):
    def __init__(self) -> None:
        packed_planes = PackedPlaneLayout(board_size=1, binary_plane_count=1, scalar_count=0)
        self._representation = RepresentationDimensions(
            channels=1,
            rows=1,
            columns=1,
            binary_channels=(0,),
            scalar_channels=(),
            packed_planes=packed_planes,
        )

    @property
    def name(self) -> str:
        return 'linear-test-game'

    @property
    def action_size(self) -> int:
        return 3

    @property
    def representation(self) -> RepresentationDimensions:
        return self._representation

    def initial_position(self) -> LinearPosition:
        return LinearPosition()

    def legal_action_ids(self, position: LinearPosition) -> tuple[int, ...]:
        return () if len(position.action_ids) == 4 else (0, 1, 2)

    def child_position(self, position: LinearPosition, action_id: int) -> LinearPosition:
        if action_id not in self.legal_action_ids(position):
            raise ValueError('Action is not legal in the linear test game.')
        return LinearPosition(position.action_ids + (action_id,))

    def is_irreversible_transition(self, position: LinearPosition, action_id: int, child: LinearPosition) -> bool:
        del position, child
        return action_id == 2

    def current_player(self, position: LinearPosition) -> Player:
        return Player.FIRST if len(position.action_ids) % 2 == 0 else Player.SECOND

    def natural_terminal_wdl(self, position: LinearPosition) -> WdlTarget | None:
        return WdlTarget(win=0.0, draw=0.0, loss=1.0) if len(position.action_ids) == 4 else None

    def adjudicated_wdl(self, position: LinearPosition, reason: TerminationReason) -> WdlTarget:
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def encode_network_input(self, position: LinearPosition) -> PackedPlanePayload:
        return self.packed_plane_layout.value(len(position.action_ids).to_bytes(8, byteorder='little'))

    @property
    def augmentation_count(self) -> int:
        return 1

    def transform_decoded_states(
        self,
        states: npt.NDArray[np.float32],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> None:
        if len(states) != len(augmentation_indices) or np.any(augmentation_indices != 0):
            raise ValueError('Linear test augmentation indices are not batch-aligned identities.')

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        return action_id


LINEAR_STATE_CONTRACT = LinearStateContract()
UNDISCOUNTED_VALUES = ConstantSchedule[float](value=1.0)


def _completed_game() -> CompletedSelfPlayGame:
    actions = (0, 1, 0, 2)
    action_ids: list[int] = []
    observations: list[SearchObservation] = []
    for ply, selected_action in enumerate(actions):
        other_action = (selected_action + 1) % LINEAR_STATE_CONTRACT.action_size
        action_ids.append(selected_action)
        observations.append(
            SearchObservation(
                ply=ply,
                model_generation=2,
                policy_target_visits=SearchVisitCounts(
                    action_ids=(other_action, selected_action), visit_counts=(3, 10)
                ),
                root_value=0.25,
                highest_visited_child_action_id=selected_action,
                highest_visited_child_visit_count=10,
                highest_visited_child_q=0.2,
                selected_action_id=selected_action,
                sample_weight=1.0,
                baseline_visits=13,
                network_root_value=0.1,
                policy_correction=0.2,
                value_correction=0.075,
                search_budget_logit=-0.4,
                predicted_search_budget=0.4,
                assigned_additional_visits=3 if ply == 1 else 13,
                parallel_searches=1,
                spend_residual=0,
                starting_visits=0,
                final_visits=3 if ply == 1 else 13,
                stop_reason=SearchStopReason.PREDICTED_BUDGET,
            )
        )
    return CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=3,
            process_instance_id=UUID('38c8809f-a49d-4d98-8da5-034614893665'),
            game_number=7,
        ),
        created_at_seconds=100.0,
        generation_seconds=1.5,
        action_ids=tuple(action_ids),
        observations=tuple(observations),
        final_wdl=WdlTarget(win=0.0, draw=0.0, loss=1.0),
        termination_reason=TerminationReason.NATURAL,
    )


def _target_layout() -> TrainingTargetLayout:
    return TrainingTargetLayout(
        action_size=LINEAR_STATE_CONTRACT.action_size,
        wdl_size=3,
        auxiliary_heads=(
            NextPolicyHeadLayout(kind='next_policy', action_size=LINEAR_STATE_CONTRACT.action_size, ply_offset=1),
        ),
    )


def test_shared_materialization_reconstructs_perspective_and_trajectory_targets() -> None:
    materialized = materialize_completed_game(
        _completed_game(),
        LINEAR_STATE_CONTRACT,
        None,
        _target_layout(),
        1,
        UNDISCOUNTED_VALUES,
    )

    assert len(materialized.samples) == 4
    assert materialized.policies_truncated == 7
    assert materialized.retained_visit_mass == 70
    assert materialized.discarded_visit_mass == 21
    assert materialized.samples[0].wdl_target == WdlTarget(win=0.0, draw=0.0, loss=1.0)
    assert isinstance(materialized.samples[0].auxiliary_targets[0], EligibleNextPolicyTarget)
    assert isinstance(materialized.samples[-1].auxiliary_targets[0], IneligibleNextPolicyTarget)
    assert materialized.samples[0].policy.visits.visit_counts[0] == 10


def test_materialization_reuses_trajectory_legal_actions_for_all_targets() -> None:
    class CountingLinearStateContract(LinearStateContract):
        def __init__(self) -> None:
            super().__init__()
            self.legal_action_calls = 0

        def legal_action_ids(self, position: LinearPosition) -> tuple[int, ...]:
            self.legal_action_calls += 1
            return super().legal_action_ids(position)

    state = CountingLinearStateContract()

    materialize_completed_game(
        _completed_game(),
        state,
        None,
        _target_layout(),
        1,
        UNDISCOUNTED_VALUES,
    )

    assert state.legal_action_calls == 2 * len(_completed_game().action_ids)


def test_materialization_reconstructs_unobserved_restart_prefix() -> None:
    game = CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=1,
            process_instance_id=UUID('f4cb1244-c91d-4897-87e0-3e9b05e54974'),
            game_number=1,
        ),
        created_at_seconds=1.0,
        generation_seconds=1.0,
        action_ids=(0, 1, 2, 0),
        observations=(
            SearchObservation(
                ply=2,
                model_generation=1,
                policy_target_visits=SearchVisitCounts(action_ids=(2,), visit_counts=(8,)),
                root_value=0.0,
                highest_visited_child_action_id=2,
                highest_visited_child_visit_count=8,
                highest_visited_child_q=0.0,
                selected_action_id=2,
                sample_weight=1.0,
                baseline_visits=8,
                network_root_value=0.0,
                policy_correction=0.0,
                value_correction=0.0,
                search_budget_logit=0.0,
                predicted_search_budget=0.5,
                assigned_additional_visits=8,
                parallel_searches=1,
                spend_residual=0,
                starting_visits=0,
                final_visits=8,
                stop_reason=SearchStopReason.FIXED_LIMIT,
            ),
        ),
        final_wdl=WdlTarget(win=0.0, draw=0.0, loss=1.0),
        termination_reason=TerminationReason.NATURAL,
    )
    targets = TrainingTargetLayout(action_size=3, wdl_size=3, auxiliary_heads=())

    materialized = materialize_completed_game(
        game,
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        maximum_policy_entries=3,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
    )

    assert len(materialized.samples) == 1
    assert materialized.samples[0].encoded_state == LINEAR_STATE_CONTRACT.encode_network_input(LinearPosition((0, 1)))


def test_remaining_game_length_uses_exact_completed_trajectory_boundary() -> None:
    targets = TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(
            RemainingGameLengthHeadLayout(
                kind='remaining_game_length',
                normalization_scale=4.0,
            ),
        ),
    )

    materialized = materialize_completed_game(
        _completed_game(),
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        3,
        UNDISCOUNTED_VALUES,
    )

    assert tuple(
        target.normalized_length
        for sample in materialized.samples
        for target in sample.auxiliary_targets
        if isinstance(target, EligibleRemainingGameLengthTarget)
    ) == pytest.approx((1.0, 0.75, 0.5, 0.25))


def test_four_ply_future_value_uses_terminal_fallback_with_current_perspective() -> None:
    targets = TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(FutureSearchValueHeadLayout(kind='future_search_value', ply_offset=4, smooth_l1_beta=0.1),),
    )

    game = _completed_game()
    materialized = materialize_completed_game(
        game,
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        3,
        UNDISCOUNTED_VALUES,
    )

    values = tuple(sample.auxiliary_targets[0] for sample in materialized.samples)
    assert values == (
        EligibleScalarAuxiliaryTarget(kind='future_search_value', value=-1.0),
        EligibleScalarAuxiliaryTarget(kind='future_search_value', value=1.0),
        EligibleScalarAuxiliaryTarget(kind='future_search_value', value=-1.0),
        EligibleScalarAuxiliaryTarget(kind='future_search_value', value=1.0),
    )


def test_irreversible_progress_records_event_distance_and_terminal_censoring() -> None:
    targets = TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(IrreversibleProgressHeadLayout(kind='irreversible_progress', horizon_plies=3),),
    )
    materialized = materialize_completed_game(
        _completed_game(),
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        3,
        UNDISCOUNTED_VALUES,
    )
    event_targets = tuple(sample.auxiliary_targets[0] for sample in materialized.samples)
    assert event_targets[2:] == (
        EligibleScalarAuxiliaryTarget(kind='irreversible_progress', value=2 / 3),
        EligibleScalarAuxiliaryTarget(kind='irreversible_progress', value=1 / 3),
    )

    game = _completed_game().model_copy(
        update={
            'action_ids': (0, 1, 0, 0),
            'observations': (
                *_completed_game().observations[:-1],
                _completed_game()
                .observations[-1]
                .model_copy(
                    update={
                        'selected_action_id': 0,
                        'highest_visited_child_action_id': 0,
                    }
                ),
            ),
        }
    )
    censored = materialize_completed_game(
        game,
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        3,
        UNDISCOUNTED_VALUES,
    )
    assert isinstance(censored.samples[2].auxiliary_targets[0], IneligibleScalarAuxiliaryTarget)


def test_ordinary_observations_materialize_ineligible_search_budget_targets() -> None:
    targets = TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(SearchBudgetHeadLayout(kind='search_budget'),),
    )
    game = _completed_game()

    materialized = materialize_completed_game(
        game,
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        3,
        UNDISCOUNTED_VALUES,
    )

    assert materialized.samples[0].auxiliary_targets[0] == IneligibleSearchBudgetTarget()


def test_materialization_uniformly_blurs_wdl_by_actual_remaining_game_plies() -> None:
    materialized = materialize_completed_game(
        _completed_game(),
        LINEAR_STATE_CONTRACT,
        None,
        _target_layout(),
        3,
        ConstantSchedule[float](value=0.5),
    )

    assert materialized.samples[0].wdl_target == WdlTarget(win=0.3125, draw=0.3125, loss=0.375)
    assert materialized.samples[1].wdl_target == WdlTarget(
        win=5.0 / 12.0,
        draw=7.0 / 24.0,
        loss=7.0 / 24.0,
    )
    assert materialized.samples[2].wdl_target == WdlTarget(win=0.25, draw=0.25, loss=0.5)
    assert materialized.samples[3].wdl_target == WdlTarget(
        win=2.0 / 3.0,
        draw=1.0 / 6.0,
        loss=1.0 / 6.0,
    )


class WinningTerminalOracle(TerminalOracle[LinearPosition]):
    def probe_wdl(self, position: LinearPosition) -> WdlTarget | None:
        assert len(position.action_ids) == 3
        return WdlTarget(win=1.0, draw=0.0, loss=0.0)


def test_materialization_revalidates_maximum_ply_result_with_terminal_oracle() -> None:
    natural_game = _completed_game()
    game = CompletedSelfPlayGame.model_validate(
        {
            **natural_game.model_dump(),
            'action_ids': natural_game.action_ids[:3],
            'observations': natural_game.observations[:3],
            'final_wdl': WdlTarget(win=1.0, draw=0.0, loss=0.0),
            'termination_reason': TerminationReason.MAXIMUM_PLIES,
        }
    )

    materialized = materialize_completed_game(
        game,
        LINEAR_STATE_CONTRACT,
        WinningTerminalOracle(),
        _target_layout(),
        3,
        UNDISCOUNTED_VALUES,
    )

    assert materialized.samples[-1].wdl_target == WdlTarget(win=0.0, draw=0.0, loss=1.0)


def _remaining_game_length_layout() -> TrainingTargetLayout:
    return TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=4.0),),
    )


def _maximum_plies_game() -> CompletedSelfPlayGame:
    natural_game = _completed_game()
    return CompletedSelfPlayGame.model_validate(
        {
            **natural_game.model_dump(),
            'action_ids': natural_game.action_ids[:3],
            'observations': natural_game.observations[:3],
            'final_wdl': WdlTarget(win=1.0, draw=0.0, loss=0.0),
            'termination_reason': TerminationReason.MAXIMUM_PLIES,
        }
    )


def test_censoring_marks_remaining_game_length_ineligible_on_cut_games() -> None:
    materialized = materialize_completed_game(
        _maximum_plies_game(),
        LINEAR_STATE_CONTRACT,
        WinningTerminalOracle(),
        _remaining_game_length_layout(),
        3,
        UNDISCOUNTED_VALUES,
        censor_remaining_game_length_on_cut_games=True,
    )

    assert materialized.samples
    assert all(
        isinstance(target, IneligibleRemainingGameLengthTarget)
        for sample in materialized.samples
        for target in sample.auxiliary_targets
    )


def test_cut_games_keep_length_targets_when_censoring_is_disabled() -> None:
    materialized = materialize_completed_game(
        _maximum_plies_game(),
        LINEAR_STATE_CONTRACT,
        WinningTerminalOracle(),
        _remaining_game_length_layout(),
        3,
        UNDISCOUNTED_VALUES,
    )

    assert materialized.samples
    assert all(
        isinstance(target, EligibleRemainingGameLengthTarget)
        for sample in materialized.samples
        for target in sample.auxiliary_targets
    )


def test_censoring_leaves_naturally_finished_games_eligible() -> None:
    materialized = materialize_completed_game(
        _completed_game(),
        LINEAR_STATE_CONTRACT,
        None,
        _remaining_game_length_layout(),
        3,
        UNDISCOUNTED_VALUES,
        censor_remaining_game_length_on_cut_games=True,
    )

    assert materialized.samples
    assert all(
        isinstance(target, EligibleRemainingGameLengthTarget)
        for sample in materialized.samples
        for target in sample.auxiliary_targets
    )


def _replay_layout() -> ReplayLayout:
    return ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )


def _open_manager(
    tmp_path: Path,
    capacity: int,
    maximum_capacity: int,
    resignation_calibrator: ResignationCalibrator | None = None,
    shard_maximum_games: int = 32,
    shard_maximum_source_bytes: int = 16 * 1024 * 1024,
    materialization_processes: int = 1,
    staging_shard_limit: int = 96,
    inbox_rename_cap: int = 4096,
    rejection_window_games: int = 512,
    rejection_rate_ceiling: float = 0.05,
) -> ReplayManager[LinearPosition]:
    configuration = ReplayConfiguration(
        capacity=capacity,
        maximum_capacity=maximum_capacity,
        maximum_policy_entries=1,
        materialization_processes=materialization_processes,
        materialization_shard_maximum_games=shard_maximum_games,
        materialization_shard_target_source_bytes=shard_maximum_source_bytes,
        materialization_staging_shard_limit=staging_shard_limit,
        materialization_inbox_rename_cap=inbox_rename_cap,
        materialization_rejection_window_games=rejection_window_games,
        materialization_rejection_rate_ceiling=rejection_rate_ceiling,
    )
    return ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        _replay_layout(),
        configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
        resignation_calibrator=resignation_calibrator,
    )


def _publish_games(inbox: Path, count: int, first_game_number: int = 7) -> None:
    game = _completed_game()
    for game_number in range(first_game_number, first_game_number + count):
        publish_completed_self_play_game(
            inbox,
            game.validated_copy(
                update={
                    'identity': {
                        'worker_id': 3,
                        'process_instance_id': '38c8809f-a49d-4d98-8da5-034614893665',
                        'game_number': game_number,
                    }
                }
            ),
        )


def _publish_malformed_games(inbox: Path, count: int, first_game_number: int = 500) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    for game_number in range(first_game_number, first_game_number + count):
        identity = GameIdentity(
            worker_id=3,
            process_instance_id=UUID('38c8809f-a49d-4d98-8da5-034614893665'),
            game_number=game_number,
        )
        (inbox / identity.file_name).write_text('{}', encoding='utf-8')


def _worker_source_paths(manager: ReplayManager[LinearPosition]) -> tuple[Path, ...]:
    return tuple(
        worker_path / name
        for worker_path in manager.worker_paths
        for name in sorted(worker_source_file_names(worker_path))
    )


SAMPLES_PER_GAME = 4


def test_game_identity_file_name_parser_requires_canonical_name() -> None:
    identity = _completed_game().identity

    assert GameIdentity.from_file_name(identity.file_name) == identity
    with pytest.raises(ValueError, match='invalid'):
        GameIdentity.from_file_name('../game.json')
    with pytest.raises(ValueError, match='canonical'):
        GameIdentity.from_file_name(identity.file_name.replace('worker-3', 'worker-03'))


@pytest.mark.parametrize('legacy_suffix', ('.rows.npy', '.meta.json'))
def test_replay_manager_rejects_legacy_per_game_staging_artifacts(
    tmp_path: Path,
    legacy_suffix: str,
) -> None:
    staging_path = tmp_path / 'completed-games' / 'staging'
    staging_path.mkdir(parents=True)
    (staging_path / f'legacy-game{legacy_suffix}').write_bytes(b'legacy')

    with pytest.raises(ValueError, match='explicit replay migration'):
        _open_manager(tmp_path, capacity=4, maximum_capacity=4)


def test_replay_manager_rejects_a_legacy_shard_queue(tmp_path: Path) -> None:
    completed_games = tmp_path / 'completed-games'
    completed_games.mkdir(parents=True)
    (completed_games / 'shard-queue.json').write_text('{"next_sequence": 2738, "pending": []}', encoding='utf-8')

    with pytest.raises(ValueError, match='explicit replay migration'):
        _open_manager(tmp_path, capacity=4, maximum_capacity=4)


def test_replay_manager_materializes_appends_and_reopens(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=4, maximum_capacity=6)

    manager.materialize_available_games()

    assert manager.inbox_depth == 0
    assert manager.staging_depth == 2

    ingestion = manager.append_staged_games(2)

    assert ingestion.games_ingested == 2
    assert ingestion.samples_added == 8
    assert ingestion.live_samples == 4
    assert ingestion.evicted_samples == 4
    assert ingestion.samples_per_second > 0.0
    assert tuple(game.length_plies for game in ingestion.completed_games) == (4, 4)
    assert tuple(game.termination_reason for game in ingestion.completed_games) == (
        TerminationReason.NATURAL,
        TerminationReason.NATURAL,
    )
    assert manager.staging_depth == 0
    assert manager.description().size == 4
    manager.close()

    reopened = _open_manager(tmp_path, capacity=4, maximum_capacity=6)
    assert reopened.live_samples == 4
    reopened.close()


def test_label_source_cohort_survives_shard_deletion_and_restart(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=16, maximum_capacity=16)
    manager.materialize_available_games()
    staged_manifest_path = next((tmp_path / 'completed-games' / 'staging').glob(f'*{MANIFEST_SUFFIX}'))
    staged_manifest_bytes = staged_manifest_path.read_bytes()
    staged_manifest_file_index = staged_manifest_path.stat().st_ino

    ingestion = manager.append_staged_games(2)
    cohort = manager.finalize_label_source_cohort(2)
    journal_text = (tmp_path / 'completed-games' / 'label-source-cohorts.json').read_text(encoding='utf-8')
    shard_paths = tuple((tmp_path / 'completed-games' / 'label-source-cohort-shards').glob(f'*{MANIFEST_SUFFIX}'))

    assert cohort.games == ingestion.label_source_games
    assert '"schema_version": 3' in journal_text
    assert '"games"' not in journal_text
    assert len(shard_paths) == 1
    assert shard_paths[0].read_bytes() == staged_manifest_bytes
    assert shard_paths[0].stat().st_ino == staged_manifest_file_index
    assert manager.pending_label_source_generations == (2,)
    assert manager.staging_depth == 0
    manager.close()

    restarted = _open_manager(tmp_path, capacity=16, maximum_capacity=16)
    recovered = restarted.finalize_label_source_cohort(2)
    assert recovered.games == ingestion.label_source_games
    restarted.acknowledge_label_source_cohort(2)
    restarted.acknowledge_label_source_cohort(2)
    assert restarted.pending_label_source_generations == ()
    assert not tuple((tmp_path / 'completed-games' / 'label-source-cohort-shards').iterdir())
    restarted.close()

    acknowledged = _open_manager(tmp_path, capacity=16, maximum_capacity=16)
    assert acknowledged.pending_label_source_generations == ()
    acknowledged.close()


def test_empty_label_source_cohort_is_finalized_for_generation_accounting(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=8, maximum_capacity=8)

    cohort = manager.finalize_label_source_cohort(7)

    assert cohort.games == ()
    assert manager.pending_label_source_generations == (7,)
    manager.acknowledge_label_source_cohort(7)
    assert manager.pending_label_source_generations == ()
    manager.close()


def test_open_empty_label_source_cohort_survives_pre_training_crash(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=8, maximum_capacity=8)

    cohort = manager.ensure_label_source_cohort(9)

    assert cohort.games == ()
    assert manager.pending_label_source_generations == (9,)
    manager.close()

    restarted = _open_manager(tmp_path, capacity=8, maximum_capacity=8)
    recovered = restarted.finalize_label_source_cohort(9)
    assert recovered.games == ()
    assert restarted.pending_label_source_generations == (9,)
    restarted.close()


def test_dispatch_moves_every_inbox_game_into_worker_directories_round_robin(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 6)
    manager = _open_manager(tmp_path, capacity=64, maximum_capacity=64, materialization_processes=3)

    assert manager.dispatch_once() == 6

    assert not tuple(inbox.glob('*.json'))
    assert [len(worker_source_file_names(path)) for path in manager.worker_paths] == [2, 2, 2]
    counters = [
        [parse_worker_source_file_name(name)[0] for name in sorted(worker_source_file_names(path))]
        for path in manager.worker_paths
    ]
    assert counters == [[0, 1], [0, 1], [0, 1]]
    manager.close()


def test_dispatch_is_bounded_by_its_rename_cap_and_resumes_on_the_next_pass(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 5)
    manager = _open_manager(tmp_path, capacity=64, maximum_capacity=64, inbox_rename_cap=2)

    assert manager.dispatch_once() == 2
    assert len(tuple(inbox.glob('*.json'))) == 3
    assert manager.dispatch_once() == 2
    assert manager.dispatch_once() == 1
    assert manager.dispatch_once() == 0
    manager.close()


def test_within_shard_order_follows_the_dispatch_counter(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 4)
    manager = _open_manager(tmp_path, capacity=64, maximum_capacity=64)
    manager.dispatch_once()
    dispatched = tuple(parse_worker_source_file_name(path.name) for path in _worker_source_paths(manager))

    manager.materialize_available_games()

    manifests = manager._staged_manifests()
    assert len(manifests) == 1
    sources = manifests[0].games
    assert tuple(game.source.counter for game in sources) == tuple(counter for counter, _ in dispatched)
    assert tuple(game.source.identity.file_name for game in sources) == tuple(name for _, name in dispatched)
    manager.close()


def test_materialization_uses_bounded_game_batches(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 5)
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32, shard_maximum_games=2)

    manager.materialize_available_games()

    assert sorted(len(manifest.games) for manifest in manager._staged_manifests()) == [1, 2, 2]
    assert manager.staging_depth == 5
    manager.close()


def test_oversize_source_forms_one_game_shard_without_blocking_later_games(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(
        tmp_path,
        capacity=16,
        maximum_capacity=16,
        shard_maximum_games=8,
        shard_maximum_source_bytes=1,
    )

    manager.materialize_available_games()

    assert sorted(len(manifest.games) for manifest in manager._staged_manifests()) == [1, 1]
    assert manager.inbox_depth == 0
    manager.close()


def test_capacity_resize_is_flushed_with_boundary_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 1)
    manager = _open_manager(tmp_path, capacity=8, maximum_capacity=8)
    manager.materialize_available_games()
    manager.store.set_logical_capacity(4)
    manager.store.flush()
    flush_count = 0
    original_flush = manager.store.flush

    def observe_flush() -> None:
        nonlocal flush_count
        flush_count += 1
        original_flush()

    monkeypatch.setattr(manager.store, 'flush', observe_flush)

    manager.append_staged_games(2)

    assert flush_count == 1
    assert manager.store.state.logical_capacity == 8
    manager.close()


def test_synthetic_flood_appends_every_game_exactly_once_with_bounded_inbox(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    manager = _open_manager(tmp_path, capacity=128, maximum_capacity=128, materialization_processes=4)
    appended_samples = 0
    flood_waves = 3
    games_per_wave = 10

    for wave in range(flood_waves):
        _publish_games(inbox, games_per_wave, first_game_number=wave * games_per_wave)
        manager.materialize_available_games()
        assert manager.inbox_depth == 0
        appended_samples += manager.append_staged_games(2).samples_added
        assert manager.staging_depth == 0

    total_games = flood_waves * games_per_wave
    assert appended_samples == total_games * SAMPLES_PER_GAME
    assert manager.store.total_appended_rows == total_games * SAMPLES_PER_GAME
    manager.close()


def test_total_materialized_samples_counts_only_appended_rows(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=16, maximum_capacity=16)

    manager.materialize_available_games()

    assert manager.staging_depth == 2
    assert manager.total_materialized_samples() == 0

    manager.append_staged_games(2)

    assert manager.total_materialized_samples() == 2 * SAMPLES_PER_GAME
    manager.close()


def test_restart_materializes_undispatched_and_unmaterialized_games_without_loss(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 4)
    manager = _open_manager(tmp_path, capacity=64, maximum_capacity=64, materialization_processes=2)
    manager.dispatch_once()
    _publish_games(inbox, 2, first_game_number=100)
    manager.close()

    restarted = _open_manager(tmp_path, capacity=64, maximum_capacity=64, materialization_processes=2)
    restarted.materialize_available_games()
    ingestion = restarted.append_staged_games(2)

    assert ingestion.games_ingested == 6
    assert restarted.store.total_appended_rows == 6 * SAMPLES_PER_GAME
    assert restarted.inbox_depth == 0
    restarted.close()


def test_restart_appends_a_sealed_but_unappended_shard(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=6, maximum_capacity=6)
    manager.materialize_available_games()
    assert manager.store.total_appended_rows == 0
    manager.close()

    restarted = _open_manager(tmp_path, capacity=6, maximum_capacity=6)

    assert restarted.staging_depth == 2
    ingestion = restarted.append_staged_games(2)
    assert ingestion.games_ingested == 2
    assert restarted.store.total_appended_rows == 2 * SAMPLES_PER_GAME
    restarted.close()


def test_restart_after_seal_without_unlink_does_not_double_ingest(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=64, maximum_capacity=64)
    manager.dispatch_once()
    sources = {path: path.read_bytes() for path in _worker_source_paths(manager)}
    manager.materialize_available_games()
    assert manager.staging_depth == 2
    # A worker killed between sealing the shard and unlinking its sources leaves both behind.
    for path, payload in sources.items():
        path.write_bytes(payload)
    manager.close()

    restarted = _open_manager(tmp_path, capacity=64, maximum_capacity=64)
    restarted.materialize_available_games()

    assert restarted.staging_depth == 2
    assert restarted.inbox_depth == 0
    ingestion = restarted.append_staged_games(2)
    assert ingestion.games_ingested == 2
    assert restarted.store.total_appended_rows == 2 * SAMPLES_PER_GAME
    restarted.close()


def test_malformed_game_is_quarantined_and_the_run_keeps_ingesting(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_malformed_games(inbox, 1)
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=16, maximum_capacity=16)

    manager.materialize_available_games()
    ingestion = manager.append_staged_games(2)

    assert ingestion.games_ingested == 2
    assert manager.materialization_failures == 1
    assert manager.inbox_depth == 0
    assert len(tuple(manager.rejected_path.glob('*.json'))) == 1
    manager.raise_if_materialization_failed()
    manager.close()


def test_a_systematic_rejection_rate_raises_instead_of_discarding_every_game(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_malformed_games(inbox, 4)
    manager = _open_manager(
        tmp_path,
        capacity=16,
        maximum_capacity=16,
        shard_maximum_games=1,
        rejection_window_games=4,
        rejection_rate_ceiling=0.5,
    )

    with pytest.raises(RuntimeError, match='above the configured ceiling'):
        manager.materialize_available_games()

    assert manager.rejection_rate == 1.0
    manager.close()


def test_orphaned_worker_directory_games_return_to_the_inbox(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 4)
    manager = _open_manager(tmp_path, capacity=64, maximum_capacity=64, materialization_processes=4)
    manager.dispatch_once()
    assert all(worker_source_file_names(path) for path in manager.worker_paths)
    manager.close()

    narrowed = _open_manager(tmp_path, capacity=64, maximum_capacity=64, materialization_processes=2)
    narrowed.materialize_available_games()
    ingestion = narrowed.append_staged_games(2)

    assert ingestion.games_ingested == 4
    assert not tuple((tmp_path / 'completed-games').glob('worker-2'))
    assert not tuple((tmp_path / 'completed-games').glob('worker-3'))
    narrowed.close()


def test_append_flushes_before_removing_staged_shards(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 1)
    manager = _open_manager(tmp_path, capacity=6, maximum_capacity=6)
    manager.materialize_available_games()
    manifest_path = replay_shard_manifest_path(manager.staging_path, manager._staged_manifests()[0].shard_identity)
    original_flush = manager.store.flush

    def assert_staged_rows_remain_until_flush() -> None:
        assert manifest_path.exists()
        original_flush()

    manager.store.flush = assert_staged_rows_remain_until_flush

    manager.append_staged_games(2)

    assert not manifest_path.exists()
    manager.store.flush = original_flush
    manager.close()


def test_low_budget_game_materializes_every_observation(tmp_path: Path) -> None:
    game = _completed_game().validated_copy(
        update={
            'observations': tuple(
                observation.validated_copy(
                    update={
                        'assigned_additional_visits': 1,
                        'final_visits': observation.starting_visits + 1,
                    }
                )
                for observation in _completed_game().observations
            )
        }
    )
    publish_completed_self_play_game(tmp_path / 'completed-games' / 'inbox', game)
    manager = _open_manager(tmp_path, capacity=4, maximum_capacity=4)
    manager.materialize_available_games()

    ingestion = manager.append_staged_games(2)

    assert ingestion.games_ingested == 1
    assert ingestion.samples_added == SAMPLES_PER_GAME
    assert manager.store.total_appended_rows == SAMPLES_PER_GAME
    manager.close()


def test_replay_ingestion_updates_central_resignation_state(tmp_path: Path) -> None:
    game = _completed_game().model_copy(
        update={
            'is_resignation_continuation': True,
            'observations': tuple(
                observation.model_copy(update={'root_value': -0.99, 'highest_visited_child_q': -0.99})
                for observation in _completed_game().observations
            ),
        }
    )
    inbox = tmp_path / 'completed-games' / 'inbox'
    publish_completed_self_play_game(inbox, game)
    resignation_configuration = CalibratedResignationConfiguration(
        first_production_generation=50,
        false_nonloss_rate_ceiling=0.99,
        continuation_game_probability=0.1,
        triggered_game_window=2000,
        candidate_threshold_minimum=-0.99,
        candidate_threshold_maximum=-0.70,
        candidate_threshold_step=0.01,
        minimum_evidence_trigger_count=1,
        confidence_level=0.95,
        maximum_relaxation_per_generation=0.01,
    )
    calibration_path = tmp_path / 'resignation' / 'calibration.json'
    calibrator = ResignationCalibrator(calibration_path, resignation_configuration)
    manager = _open_manager(tmp_path, capacity=4, maximum_capacity=4, resignation_calibrator=calibrator)

    manager.materialize_available_games()
    manager.append_staged_games(2)
    calibrator.advance_generation(50)
    manager.close()

    restarted = ResignationCalibrator(calibration_path, resignation_configuration)
    assert restarted.state.completed_continuation_games == 1
    assert restarted.state.broadest_candidate_triggers == 1
    assert restarted.published_policy(50).threshold == pytest.approx(-0.99)


def test_staged_shard_backlog_is_bounded_when_the_appender_stalls(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    manager = _open_manager(
        tmp_path,
        capacity=4096,
        maximum_capacity=4096,
        shard_maximum_games=1,
        staging_shard_limit=8,
    )
    _publish_games(inbox, 24)

    manager.materialize_available_games()

    assert manager.staging_depth == 8
    assert manager.inbox_depth == 16
    assert manager.append_staged_games(2).games_ingested == 8
    manager.close()


def test_append_with_nothing_staged_does_not_flush_the_store(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=8, maximum_capacity=8)
    flushes = 0
    original_flush = manager.store.flush

    def counted_flush() -> None:
        nonlocal flushes
        flushes += 1
        original_flush()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager.store, 'flush', counted_flush)
        ingestion = manager.append_staged_games(2)

    assert ingestion.games_ingested == 0
    assert flushes == 0
    manager.close()


def test_staged_shards_are_parsed_once_and_appended_without_reparsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=16, maximum_capacity=16)
    manager.materialize_available_games()
    assert manager.staging_depth == 2
    assert manager.staging_depth == 2

    parses = 0
    original_validate = SealedReplayShardManifest.model_validate_json

    def counted_validate(*arguments: object, **keywords: object) -> SealedReplayShardManifest:
        nonlocal parses
        parses += 1
        return original_validate(*arguments, **keywords)  # type: ignore[arg-type]

    monkeypatch.setattr(SealedReplayShardManifest, 'model_validate_json', counted_validate)
    ingestion = manager.append_staged_games(2)

    assert ingestion.games_ingested == 2
    assert parses == 0
    manager.close()


def _cut_game_with_trailing_observation() -> CompletedSelfPlayGame:
    full = _completed_game()
    # Cut the game short so the final position is non-terminal, which is what a ply cap produces.
    game = full.model_copy(update={'action_ids': full.action_ids[:2], 'observations': full.observations[:2]})
    position = LINEAR_STATE_CONTRACT.initial_position()
    for action_id in game.action_ids:
        position = LINEAR_STATE_CONTRACT.child_position(position, action_id)
    legal = LINEAR_STATE_CONTRACT.legal_action_ids(position)
    last = game.observations[-1]
    trailing = last.model_copy(
        update={
            'ply': len(game.action_ids),
            'selected_action_id': None,
            'root_value': 0.6,
            'policy_target_visits': SearchVisitCounts(action_ids=(legal[0],), visit_counts=(11,)),
            'highest_visited_child_action_id': legal[0],
            'highest_visited_child_visit_count': 11,
        }
    )
    return game.model_copy(
        update={
            'observations': game.observations + (trailing,),
            'termination_reason': TerminationReason.MAXIMUM_PLIES,
        }
    )


def test_cut_position_search_becomes_its_own_training_sample() -> None:
    cut = _cut_game_with_trailing_observation()
    without_trailing = cut.model_copy(update={'observations': cut.observations[:-1]})
    baseline = materialize_completed_game(
        without_trailing, LINEAR_STATE_CONTRACT, None, _target_layout(), 1, UNDISCOUNTED_VALUES
    )

    materialized = materialize_completed_game(
        cut,
        LINEAR_STATE_CONTRACT,
        None,
        _target_layout(),
        1,
        UNDISCOUNTED_VALUES,
    )

    # The searched cut position contributes one extra sample beyond the played plies.
    assert len(materialized.samples) == len(baseline.samples) + 1
    assert materialized.samples[-1].root_value == 0.6
    # It has no successor, so the next-policy target cannot be eligible for it.
    assert isinstance(materialized.samples[-1].auxiliary_targets[0], IneligibleNextPolicyTarget)


def _label_writeback_samples() -> tuple[ReplaySample, ...]:
    return tuple(
        materialize_completed_game(
            _completed_game(),
            LINEAR_STATE_CONTRACT,
            None,
            _target_layout(),
            1,
            UNDISCOUNTED_VALUES,
        ).samples
    )


def _prepared_label_writeback(
    manager: ReplayManager[LinearPosition],
    source_generation: int,
    samples: tuple[ReplaySample, ...],
) -> LabelledReplayWritebackReceipt:
    rows = encode_replay_rows(manager.store.layout, samples)
    state = manager.store.state
    return LabelledReplayWritebackReceipt(
        source_generation=source_generation,
        row_count=len(samples),
        rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        pre_append_total_rows=state.total_appended_rows,
        pre_append_head=state.head,
        pre_append_size=state.size,
        pre_append_sequence=state.append_sequence,
        committed=False,
    )


def _persist_prepared_writeback(
    manager: ReplayManager[LinearPosition],
    receipt: LabelledReplayWritebackReceipt,
) -> None:
    state = LabelledReplayWritebackState(receipts=(receipt,))
    write_text_atomically(manager._labelled_writeback_path, state.model_dump_json(indent=2) + '\n')


def test_labelled_replay_writeback_is_idempotent_after_commit(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    samples = _label_writeback_samples()

    first = manager.append_labelled_samples(2, samples)
    second = manager.append_labelled_samples(2, samples)

    assert first.applied
    assert not second.applied
    assert manager.store.total_appended_rows == len(samples)
    manager.close()


def test_labelled_replay_writeback_does_not_count_as_new_materialized_data(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    _publish_games(inbox, 2)
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    manager.materialize_available_games()
    manager.append_staged_games(2)
    materialized_samples = manager.total_materialized_samples()

    samples = _label_writeback_samples()
    manager.append_labelled_samples(2, samples)

    assert manager.store.total_appended_rows == materialized_samples + len(samples)
    assert manager.total_materialized_samples() == materialized_samples
    manager.close()


def test_prepared_label_writeback_recovers_before_store_append(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    samples = _label_writeback_samples()
    receipt = _prepared_label_writeback(manager, 2, samples)
    _persist_prepared_writeback(manager, receipt)
    manager.close()
    restarted = _open_manager(tmp_path, capacity=32, maximum_capacity=32)

    recovered = restarted.append_labelled_samples(2, samples)

    assert recovered.applied
    assert restarted.store.total_appended_rows == len(samples)
    assert (
        LabelledReplayWritebackState.model_validate_json(restarted._labelled_writeback_path.read_text(encoding='utf-8'))
        .receipts[0]
        .committed
    )
    restarted.close()


def test_prepared_label_writeback_recovers_after_store_append(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    samples = _label_writeback_samples()
    receipt = _prepared_label_writeback(manager, 2, samples)
    _persist_prepared_writeback(manager, receipt)
    rows = encode_replay_rows(manager.store.layout, samples)
    manager.store.extend_rows(rows, transaction_identity='search-budget-labels-2')
    manager.store.flush()
    manager.close()
    restarted = _open_manager(tmp_path, capacity=32, maximum_capacity=32)

    recovered = restarted.append_labelled_samples(2, samples)

    assert not recovered.applied
    assert restarted.store.total_appended_rows == len(samples)
    restarted.close()


def test_prepared_label_writeback_fails_closed_after_intervening_append(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    samples = _label_writeback_samples()
    receipt = _prepared_label_writeback(manager, 2, samples)
    _persist_prepared_writeback(manager, receipt)
    rows = encode_replay_rows(manager.store.layout, samples)
    manager.store.extend_rows(rows, transaction_identity='ordinary-materialization')
    manager.store.flush()
    manager.close()
    restarted = _open_manager(tmp_path, capacity=32, maximum_capacity=32)

    with pytest.raises(ValueError, match='ambiguous'):
        restarted.append_labelled_samples(2, samples)

    assert restarted.store.total_appended_rows == len(samples)
    restarted.close()


def test_prepared_label_writeback_never_duplicates_after_label_then_ordinary_append(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    samples = _label_writeback_samples()
    receipt = _prepared_label_writeback(manager, 2, samples)
    _persist_prepared_writeback(manager, receipt)
    rows = encode_replay_rows(manager.store.layout, samples)
    manager.store.extend_rows(rows, transaction_identity='search-budget-labels-2')
    manager.store.extend_rows(rows, transaction_identity='ordinary-materialization')
    manager.store.flush()
    manager.close()
    restarted = _open_manager(tmp_path, capacity=32, maximum_capacity=32)

    with pytest.raises(ValueError, match='ambiguous'):
        restarted.append_labelled_samples(2, samples)

    assert restarted.store.total_appended_rows == 2 * len(samples)
    restarted.close()


def test_training_snapshot_lease_blocks_labelled_replay_append(tmp_path: Path) -> None:
    manager = _open_manager(tmp_path, capacity=32, maximum_capacity=32)
    samples = _label_writeback_samples()
    started = threading.Event()
    finished = threading.Event()

    def append_labels() -> None:
        started.set()
        manager.append_labelled_samples(2, samples)
        finished.set()

    with manager.training_snapshot() as description:
        thread = threading.Thread(target=append_labels)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)
        assert description == manager.description()

    thread.join(timeout=1.0)
    assert finished.is_set()
    assert manager.store.total_appended_rows == len(samples)
    manager.close()
