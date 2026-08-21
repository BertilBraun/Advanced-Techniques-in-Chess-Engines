from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest
from AlphaZeroCpp import GameSearchVisit
from src.experiment.generation_schedule import ConstantSchedule
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
    ReplaySample,
)
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayManager
from src.replay.materialization import materialize_completed_game
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
    SearchCorrectionHeadLayout,
    TrainingTargetLayout,
)


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

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        return action_id

    def transform_encoded_state(
        self,
        encoded_state: PackedPlanePayload,
        augmentation_index: int,
    ) -> PackedPlanePayload:
        return encoded_state

    def transform_replay_targets(self, sample: ReplaySample, augmentation_index: int) -> ReplaySample:
        return sample


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
                policy_target_visits=SearchVisitCounts.from_native(
                    (
                        GameSearchVisit(action_id=other_action, visit_count=3),
                        GameSearchVisit(action_id=selected_action, visit_count=10),
                    )
                ),
                root_value=0.25,
                highest_visited_child_action_id=selected_action,
                highest_visited_child_visit_count=10,
                highest_visited_child_q=0.2,
                selected_action_id=selected_action,
                full_search=ply != 1,
                sample_weight=1.0,
                search_budget=13,
                network_root_value=0.1,
                policy_correction=0.2,
                value_correction=0.075,
                search_correction_target=0.2,
                predicted_search_correction=0.15,
                starting_visits=0,
                final_visits=13,
                stop_reason=SearchStopReason.FIXED_LIMIT,
                learned_gate_evaluated=False,
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

    assert len(materialized.samples) == 3
    assert materialized.policies_truncated == 5
    assert materialized.retained_visit_mass == 50
    assert materialized.discarded_visit_mass == 15
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
                policy_target_visits=SearchVisitCounts.from_native((GameSearchVisit(action_id=2, visit_count=8),)),
                root_value=0.0,
                highest_visited_child_action_id=2,
                highest_visited_child_visit_count=8,
                highest_visited_child_q=0.0,
                selected_action_id=2,
                full_search=True,
                sample_weight=1.0,
                search_budget=8,
                network_root_value=0.0,
                policy_correction=0.0,
                value_correction=0.0,
                search_correction_target=0.0,
                predicted_search_correction=0.0,
                starting_visits=0,
                final_visits=8,
                stop_reason=SearchStopReason.FIXED_LIMIT,
                learned_gate_evaluated=False,
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
    ) == pytest.approx((1.0, 0.5, 0.25))


def test_four_ply_future_value_uses_terminal_fallback_with_current_perspective() -> None:
    targets = TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(FutureSearchValueHeadLayout(kind='future_search_value', ply_offset=4, smooth_l1_beta=0.1),),
    )

    game = _completed_game().model_copy(
        update={
            'observations': tuple(
                observation.model_copy(update={'full_search': True}) for observation in _completed_game().observations
            )
        }
    )
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
    assert event_targets[1:] == (
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
    assert isinstance(censored.samples[1].auxiliary_targets[0], IneligibleScalarAuxiliaryTarget)


def test_search_correction_materializes_final_larger_correction() -> None:
    targets = TrainingTargetLayout(
        action_size=3,
        wdl_size=3,
        auxiliary_heads=(SearchCorrectionHeadLayout(kind='search_correction'),),
    )
    observation = (
        _completed_game()
        .observations[0]
        .model_copy(
            update={
                'policy_correction': 0.1,
                'value_correction': 0.3,
                'search_correction_target': 0.3,
            }
        )
    )
    game = _completed_game().model_copy(update={'observations': (observation, *_completed_game().observations[1:])})

    materialized = materialize_completed_game(
        game,
        LINEAR_STATE_CONTRACT,
        None,
        targets,
        3,
        UNDISCOUNTED_VALUES,
    )

    assert materialized.samples[0].auxiliary_targets[0] == EligibleScalarAuxiliaryTarget(
        kind='search_correction',
        value=0.3,
    )


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
    assert materialized.samples[1].wdl_target == WdlTarget(win=0.25, draw=0.25, loss=0.5)
    assert materialized.samples[2].wdl_target == WdlTarget(
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


def test_replay_manager_drains_all_games_and_reopens_fifo(tmp_path: Path) -> None:
    game = _completed_game()
    inbox = tmp_path / 'completed-games' / 'inbox'
    publish_completed_self_play_game(inbox, game)
    second_game = game.validated_copy(
        update={
            'identity': {
                'worker_id': 3,
                'process_instance_id': '38c8809f-a49d-4d98-8da5-034614893665',
                'game_number': 8,
            }
        }
    )
    publish_completed_self_play_game(inbox, second_game)
    configuration = ReplayConfiguration(
        capacity=4,
        maximum_capacity=6,
        maximum_policy_entries=1,
    )
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )
    manager = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
    )

    ingestion = manager.ingest_available_games(2)

    assert ingestion.games_ingested == 2
    assert ingestion.samples_added == 6
    assert ingestion.live_samples == 4
    assert ingestion.evicted_samples == 2
    assert ingestion.samples_per_second > 0.0
    assert tuple(game.length_plies for game in ingestion.completed_games) == (4, 4)
    assert tuple(game.termination_reason for game in ingestion.completed_games) == (
        TerminationReason.NATURAL,
        TerminationReason.NATURAL,
    )
    assert not tuple(inbox.glob('*.json'))
    description = manager.description()
    assert description.size == 4
    manager.close()

    reopened = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
    )
    assert reopened.live_samples == 4
    reopened.close()


def test_replay_manager_stops_after_reaching_sample_limit(tmp_path: Path) -> None:
    game = _completed_game()
    inbox = tmp_path / 'completed-games' / 'inbox'
    publish_completed_self_play_game(inbox, game)
    second_game = game.validated_copy(
        update={
            'identity': {
                'worker_id': 3,
                'process_instance_id': '38c8809f-a49d-4d98-8da5-034614893665',
                'game_number': 8,
            }
        }
    )
    publish_completed_self_play_game(inbox, second_game)
    configuration = ReplayConfiguration(capacity=6, maximum_capacity=6, maximum_policy_entries=1)
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )
    manager = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
    )

    ingestion = manager.ingest_available_games(2, maximum_samples=1)

    assert ingestion.games_ingested == 1
    assert ingestion.samples_added == 3
    assert len(tuple(inbox.glob('*.json'))) == 1
    manager.close()


def test_replay_manager_does_not_flush_when_inbox_is_empty(tmp_path: Path) -> None:
    configuration = ReplayConfiguration(capacity=6, maximum_capacity=6, maximum_policy_entries=1)
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )
    manager = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
    )
    flush = Mock()
    manager.store.flush = flush

    ingestion = manager.ingest_available_games(2)

    assert ingestion.games_ingested == 0
    flush.assert_not_called()
    manager.close()


def test_replay_manager_flushes_before_removing_ingested_games(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    publish_completed_self_play_game(inbox, _completed_game())
    configuration = ReplayConfiguration(capacity=6, maximum_capacity=6, maximum_policy_entries=1)
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )
    manager = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
    )
    original_flush = manager.store.flush
    completed_path = next(inbox.glob('*.json'))

    def assert_game_remains_until_flush() -> None:
        assert completed_path.exists()
        original_flush()

    manager.store.flush = assert_game_remains_until_flush

    manager.ingest_available_games(2)

    assert not completed_path.exists()
    manager.store.flush = original_flush
    manager.close()


def test_replay_manager_keeps_malformed_game_for_inspection(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    inbox.mkdir(parents=True)
    malformed = inbox / 'malformed.json'
    malformed.write_text('{}', encoding='utf-8')
    configuration = ReplayConfiguration(
        capacity=4,
        maximum_capacity=4,
        maximum_policy_entries=1,
    )
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )
    manager = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        configuration,
        model_generation=0,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
    )

    with pytest.raises(ValueError):
        manager.ingest_available_games(0)

    assert malformed.exists()
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
    replay_configuration = ReplayConfiguration(
        capacity=4,
        maximum_capacity=4,
        maximum_policy_entries=1,
    )
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
        maximum_legal_actions=LINEAR_STATE_CONTRACT.maximum_legal_action_count,
    )
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
    manager = ReplayManager.open(
        tmp_path,
        LINEAR_STATE_CONTRACT,
        layout,
        replay_configuration,
        model_generation=2,
        value_discount_per_ply=UNDISCOUNTED_VALUES,
        terminal_oracle=None,
        resignation_calibrator=calibrator,
    )

    manager.ingest_available_games(2)
    calibrator.advance_generation(50)
    manager.close()

    restarted = ResignationCalibrator(calibration_path, resignation_configuration)
    assert restarted.state.completed_continuation_games == 1
    assert restarted.state.broadest_candidate_triggers == 1
    assert restarted.published_policy(50).threshold == pytest.approx(-0.99)
