from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from src.games.contracts import GameStateContract, Player, RepresentationDimensions, WdlTarget
from src.packed_planes import PackedPlaneLayout, PackedPlanePayload
from src.replay.contracts import EligibleNextPolicyTarget, IneligibleNextPolicyTarget, ReplaySample
from src.replay.layout import ReplayLayout
from src.replay.manager import ReplayManager
from src.replay.materialization import materialize_completed_game
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SparseSearchVisit,
    TerminationReason,
    publish_completed_self_play_game,
)
from src.training.configuration import ReplayConfiguration
from src.training.targets import NextPolicyHeadLayout, TrainingTargetLayout


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
                visits=(
                    SparseSearchVisit(action_id=other_action, visit_count=3),
                    SparseSearchVisit(action_id=selected_action, visit_count=10),
                ),
                root_value=0.25,
                selected_action_id=selected_action,
                full_search=ply != 1,
                sample_weight=1.0,
                search_budget=13,
                minimum_root_visits=1,
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
    materialized = materialize_completed_game(_completed_game(), LINEAR_STATE_CONTRACT, _target_layout(), 1)

    assert len(materialized.samples) == 3
    assert materialized.policies_truncated == 5
    assert materialized.retained_visit_mass == 45
    assert materialized.discarded_visit_mass == 10
    assert materialized.samples[0].wdl_target == WdlTarget(win=0.0, draw=0.0, loss=1.0)
    assert isinstance(materialized.samples[0].auxiliary_targets[0], EligibleNextPolicyTarget)
    assert isinstance(materialized.samples[-1].auxiliary_targets[0], IneligibleNextPolicyTarget)
    assert materialized.samples[0].policy.visits[0].visit_count == 9


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
        capacity={'kind': 'constant', 'value': 4},
        maximum_capacity=6,
        maximum_policy_entries=1,
    )
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
    )
    manager = ReplayManager.open(tmp_path, LINEAR_STATE_CONTRACT, layout, configuration, model_generation=2)

    ingestion = manager.ingest_available_games(2)

    assert ingestion.games_ingested == 2
    assert ingestion.samples_added == 6
    assert ingestion.live_samples == 4
    assert ingestion.evicted_samples == 2
    assert ingestion.samples_per_second > 0.0
    assert not tuple(inbox.glob('*.json'))
    description = manager.description()
    assert description.size == 4
    manager.close()

    reopened = ReplayManager.open(tmp_path, LINEAR_STATE_CONTRACT, layout, configuration, model_generation=2)
    assert reopened.live_samples == 4
    reopened.close()


def test_replay_manager_keeps_malformed_game_for_inspection(tmp_path: Path) -> None:
    inbox = tmp_path / 'completed-games' / 'inbox'
    inbox.mkdir(parents=True)
    malformed = inbox / 'malformed.json'
    malformed.write_text('{}', encoding='utf-8')
    configuration = ReplayConfiguration(
        capacity={'kind': 'constant', 'value': 4},
        maximum_capacity=4,
        maximum_policy_entries=1,
    )
    layout = ReplayLayout(
        packed_planes=LINEAR_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
    )
    manager = ReplayManager.open(tmp_path, LINEAR_STATE_CONTRACT, layout, configuration, model_generation=0)

    with pytest.raises(ValueError):
        manager.ingest_available_games(0)

    assert malformed.exists()
    manager.close()
