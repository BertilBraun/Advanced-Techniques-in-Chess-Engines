from pathlib import Path
from uuid import UUID

import chess
import pytest

from src.games.chess.board import MAX_MATERIAL_VALUE, PIECE_VALUE, ChessBoard
from src.games.chess.contract import ChessStateContract
from src.games.chess.game import ChessGame
from src.games.contracts import Player, WdlTarget
from src.packed_planes import PackedPlanePayload, encode_packed_planes
from src.replay.contracts import EligibleNextPolicyTarget, IneligibleNextPolicyTarget
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


class PythonChessStateContract(ChessStateContract):
    def __init__(self) -> None:
        super().__init__()
        self.game = ChessGame()

    def initial_position(self) -> ChessBoard:
        return ChessBoard()

    def legal_action_ids(self, position: ChessBoard) -> tuple[int, ...]:
        return tuple(sorted(self.game.encode_move(move, position) for move in position.get_valid_moves()))

    def child_position(self, position: ChessBoard, action_id: int) -> ChessBoard:
        child = position.copy()
        child.make_move(self.game.decode_move(action_id, position))
        return child

    def current_player(self, position: ChessBoard) -> Player:
        return Player(position.current_player)

    def terminal_wdl(self, position: ChessBoard) -> WdlTarget | None:
        if not position.is_game_over():
            return None
        winner = position.check_winner()
        if winner is None:
            return WdlTarget(win=0.0, draw=1.0, loss=0.0)
        return WdlTarget.from_scalar(1.0 if winner == position.current_player else -1.0)

    def adjudicated_wdl(self, position: ChessBoard, reason: TerminationReason) -> WdlTarget:
        material_score = 0
        for piece_type, value in PIECE_VALUE.items():
            material_score += value * len(position.board.pieces(piece_type, position.board.turn))
            material_score -= value * len(position.board.pieces(piece_type, not position.board.turn))
        return WdlTarget.from_scalar(material_score / MAX_MATERIAL_VALUE)

    def encode_network_input(self, position: ChessBoard) -> PackedPlanePayload:
        state = self.game.get_canonical_board(position)
        return encode_packed_planes(
            state,
            self.representation.packed_planes,
            self.representation.binary_channels,
            self.representation.scalar_channels,
        )


PYTHON_CHESS_STATE_CONTRACT = PythonChessStateContract()


def _completed_game() -> CompletedSelfPlayGame:
    moves = ('f2f3', 'e7e5', 'g2g4', 'd8h4')
    board = ChessBoard()
    action_ids: list[int] = []
    observations: list[SearchObservation] = []
    for ply, move_uci in enumerate(moves):
        move = chess.Move.from_uci(move_uci)
        selected_action = PYTHON_CHESS_STATE_CONTRACT.game.encode_move(move, board)
        other_action = next(
            action for action in PYTHON_CHESS_STATE_CONTRACT.legal_action_ids(board) if action != selected_action
        )
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
        board.make_move(move)
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
        action_size=PYTHON_CHESS_STATE_CONTRACT.action_size,
        wdl_size=3,
        auxiliary_heads=(
            NextPolicyHeadLayout(kind='next_policy', action_size=PYTHON_CHESS_STATE_CONTRACT.action_size, ply_offset=1),
        ),
    )


def test_shared_materialization_reconstructs_perspective_and_trajectory_targets() -> None:
    materialized = materialize_completed_game(_completed_game(), PYTHON_CHESS_STATE_CONTRACT, _target_layout(), 1)

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
        packed_planes=PYTHON_CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
    )
    manager = ReplayManager.open(tmp_path, PYTHON_CHESS_STATE_CONTRACT, layout, configuration, model_generation=2)

    ingestion = manager.ingest_available_games(2)

    assert ingestion.games_ingested == 2
    assert ingestion.samples_added == 6
    assert ingestion.live_samples == 4
    assert ingestion.evicted_samples == 2
    assert not tuple(inbox.glob('*.json'))
    description = manager.description()
    assert description.size == 4
    manager.close()

    reopened = ReplayManager.open(tmp_path, PYTHON_CHESS_STATE_CONTRACT, layout, configuration, model_generation=2)
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
        packed_planes=PYTHON_CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=_target_layout(),
        maximum_policy_entries=1,
    )
    manager = ReplayManager.open(tmp_path, PYTHON_CHESS_STATE_CONTRACT, layout, configuration, model_generation=0)

    with pytest.raises(ValueError):
        manager.ingest_available_games(0)

    assert malformed.exists()
    manager.close()
