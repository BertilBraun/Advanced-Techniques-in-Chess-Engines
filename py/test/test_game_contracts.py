from pathlib import Path

import chess
import pytest
from pydantic import ValidationError

from src.experiment.configuration import load_experiment_configuration
from src.games.chess.board import ChessBoard
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.training import ChessImplementation
from src.games.contracts import Player, WdlTarget
from src.replay.contracts import EligibleNextPolicyTarget, IneligibleNextPolicyTarget, SparsePolicyTarget
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SparseSearchVisit,
    TerminationReason,
)


def test_wdl_target_validates_and_reverses_perspective() -> None:
    target = WdlTarget.from_scalar(0.4)

    assert (target.win, target.draw, target.loss) == pytest.approx((0.6, 0.2, 0.2))
    reversed_target = target.reversed()
    assert (reversed_target.win, reversed_target.draw, reversed_target.loss) == pytest.approx((0.2, 0.2, 0.6))
    with pytest.raises(ValidationError, match='sum to one'):
        WdlTarget(win=0.5, draw=0.5, loss=0.5)


def test_chess_state_contract_operates_in_action_id_space() -> None:
    initial = CHESS_STATE_CONTRACT.initial_position()
    legal_actions = CHESS_STATE_CONTRACT.legal_action_ids(initial)
    child = CHESS_STATE_CONTRACT.child_position(initial, legal_actions[0])

    assert legal_actions
    assert initial.board == chess.Board()
    assert child.board != initial.board
    assert CHESS_STATE_CONTRACT.current_player(initial) is Player.FIRST
    assert CHESS_STATE_CONTRACT.current_player(child) is Player.SECOND
    assert CHESS_STATE_CONTRACT.terminal_wdl(initial) is None
    assert (
        len(CHESS_STATE_CONTRACT.encode_network_input(initial))
        == CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes
    )


def test_chess_adjudication_uses_fixed_starting_material_normalization() -> None:
    position = ChessBoard.from_fen('4k3/8/8/8/8/8/8/3QK3 w - - 0 1')

    target = CHESS_STATE_CONTRACT.adjudicated_wdl(position)

    assert target == WdlTarget.from_scalar(9 / 39)


@pytest.mark.parametrize(
    ('path', 'expected_action_size', 'expected_augmentations'),
    ((Path('configs/chess-experiment-template.yaml'), 1_880, 2),),
)
def test_root_game_implementation_owns_state_and_fixed_target_layout(
    path: Path,
    expected_action_size: int,
    expected_augmentations: int,
) -> None:
    configuration = load_experiment_configuration(path)
    assert configuration.game == 'chess'
    implementation = ChessImplementation(configuration)

    assert implementation.state.action_size == expected_action_size
    assert implementation.state.augmentation_count == expected_augmentations
    assert implementation.target_layout.action_size == expected_action_size
    assert implementation.target_layout.wdl_size == 3
    assert implementation.target_layout.auxiliary_heads == ()


def test_completed_self_play_game_round_trip_uses_shared_trajectory_values() -> None:
    observation = SearchObservation(
        ply=0,
        model_generation=3,
        visits=(SparseSearchVisit(action_id=7, visit_count=12),),
        root_value=0.25,
        selected_action_id=7,
        full_search=True,
        sample_weight=1.0,
        search_budget=16,
        minimum_root_visits=0,
    )
    game = CompletedSelfPlayGame(
        identity=GameIdentity(run_id=1, worker_id=2, game_number=3),
        created_at_seconds=4.0,
        generation_seconds=5.0,
        action_ids=(7,),
        observations=(observation,),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.NATURAL,
    )

    assert CompletedSelfPlayGame.model_validate_json(game.model_dump_json()) == game
    with pytest.raises(ValidationError, match='selected actions'):
        game.validated_copy(update={'action_ids': [8]})


def test_auxiliary_target_layout_is_run_fixed_and_ordered() -> None:
    configuration = load_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    assert configuration.game == 'chess'
    objective = configuration.chess.objective.validated_copy(
        update={
            'auxiliary_targets': [
                {
                    'kind': 'next_policy',
                    'ply_offset': 1,
                    'loss_weight': {'kind': 'constant', 'value': 0.25},
                },
                {
                    'kind': 'next_policy',
                    'ply_offset': 3,
                    'loss_weight': {'kind': 'constant', 'value': 0.1},
                },
            ]
        }
    )
    chess_configuration = configuration.chess.validated_copy(update={'objective': objective.model_dump(mode='json')})
    configured = configuration.validated_copy(update={'chess': chess_configuration.model_dump(mode='json')})

    layout = ChessImplementation(configured).target_layout

    assert tuple(head.ply_offset for head in layout.auxiliary_heads) == (1, 3)
    assert all(head.action_size == 1_880 for head in layout.auxiliary_heads)


def test_next_policy_eligibility_is_explicit_and_uses_future_action_space() -> None:
    future_policy = SparsePolicyTarget(visits=(SparseSearchVisit(action_id=42, visit_count=10),))

    eligible = EligibleNextPolicyTarget(policy=future_policy)
    terminal_tail = IneligibleNextPolicyTarget()

    assert eligible.eligible
    assert eligible.policy.visits[0].action_id == 42
    assert not terminal_tail.eligible
