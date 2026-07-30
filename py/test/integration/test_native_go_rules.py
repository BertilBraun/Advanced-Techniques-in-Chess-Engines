from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

try:
    import az_go_native as native
except ImportError:
    native = None

pytestmark = pytest.mark.skipif(native is None, reason='focused native Go extension has not been built')

from test.oracles.go_rules import (
    OracleGoState,
    OracleRules,
    OracleTermination,
)
from src.az.games.go.configuration import MAXIMUM_HISTORY_LENGTH


def native_state(board_size: int, komi_half_points: int = 15, safety_ply_cap: int = 180) -> native.GoState:
    rules = native.GoRules(board_size, komi_half_points, safety_ply_cap, history_length=4)
    return native.GoState(rules)


def test_native_and_python_history_limits_match() -> None:
    assert native.MAXIMUM_HISTORY_LENGTH == MAXIMUM_HISTORY_LENGTH


def assert_states_agree(native_game: native.GoState, oracle: OracleGoState) -> None:
    assert native_game.board_size == oracle.rules.board_size
    assert native_game.action_count == oracle.action_count
    assert native_game.pass_action == oracle.pass_action
    assert int(native_game.current_player) == int(oracle.current_player)
    assert native_game.ply == oracle.ply
    assert native_game.consecutive_passes == oracle.consecutive_passes
    assert int(native_game.termination_reason) == int(oracle.termination)
    assert [int(stone) for stone in native_game.board] == [int(stone) for stone in oracle.board]
    assert native_game.legal_actions() == oracle.legal_actions()
    native_encoding = native_game.canonical_encoding()
    oracle_planes, oracle_size, oracle_values = oracle.canonical_encoding()
    assert native_encoding.planes == oracle_planes
    assert native_encoding.board_size == oracle_size
    assert tuple(native_encoding.values) == oracle_values
    native_score = native_game.area_score()
    oracle_score = oracle.area_score()
    assert native_score.black_twice == oracle_score.black_twice
    assert native_score.white_twice == oracle_score.white_twice


@pytest.mark.parametrize('board_size', [3, 5, 7, 9, 13])
@pytest.mark.parametrize('seed', [20260730, 20260731])
def test_randomized_legal_trajectories_match_oracle(board_size: int, seed: int) -> None:
    oracle = OracleGoState(
        OracleRules(
            board_size=board_size,
            komi_half_points=15,
            safety_ply_cap=board_size * board_size * 3,
            history_length=4,
        )
    )
    native_game = native_state(board_size, safety_ply_cap=board_size * board_size * 3)
    generator = random.Random(seed)
    while oracle.termination == OracleTermination.ONGOING:
        assert_states_agree(native_game, oracle)
        legal = oracle.legal_actions()
        placements = [action for action in legal if action != oracle.pass_action]
        action = oracle.pass_action if not placements or generator.random() < 0.04 else generator.choice(placements)
        oracle.apply(action)
        native_game.apply(action)
    assert_states_agree(native_game, oracle)
    native_result = native_game.terminal_result()
    assert int(native_result.reason) == int(oracle.termination)
    if oracle.termination == OracleTermination.SAFETY_PLY_CAP:
        assert native_result.score is None
        assert native_result.winner is None
    else:
        oracle_score = oracle.area_score()
        assert native_result.score.black_twice == oracle_score.black_twice
        assert native_result.score.white_twice == oracle_score.white_twice
        expected_winner = oracle_score.winner
        assert (None if native_result.winner is None else int(native_result.winner)) == (
            None if expected_winner is None else int(expected_winner)
        )


@pytest.mark.parametrize('board_size', [3, 5, 7, 9, 13])
def test_pass_is_last_action_and_exempt_from_repetition(board_size: int) -> None:
    native_game = native_state(board_size)
    assert native_game.pass_action == board_size * board_size
    native_game.apply(native_game.pass_action)
    assert native_game.is_legal(native_game.pass_action)
    native_game.apply(native_game.pass_action)
    assert native_game.is_terminal
    assert native_game.termination_reason == native.TerminationReason.TWO_PASSES


def restored_state(
    stones: Sequence[tuple[int, native.Stone]],
    current_player: native.Player | None = None,
    history: Sequence[Sequence[native.Stone]] | None = None,
) -> native.GoState:
    board = [native.Stone.EMPTY] * 49
    for point, stone in stones:
        board[point] = stone
    positions = [board] if history is None else [list(item) for item in history]
    player = native.Player.BLACK if current_player is None else current_player
    return native.GoState.restore(
        native.GoRules(7, 15, 200, 4),
        board,
        player,
        len(positions) - 1,
        0,
        positions,
    )


def test_capture_multi_stone_suicide_ko_and_longer_superko() -> None:
    capture = restored_state(
        (
            (8, native.Stone.WHITE),
            (1, native.Stone.BLACK),
            (7, native.Stone.BLACK),
            (9, native.Stone.BLACK),
        )
    )
    capture.apply(15)
    assert capture.board[8] == native.Stone.EMPTY

    multi_capture = restored_state(
        (
            (8, native.Stone.WHITE),
            (9, native.Stone.WHITE),
            (1, native.Stone.BLACK),
            (2, native.Stone.BLACK),
            (7, native.Stone.BLACK),
            (10, native.Stone.BLACK),
            (15, native.Stone.BLACK),
        )
    )
    multi_capture.apply(16)
    assert multi_capture.board[8] == native.Stone.EMPTY
    assert multi_capture.board[9] == native.Stone.EMPTY

    suicide = restored_state(
        (
            (1, native.Stone.BLACK),
            (2, native.Stone.WHITE),
            (7, native.Stone.WHITE),
            (8, native.Stone.WHITE),
        )
    )
    assert not suicide.is_legal(0)

    before = [native.Stone.EMPTY] * 49
    for point in (8, 14, 16, 22):
        before[point] = native.Stone.WHITE
    for point in (1, 7, 9):
        before[point] = native.Stone.BLACK
    ko = native.GoState.restore(
        native.GoRules(7, 15, 200, 4),
        before,
        native.Player.BLACK,
        0,
        0,
        [before],
    )
    ko.apply(15)
    assert not ko.is_legal(8)

    repeated = list(ko.board)
    middle = [native.Stone.EMPTY] * 49
    middle[30] = native.Stone.WHITE
    longer_cycle = native.GoState.restore(
        native.GoRules(7, 15, 200, 4),
        before,
        native.Player.BLACK,
        2,
        0,
        [repeated, middle, before],
    )
    assert not longer_cycle.is_legal(15)


def test_copy_hash_cap_scoring_and_restore_validation() -> None:
    native_game = native.GoState(native.GoRules(7, 15, 200, 4))
    copied = native_game.copy()
    assert copied == native_game
    assert copied.state_hash() == native_game.state_hash()
    capped = native.GoState(native.GoRules(7, 15, 49, 4))
    for _ in range(48):
        placement = next(action for action in capped.legal_actions() if action != capped.pass_action)
        capped.apply(placement)
    capped.apply(capped.pass_action)
    assert capped.termination_reason == native.TerminationReason.SAFETY_PLY_CAP
    assert capped.terminal_result().score is None
    assert capped.terminal_result().winner is None

    neutral = restored_state(((0, native.Stone.BLACK), (48, native.Stone.WHITE)))
    score = neutral.area_score()
    assert (score.black_twice, score.white_twice) == (2, 17)

    with pytest.raises(ValueError, match='history'):
        native.GoState.restore(
            native.GoRules(7, 15, 200, 4),
            [native.Stone.EMPTY] * 49,
            native.Player.BLACK,
            0,
            0,
            [],
        )


@pytest.mark.parametrize('board_size', [3, 5, 7, 9, 13])
def test_all_symmetry_round_trips(board_size: int) -> None:
    native_game = native_state(board_size)
    native_game.apply(0)
    native_game.apply(board_size + 1)
    encoding = native_game.canonical_encoding()
    symmetries = (
        native.Symmetry.IDENTITY,
        native.Symmetry.ROTATE_90,
        native.Symmetry.ROTATE_180,
        native.Symmetry.ROTATE_270,
        native.Symmetry.REFLECT,
        native.Symmetry.REFLECT_ROTATE_90,
        native.Symmetry.REFLECT_ROTATE_180,
        native.Symmetry.REFLECT_ROTATE_270,
    )
    for symmetry in symmetries:
        inverse = native.inverse_symmetry(symmetry)
        for action in range(native_game.action_count):
            transformed = native.transform_action(action, board_size, symmetry)
            assert native.transform_action(transformed, board_size, inverse) == action
        transformed_encoding = native.transform_encoding(encoding, symmetry)
        assert native.transform_encoding(transformed_encoding, inverse) == encoding


def test_oracle_is_not_a_production_dependency() -> None:
    from src.az.games.go import module

    assert 'test.oracles' not in module.__dict__
