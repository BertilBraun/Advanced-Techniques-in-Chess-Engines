from __future__ import annotations

import pytest

AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

if not hasattr(AlphaZeroCpp, 'GoPosition7'):
    pytest.skip('AlphaZeroCpp must be rebuilt before native Go tests run.', allow_module_level=True)


def rules(maximum_moves: int = 200) -> AlphaZeroCpp.GoRules:
    return AlphaZeroCpp.GoRules(15, maximum_moves)


def restore_7(
    black: list[int],
    white: list[int],
    player: AlphaZeroCpp.GoPlayer,
    *,
    ko_point: int | None = None,
    consecutive_passes: int = 0,
    move_number: int = 0,
    maximum_moves: int = 200,
) -> AlphaZeroCpp.GoPosition7:
    return AlphaZeroCpp.GoPosition7.restore(
        [black],
        [white],
        player,
        ko_point,
        consecutive_passes,
        move_number,
        rules(maximum_moves),
    )


@pytest.mark.parametrize(
    ('position_type', 'board_size', 'packed_bytes'),
    [
        (AlphaZeroCpp.GoPosition7, 7, 129),
        (AlphaZeroCpp.GoPosition9, 9, 257),
    ],
)
def test_initial_position_actions_and_encoding(
    position_type: type[AlphaZeroCpp.GoPosition7] | type[AlphaZeroCpp.GoPosition9],
    board_size: int,
    packed_bytes: int,
) -> None:
    position = position_type(rules(board_size * board_size * 4))

    assert position.player == AlphaZeroCpp.GoPlayer.BLACK
    assert position.black_points(0) == []
    assert position.white_points(0) == []
    assert position.legal_actions() == list(range(board_size * board_size + 1))
    assert len(position.packed_encoding()) == packed_bytes
    assert len(position.tensor_encoding()) == 17 * board_size * board_size


def test_capture_suicide_and_simple_ko() -> None:
    capture = restore_7([1, 7, 9], [8], AlphaZeroCpp.GoPlayer.BLACK, move_number=6)
    captured = capture.child(15)
    assert captured.black_points(0) == [1, 7, 9, 15]
    assert captured.white_points(0) == []
    assert capture.white_points(0) == [8]

    suicide = restore_7([1, 7, 9, 15], [], AlphaZeroCpp.GoPlayer.WHITE, move_number=4)
    assert not suicide.is_legal(8)

    before_ko = restore_7([0, 2], [1, 7, 9, 15], AlphaZeroCpp.GoPlayer.BLACK, move_number=8)
    after_ko = before_ko.child(8)
    assert after_ko.ko_point == 1
    assert not after_ko.is_legal(1)
    after_pass = after_ko.child(49)
    assert after_pass.ko_point is None
    assert after_pass.is_legal(1)


def test_history_actions_hash_and_encoding() -> None:
    initial = AlphaZeroCpp.GoPosition7(rules())
    after_black = initial.child(0)
    after_white = after_black.child(8)

    assert after_white.black_points(0) == [0]
    assert after_white.white_points(0) == [8]
    assert after_white.black_points(1) == [0]
    assert after_white.white_points(1) == []
    assert after_white.action_id(49) == 49
    assert after_white.decode_actions([0, 8, 49]) == [0, 8, 49]
    assert after_white.state_hash() == after_white.state_hash()
    assert after_white.state_hash() != after_black.state_hash()

    packed = after_white.packed_encoding()
    tensor = after_white.tensor_encoding()
    assert packed[0] & 1
    assert tensor[0] == 1
    assert tensor[49 + 8] == 1
    assert tensor[2 * 49] == 1
    assert set(tensor[16 * 49 :]) == {1}


def test_pass_termination_scoring_and_maximum_move_result() -> None:
    scoring = restore_7([1, 7], [], AlphaZeroCpp.GoPlayer.BLACK, move_number=2)
    score = scoring.area_score()
    assert (score.black_half_points, score.white_half_points) == (98, 15)
    terminal = scoring.child(49).child(49)
    assert terminal.is_terminal
    assert terminal.termination_reason == AlphaZeroCpp.GoTerminationReason.TWO_PASSES
    assert terminal.terminal_result().winner == AlphaZeroCpp.GoPlayer.BLACK
    assert terminal.terminal_value() == 1.0

    capped = restore_7(
        [],
        [],
        AlphaZeroCpp.GoPlayer.WHITE,
        move_number=49,
        maximum_moves=49,
    )
    assert capped.termination_reason == AlphaZeroCpp.GoTerminationReason.MAXIMUM_MOVES
    assert capped.terminal_result().score is None
    assert capped.terminal_value() is None
    assert capped.legal_actions() == []


def test_action_symmetries_are_invertible() -> None:
    symmetries = [
        AlphaZeroCpp.GoSymmetry.IDENTITY,
        AlphaZeroCpp.GoSymmetry.ROTATE_90,
        AlphaZeroCpp.GoSymmetry.ROTATE_180,
        AlphaZeroCpp.GoSymmetry.ROTATE_270,
        AlphaZeroCpp.GoSymmetry.REFLECT,
        AlphaZeroCpp.GoSymmetry.REFLECT_ROTATE_90,
        AlphaZeroCpp.GoSymmetry.REFLECT_ROTATE_180,
        AlphaZeroCpp.GoSymmetry.REFLECT_ROTATE_270,
    ]
    for symmetry in symmetries:
        inverse = AlphaZeroCpp.GoPosition7.inverse_symmetry(symmetry)
        for action in range(50):
            transformed = AlphaZeroCpp.GoPosition7.transform_action(action, symmetry)
            assert AlphaZeroCpp.GoPosition7.transform_action(transformed, inverse) == action
