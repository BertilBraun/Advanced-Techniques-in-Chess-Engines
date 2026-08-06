from __future__ import annotations

import pytest

AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

from src.games.go.GoVisuals import GoVisuals
from src.games.go.NativeGoBoard import NativeGoBoard


class RecordingGUI:
    def __init__(self) -> None:
        self.circles: list[tuple[int, int, str]] = []

    def draw_circle(self, row: int, column: int, color: str) -> None:
        self.circles.append((row, column, color))


@pytest.mark.parametrize('board_size', [7, 9])
def test_go_visuals_map_clicks_and_stones(board_size: int) -> None:
    board = NativeGoBoard(board_size)
    visuals = GoVisuals()

    action = visuals.try_make_move(board, None, (1, 1))
    assert action == board_size + 1
    assert not visuals.is_two_click_game()
    assert visuals.get_moves_from_square(board, 1, 1) == []

    assert action is not None
    board.make_move(action)
    gui = RecordingGUI()
    visuals.draw_pieces(board, gui)  # type: ignore[arg-type]
    assert gui.circles == [(1, 1, 'black')]
    assert visuals.try_make_move(board, None, (1, 1)) is None


def test_native_go_visual_board_copy_and_pass_termination() -> None:
    board = NativeGoBoard(7)
    initial = board.copy()

    board.make_move(board.pass_action)
    assert board.consecutive_passes == 1
    assert initial.consecutive_passes == 0
    board.make_move(board.pass_action)

    assert board.is_game_over()
    assert board.get_valid_moves() == []
    assert board.quick_hash() != initial.quick_hash()


def test_pass_remains_legal_when_no_placement_is_legal() -> None:
    position = AlphaZeroCpp.GoPosition7.restore(
        [list(range(0, 49, 2))],
        [list(range(1, 49, 2))],
        AlphaZeroCpp.GoPlayer.BLACK,
        None,
        0,
        0,
        AlphaZeroCpp.GoRules(15, 196),
    )
    board = NativeGoBoard.from_position(position)

    assert not board.is_game_over()
    assert board.get_valid_moves() == [board.pass_action]
    board.make_move(board.pass_action)
    assert not board.is_game_over()
    board.make_move(board.pass_action)
    assert board.is_game_over()
