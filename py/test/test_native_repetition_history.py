from __future__ import annotations

from collections.abc import Callable

import pytest

AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

if not hasattr(AlphaZeroCpp.MCTSNode, 'repetition_count'):
    pytest.skip('AlphaZeroCpp must be rebuilt before native history tests run.', allow_module_level=True)

STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
KNIGHT_CYCLE = (
    'g1f3',
    'g8f6',
    'f3g1',
    'f6g8',
)
EIGHT_PLY_CYCLE = (
    'g1f3',
    'g8f6',
    'b1c3',
    'b8c6',
    'f3g1',
    'f6g8',
    'c3b1',
    'c6b8',
)


@pytest.mark.parametrize(
    'root_factory',
    (AlphaZeroCpp.new_root_with_history, AlphaZeroCpp.new_eval_root_with_history),
)
def test_native_threefold_ends_on_third_occurrence(
    root_factory: Callable[[str, tuple[str, ...]], object],
) -> None:
    second_occurrence = root_factory(STARTING_FEN, KNIGHT_CYCLE)
    third_occurrence = root_factory(STARTING_FEN, KNIGHT_CYCLE * 2)

    assert second_occurrence.repetition_count == 1
    assert not second_occurrence.is_terminal
    assert third_occurrence.repetition_count == 2
    assert third_occurrence.is_terminal


def test_native_history_supports_longer_cycles() -> None:
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, EIGHT_PLY_CYCLE * 2)

    assert root.repetition_count == 2
    assert root.is_terminal


def test_native_irreversible_move_resets_history() -> None:
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, KNIGHT_CYCLE + ('e2e4',))

    assert root.repetition_count == 0
    assert not root.is_terminal


def test_native_history_is_bounded_to_fifty_move_window() -> None:
    reversible_moves = EIGHT_PLY_CYCLE * 13
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, reversible_moves)

    assert root.is_terminal
