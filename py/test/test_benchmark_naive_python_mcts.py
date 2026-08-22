from __future__ import annotations

import chess
import pytest

pytest.importorskip('AlphaZeroCpp')
from tools.benchmark_naive_python_mcts import SearchNode, backup, select_child, terminal_value


def test_select_child_uses_parent_perspective_value_and_exploration() -> None:
    parent = SearchNode(prior=1.0, visits=16)
    losing_for_child = SearchNode(prior=0.1, visits=4, value_sum=-3.0)
    unexplored = SearchNode(prior=0.9)
    move_a = chess.Move.from_uci('e2e4')
    move_b = chess.Move.from_uci('d2d4')
    parent.children = {move_a: losing_for_child, move_b: unexplored}

    move, child = select_child(parent, exploration_constant=1.0)

    assert move == move_b
    assert child is unexplored


def test_backup_alternates_leaf_value_between_players() -> None:
    root = SearchNode(prior=1.0)
    child = SearchNode(prior=0.5)
    leaf = SearchNode(prior=0.5)

    backup([root, child, leaf], leaf_value=0.75)

    assert (root.visits, root.value_sum) == (1, 0.75)
    assert (child.visits, child.value_sum) == (1, -0.75)
    assert (leaf.visits, leaf.value_sum) == (1, 0.75)


def test_terminal_value_uses_player_to_move_perspective() -> None:
    white_win = chess.Outcome(chess.Termination.CHECKMATE, chess.WHITE)
    draw = chess.Outcome(chess.Termination.STALEMATE, None)

    assert terminal_value(white_win, chess.WHITE) == 1.0
    assert terminal_value(white_win, chess.BLACK) == -1.0
    assert terminal_value(draw, chess.WHITE) == 0.0
