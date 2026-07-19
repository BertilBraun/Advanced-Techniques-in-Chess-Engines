from __future__ import annotations

from dataclasses import dataclass

import chess

REPETITION_HISTORY_PLIES = 8


@dataclass(frozen=True)
class RepetitionHistory:
    starting_fen: str
    moves_uci: tuple[str, ...]


def bounded_repetition_history(
    board: chess.Board,
    maximum_plies: int,
) -> RepetitionHistory:
    if maximum_plies < 1:
        raise ValueError('maximum_plies must be positive.')

    board_with_history = board.copy(stack=maximum_plies)
    recent_moves = tuple(board_with_history.move_stack[-maximum_plies:])
    for _ in recent_moves:
        board_with_history.pop()

    return RepetitionHistory(
        starting_fen=board_with_history.fen(),
        moves_uci=tuple(move.uci() for move in recent_moves),
    )
