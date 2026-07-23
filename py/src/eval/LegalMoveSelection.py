from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class LegalMoveSelection:
    move: chess.Move
    candidate_rank: int | None


def select_legal_analysis_move(
    board: chess.Board,
    chosen_move_uci: str,
    ordered_candidates_uci: tuple[str, ...],
) -> LegalMoveSelection:
    candidates_uci = (
        chosen_move_uci,
        *(move_uci for move_uci in ordered_candidates_uci if move_uci != chosen_move_uci),
    )
    for candidate_rank, move_uci in enumerate(candidates_uci):
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            continue
        if move in board.legal_moves:
            return LegalMoveSelection(move=move, candidate_rank=candidate_rank)

    legal_moves = sorted(board.legal_moves, key=lambda legal_move: legal_move.uci())
    if not legal_moves:
        raise ValueError('Cannot select a move from a position without legal moves.')
    return LegalMoveSelection(move=legal_moves[0], candidate_rank=None)
