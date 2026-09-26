"""Builds and verifies the permutation between Lc0's 1858 policy indices and this project's 1880.

The mapping is derived empirically rather than assumed: positions are walked, and for every legal
move the project's own action id is compared against Lc0's index for the same move in the canonical
(side-to-move) frame. A move that ever maps two different ways is a hard failure, because the whole
teacher experiment rests on this table being exact.

Requires the native extension, so it runs on the node.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.games.chess.contract import CHESS_ACTION_SIZE
from src.util.atomic_file import write_text_atomically

LC0_POLICY_SIZE = 1858
UCI_PATTERN = re.compile(r'"([a-h][1-8][a-h][1-8][qrbn]?)"')


@dataclass(frozen=True)
class PolicyMap:
    # Indexed by this project's action id. Several actions may share one Lc0 index: Lc0 spells a knight
    # promotion as the bare move, so e7e8n and a piece moving e7e8 both read index e7e8. Only one of them
    # can be legal in any position, so gathering the same logit into both is exact.
    action_id_to_lc0_index: tuple[int, ...]
    covered_lc0_indices: int
    positions_walked: int


def parse_lc0_move_table(source: Path) -> tuple[str, ...]:
    """Extracts kIdxToMove from Lc0's bitboard.cc in declaration order."""
    moves = UCI_PATTERN.findall(source.read_text(encoding='utf-8'))
    if len(moves) < LC0_POLICY_SIZE:
        raise SystemExit(f'Found only {len(moves)} UCI strings in {source}; expected {LC0_POLICY_SIZE}.')
    table = tuple(moves[:LC0_POLICY_SIZE])
    if len(set(table)) != LC0_POLICY_SIZE:
        raise SystemExit('Lc0 move table contains duplicates; the parse picked up unrelated strings.')
    return table


def flip_uci_ranks(move_uci: str) -> str:
    """Mirrors a move into the side-to-move frame, which is how both engines index policy."""
    flipped = ''.join(str(9 - int(character)) if character.isdigit() else character for character in move_uci[:4])
    return flipped + move_uci[4:]


def lc0_move_notation(fen: str, move_uci: str) -> str:
    """Rewrites a UCI move into the spelling Lc0's policy table uses.

    Lc0 encodes castling as the king capturing its own rook (e1h1, not e1g1) and a knight promotion as
    the bare move (a7a8, not a7a8n); queen, rook and bishop promotions keep their suffix.
    """
    import chess

    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if board.is_castling(move):
        rook_file = 'h' if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 'a'
        return move_uci[:2] + rook_file + move_uci[1]
    if move.promotion == chess.KNIGHT:
        return move_uci[:4]
    return move_uci


def promotion_positions() -> tuple[str, ...]:
    """Positions where every straight and capturing promotion is legal; random games rarely reach them."""
    import chess

    fens: list[str] = []
    for file in range(8):
        for target_file in (file - 1, file, file + 1):
            if not 0 <= target_file <= 7:
                continue
            for colour in (chess.WHITE, chess.BLACK):
                board = chess.Board(None)
                from_rank, to_rank = (6, 7) if colour == chess.WHITE else (1, 0)
                board.set_piece_at(chess.square(file, from_rank), chess.Piece(chess.PAWN, colour))
                if target_file != file:
                    board.set_piece_at(chess.square(target_file, to_rank), chess.Piece(chess.ROOK, not colour))
                board.turn = colour
                for own_king, their_king in ((chess.E3, chess.E5), (chess.A4, chess.H4), (chess.H3, chess.A5)):
                    trial = board.copy()
                    if colour == chess.BLACK:
                        own_king, their_king = chess.square_mirror(own_king), chess.square_mirror(their_king)
                    if trial.piece_at(own_king) or trial.piece_at(their_king):
                        continue
                    trial.set_piece_at(own_king, chess.Piece(chess.KING, colour))
                    trial.set_piece_at(their_king, chess.Piece(chess.KING, not colour))
                    promotions = [move for move in trial.legal_moves if move.promotion]
                    if trial.is_valid() and len(promotions) >= 4:
                        fens.append(trial.fen())
                        break
    return tuple(fens)


def build(move_table: tuple[str, ...], position_count: int, seed: int) -> PolicyMap:
    import AlphaZeroCpp

    lc0_index_of = {move: index for index, move in enumerate(move_table)}
    mapping: dict[int, int] = {}
    positions_walked = 0

    def record(position: AlphaZeroCpp.ChessPosition) -> None:
        fen = position.fen
        for action_id in position.legal_actions():
            move_uci = lc0_move_notation(fen, position.action_uci(action_id))
            canonical = move_uci if position.current_player == 1 else flip_uci_ranks(move_uci)
            lc0_index = lc0_index_of.get(canonical)
            if lc0_index is None:
                raise SystemExit(f'Legal move {move_uci} (canonical {canonical}) is absent from the Lc0 table.')
            previous = mapping.get(action_id)
            if previous is not None and previous != lc0_index:
                raise SystemExit(
                    f'Action {action_id} maps to Lc0 index {previous} and {lc0_index} ({canonical}); '
                    'the project encoding is not canonical for this move.'
                )
            mapping[action_id] = lc0_index

    generator = random.Random(seed)
    for _ in range(position_count):
        position = AlphaZeroCpp.ChessPosition()
        for _ in range(generator.randint(0, 160)):
            if position.is_terminal:
                break
            positions_walked += 1
            record(position)
            position = position.child(generator.choice(position.legal_actions()))
    for fen in promotion_positions():
        positions_walked += 1
        record(AlphaZeroCpp.ChessPosition(fen))

    table = tuple(mapping.get(action_id, -1) for action_id in range(CHESS_ACTION_SIZE))
    covered = len(set(mapping.values()))
    print(
        f'Walked {positions_walked} positions; {len(mapping)} of {CHESS_ACTION_SIZE} actions mapped, '
        f'covering {covered} of {LC0_POLICY_SIZE} Lc0 indices.'
    )
    return PolicyMap(action_id_to_lc0_index=table, covered_lc0_indices=covered, positions_walked=positions_walked)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lc0-bitboard-source', type=Path, required=True, help="Lc0's src/chess/bitboard.cc.")
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--position-count', type=int, default=4000, help='Random games walked to cover the table.')
    parser.add_argument('--seed', type=int, default=20260926)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    move_table = parse_lc0_move_table(arguments.lc0_bitboard_source)
    policy_map = build(move_table, arguments.position_count, arguments.seed)
    payload = {
        'lc0_policy_size': LC0_POLICY_SIZE,
        'project_action_size': CHESS_ACTION_SIZE,
        'positions_walked': policy_map.positions_walked,
        'covered_lc0_indices': policy_map.covered_lc0_indices,
        'mapped_action_ids': sum(1 for entry in policy_map.action_id_to_lc0_index if entry >= 0),
        'action_id_to_lc0_index': list(policy_map.action_id_to_lc0_index),
    }
    write_text_atomically(arguments.output, json.dumps(payload, indent=2) + '\n')
    print(f'Wrote {arguments.output}')


if __name__ == '__main__':
    main()
