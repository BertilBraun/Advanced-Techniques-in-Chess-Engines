from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from src.games.Game import Game
from src.games.chess.ChessBoard import ChessBoard, ChessMove

BOARD_LENGTH = 8
BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH
ENCODING_CHANNELS = 12 + 1 + 1 + 1  # 12 for pieces + 1 for castling rights + 1 for color + 1 for en passant square


def square_to_index(square: int) -> tuple[int, int]:
    row, col = divmod(square, BOARD_LENGTH)
    return row, col


def index_to_square(row: int, col: int) -> int:
    return row * BOARD_LENGTH + col


class DictMove(NamedTuple):
    from_square: int
    to_square: int
    promotion: chess.PieceType | None


def _build_action_dicts() -> tuple[dict[DictMove, int], dict[int, DictMove]]:
    """We need to define a efficient dense representation for the chess Moves.
    In principal I have the following move options:
    from each square:
    - walk in the 8 queen directions (I think 28 moves in total)
    - walk the knight moves (up to 8 per pos, can be less)

    For the second and second last row, we also have (8x3-2)*4 many promotion moves. Pawns can move forward or capture to the right/left but not at the boarders (8x3-2) and we have four different promotion pieces.

    I want to precalculate a dict mapping which I can utilize to encode all moves into a dense representation, i.e. each value from 0..ACTION_SIZE represents one of these moves, and we need to be able to decode them again."""

    DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    moves: list[DictMove] = []
    for from_square in range(BOARD_SIZE):
        row, col = square_to_index(from_square)

        # Queen moves
        for dr, dc in DIRECTIONS:
            for distance in range(1, 8):
                to_row, to_col = row + dr * distance, col + dc * distance
                if 0 <= to_row < BOARD_LENGTH and 0 <= to_col < BOARD_LENGTH:
                    moves.append(DictMove(from_square, index_to_square(to_row, to_col), None))

        # Knight moves
        for dr, dc in KNIGHT_MOVES:
            to_row, to_col = row + dr, col + dc
            if 0 <= to_row < BOARD_LENGTH and 0 <= to_col < BOARD_LENGTH:
                moves.append(DictMove(from_square, index_to_square(to_row, to_col), None))

        # Promotion moves
        if row in (1, 6):
            for dc in (-1, 0, 1):
                to_row, to_col = row + (-1 if row == 1 else 1), col + dc
                if 0 <= to_row < BOARD_LENGTH and 0 <= to_col < BOARD_LENGTH:
                    for promotion_piece in PROMOTION_PIECES:
                        moves.append(DictMove(from_square, index_to_square(to_row, to_col), promotion_piece))

    move2index = {move: i for i, move in enumerate(moves)}
    index2move = {i: move for i, move in enumerate(moves)}
    return move2index, index2move


_BIT_MASK = 1 << np.arange(64, dtype=np.uint64)  # use uint64 to prevent overflow


def _bitfield_to_board_state(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.int8]:
    """Convert a tuple of integers into a binary state. Each integer represents a channel of the state. This assumes that the state is a binary state."""
    assert state.dtype == np.uint64, 'The state must be encoded as uint64 to prevent overflow'

    encoded_array = state.reshape(-1, 1)  # shape: (channels, 1)

    # Extract bits for each channel
    bits = ((encoded_array & _BIT_MASK) > 0).astype(np.int8)

    return bits.reshape(ChessGame().representation_shape)


class ChessGame(Game[ChessMove]):
    move2index, index2move = _build_action_dicts()

    @property
    def action_size(self) -> int:
        return len(self.move2index)

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return (ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH)

    def get_canonical_board(self, board: ChessBoard) -> np.ndarray:
        colors = (chess.WHITE, chess.BLACK)
        encoded_pieces: list[int] = [
            piece
            for co in colors
            for piece in (
                board.board.pawns & board.board.occupied_co[co],
                board.board.knights & board.board.occupied_co[co],
                board.board.bishops & board.board.occupied_co[co],
                board.board.rooks & board.board.occupied_co[co],
                board.board.queens & board.board.occupied_co[co],
                board.board.kings & board.board.occupied_co[co],
            )
        ]

        castling_rights = [
            right
            for co in colors
            for right in (
                board.board.has_kingside_castling_rights(co),
                board.board.has_queenside_castling_rights(co),
            )
        ]

        encoded_castling_rights = (
            castling_rights[0]
            + (castling_rights[1] << (BOARD_LENGTH - 1))
            + (castling_rights[2] << (BOARD_SIZE - BOARD_LENGTH))
            + (castling_rights[3] << (BOARD_SIZE - 1))
        )
        color = 0xFFFFFFFFFFFFFFFF if board.current_player == 1 else 0
        ep_square = (1 << board.board.ep_square) if board.board.ep_square else 0

        canonical_board = _bitfield_to_board_state(
            np.array(encoded_pieces + [encoded_castling_rights, ep_square, color], dtype=np.uint64)
        )

        return canonical_board

    def decode_canonical_board(self, canonical_board: np.ndarray) -> ChessBoard:
        board = self.get_initial_board()
        board.board.clear()

        if canonical_board[13, 0, 0] == 0:
            canonical_board = np.flip(canonical_board, axis=1)

        colors = (chess.WHITE, chess.BLACK) if canonical_board[13, 0, 0] == 1 else (chess.BLACK, chess.WHITE)

        for i, color in enumerate(colors):
            for j, piece in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)):
                for row in range(8):
                    for col in range(8):
                        if canonical_board[i * 6 + j, row, col] == 1:
                            board.board.set_piece_at(chess.square(col, row), chess.Piece(piece, color))

        return board

    def encode_move(self, move: ChessMove) -> int:
        return self.move2index[DictMove(move.from_square, move.to_square, move.promotion)]

    def decode_move(self, move: int) -> ChessMove:
        dict_move = self.index2move[move]
        return ChessMove(dict_move.from_square, dict_move.to_square, dict_move.promotion)

    def symmetric_variations(
        self, board: np.ndarray, visit_counts: list[tuple[int, int]]
    ) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
        return [
            # Original board
            (board, visit_counts),
            # Vertical flip
            # 1234 -> becomes -> 4321
            # 5678               8765
            # (np.flip(board, axis=2), self._flip_action_probs(visit_counts)),
        ]

    def _flip_action_probs(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        flipped_visit_counts: list[tuple[int, int]] = []
        for move, count in visit_counts:
            flipped_move = self.decode_move(move)
            move_from_row, move_from_col = square_to_index(flipped_move.from_square)
            move_to_row, move_to_col = square_to_index(flipped_move.to_square)
            flipped_move_from = index_to_square(move_from_row, BOARD_LENGTH - 1 - move_from_col)
            flipped_move_to = index_to_square(move_to_row, BOARD_LENGTH - 1 - move_to_col)
            flipped_index = self.encode_move(
                chess.Move(flipped_move_from, flipped_move_to, promotion=flipped_move.promotion)
            )
            flipped_visit_counts.append((flipped_index, count))
        return flipped_visit_counts

    def get_initial_board(self) -> ChessBoard:
        return ChessBoard()
