from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from src.games.Game import Game
from src.games.chess.ChessBoard import ChessBoard, ChessMove

BOARD_LENGTH = 8
BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH
# 12 piece types, 4 castling rights, 2 player pieces, 1 checkers, 6 material difference channels
ENCODING_CHANNELS = 12 + 4 + 2 + 1 + 6
BINARY_CHANNELS = tuple(range(12 + 4 + 2 + 1))  # All channels are binary except the material difference channels
SCALAR_CHANNELS = tuple(range(max(BINARY_CHANNELS) + 1, ENCODING_CHANNELS))  # Material difference channels


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
    Note: given the current board encoding, we always present the current player's perspective as white, so we only need to encode the white promotion moves. The black moves are always mirrored to the equivalent white moves before.
    Note: given the goal of this project to achieve a high amateur level of play, we can ignore the possibility of promoting to a knight, rook or bishop, as these moves are extremely rare in practice. We can just promote to a queen.
    Note: these reduce the number of moves to encode by 154 moves in total, which is a significant reduction.

    I want to precalculate a dict mapping which I can utilize to encode all moves into a dense representation, i.e. each value from 0..ACTION_SIZE represents one of these moves, and we need to be able to decode them again."""

    DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    PROMOTION_PIECES = [chess.QUEEN]
    # Unless for really professional play, these promotion pieces never come to play: [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

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
        # Note: we dont need blacks promotion moves anymore, as black moves are always mirrored to the equivalent white moves before (1, 6):
        if row in (6,):
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

    return bits.reshape((BINARY_CHANNELS[-1] + 1, BOARD_LENGTH, BOARD_LENGTH))


class ChessGame(Game[ChessMove]):
    move2index, index2move = _build_action_dicts()

    @property
    def action_size(self) -> int:
        return len(self.move2index)

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return (ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH)

    @property
    def binary_channels(self) -> tuple[int, ...]:
        """Returns which channels of the board state are binary."""
        return BINARY_CHANNELS

    @property
    def scalar_channels(self) -> tuple[int, ...]:
        """Returns which channels of the board state are scalar."""
        return SCALAR_CHANNELS

    def get_canonical_board(self, board: ChessBoard) -> np.ndarray:  # type: ignore[override]
        """Returns a canonical representation of the board from the perspective of the white player."""

        # 1. If Black to move, mirror first
        if board.board.turn == chess.BLACK:
            tmp = board.board.mirror()
        else:
            tmp = board.board

        # 2. Encode, but always assume tmp.turn == WHITE
        colors = (chess.WHITE, chess.BLACK)
        encoded_pieces = [
            piece
            for co in colors
            for piece in (
                tmp.pawns & tmp.occupied_co[co],
                tmp.knights & tmp.occupied_co[co],
                tmp.bishops & tmp.occupied_co[co],
                tmp.rooks & tmp.occupied_co[co],
                tmp.queens & tmp.occupied_co[co],
                tmp.kings & tmp.occupied_co[co],
            )
        ]

        castling_bits = [
            right * 0xFFFF_FFFF_FFFF_FFFF
            for co in colors
            for right in (tmp.has_kingside_castling_rights(co), tmp.has_queenside_castling_rights(co))
        ]

        # ep_bit = (1 << tmp.ep_square) if tmp.ep_square else 0 # EP bit is so rarely used that we can just ignore it for now
        # based on: https://arxiv.org/html/2304.14918v2
        p1_pieces = tmp.occupied_co[chess.WHITE]
        p2_pieces = tmp.occupied_co[chess.BLACK]
        material_difference = [
            chess.popcount(p1_bb) - chess.popcount(p2_bb)
            for p1_bb, p2_bb in zip(encoded_pieces[:6], encoded_pieces[6:])
        ]
        checkers = tmp.checkers().mask

        state = np.array(encoded_pieces + castling_bits + [p1_pieces, p2_pieces, checkers], dtype=np.uint64)
        binary_state = _bitfield_to_board_state(state)
        # scalar state should be a array of BOARD_LENGTH x BOARD_LENGTH with the material difference for each square
        scalar_state = np.zeros((len(SCALAR_CHANNELS), BOARD_LENGTH, BOARD_LENGTH), dtype=np.int8)
        for i, diff in enumerate(material_difference):
            scalar_state[i, :, :] = diff
        state = np.concatenate((binary_state, scalar_state), axis=0)
        return state  # shape: (ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH)

    def encode_move(self, move: ChessMove, board: ChessBoard) -> int:  # type: ignore[override]
        """Encodes a chess move into an integer index based on the predefined move2index mapping.
        Notably:
        - Black moves are mirrored to the equivalent white moves before encoding. This is because we always represent the current player's perspective as white.
        - Castling moves are handled separately, because the Cpp-Stockfish engine encodes castling moves differently than python-chess does. This function converts the castling move to the format used by stockfish before encoding.
        - Only queen promotions are supported in this encoding, as the level of play we are targeting does not require knight, rook, or bishop promotions.
        """
        assert move.promotion in (None, chess.QUEEN), 'Only queen promotions are supported in this encoding.'

        if board.board.turn == chess.BLACK:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )

        # Handle castling moves
        if _is_castling_move(move, board):
            assert move.promotion is None, 'Castling moves should not have a promotion.'
            # In python-chess, castling moves are represented as king moving two squares
            # Need to convert to our representation format before lookup

            king_square = move.from_square

            # Determine if it's kingside or queenside castling
            if move.to_square > king_square:  # Kingside
                rook_square = chess.H1  # only ever white side castling is relevant
            else:  # Queenside
                rook_square = chess.A1

            move = chess.Move(king_square, rook_square)

        # Handle non-castling moves
        return self.move2index[DictMove(move.from_square, move.to_square, move.promotion)]

    def decode_move(self, idx: int, board: ChessBoard) -> ChessMove:  # type: ignore[override]
        """Decodes an integer index back to a chess move. See `encode_move` for details on how moves are encoded."""

        m = self.index2move[idx]

        move = chess.Move(m.from_square, m.to_square, promotion=m.promotion)

        # Check if move looks like a castling pattern
        if _is_castling_move(move, board):
            assert move.promotion is None, 'Castling moves should not have a promotion.'

            king_square = move.from_square

            # It's a castling move, compute the right destination square for the king
            if move.to_square > king_square:  # Kingside
                king_dest = chess.G1  # only ever white side castling is relevant
            else:  # Queenside
                king_dest = chess.C1

            move = chess.Move(king_square, king_dest)

        # Handle regular moves
        if board.board.turn == chess.BLACK:
            # Mirror the move for black
            move = _mirror_move_for_black(move)

        return move

    def symmetric_variations(  # type: ignore[override]
        self, board: ChessBoard, visit_counts: list[tuple[int, int]]
    ) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
        encoded_board = self.get_canonical_board(board)

        def mirror_left_right(square: int) -> int:
            row, col = square_to_index(square)
            mirrored_col = BOARD_LENGTH - 1 - col  # flip column, keep row
            return index_to_square(row, mirrored_col)

        def mirror_move(encoded_move: int) -> int:
            move = self.decode_move(encoded_move, board)
            mirrored_move = chess.Move(
                mirror_left_right(move.from_square),
                mirror_left_right(move.to_square),
                promotion=move.promotion,
            )
            return self.encode_move(mirrored_move, board)

        return [
            # Original board
            (encoded_board, visit_counts),
            # Mirrored board around the horizontal axis (i.e. left-right mirroring)
            (
                np.flip(encoded_board, axis=2),
                [(mirror_move(move), count) for move, count in visit_counts],
            ),
        ]

    def get_initial_board(self) -> ChessBoard:
        return ChessBoard()


def _is_castling_move(move: ChessMove, board: ChessBoard) -> bool:
    """Checks if a move is a castling move."""
    from_piece = board.board.piece_at(move.from_square)
    to_piece = board.board.piece_at(move.to_square)
    return bool(
        from_piece
        and to_piece
        and from_piece.piece_type == chess.KING
        and to_piece.piece_type == chess.ROOK
        and from_piece.color == to_piece.color
        and move.from_square == chess.E1
        and move.to_square in (chess.G1, chess.C1)
    )


def _mirror_move_for_black(move: ChessMove) -> ChessMove:
    """Mirrors a move for the black player."""
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return chess.Move(from_square, to_square, promotion=move.promotion)


_REPRESENTATION_SHAPE = ChessGame().representation_shape
