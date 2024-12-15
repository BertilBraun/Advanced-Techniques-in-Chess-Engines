from __future__ import annotations
from enum import Enum
from typing import Optional, List, Tuple

import numpy as np
from numba import njit, uint8
from numba.experimental import jitclass

from AIZeroConnect4Bot.src.games.Game import Board, Player

CheckersMove = Tuple[int, int]
_ValidCheckersMovesList = List[Tuple[int, int, List[int]]]  # (start, end, captures)


@jitclass(
    [
        ('starts', uint8[:]),
        ('ends', uint8[:]),
        ('captures', uint8[:, :]),
        ('length', uint8),
        ('num_captures', uint8),
    ]
)
class _CheckersMoves:
    def __init__(self):
        self.starts = np.zeros((36,), dtype=np.uint8)
        self.ends = np.zeros((36,), dtype=np.uint8)
        self.captures = np.zeros((36, 13), dtype=np.uint8)
        self.length = 0
        self.num_captures = 0

    def append(self, start: int, end: int) -> int:
        assert self.length < 36, 'Too many moves'
        self.starts[self.length] = start
        self.ends[self.length] = end
        self.captures[self.length, 0] = 0
        self.length += 1
        return self.length - 1

    def add_capture(self, index: int, capture: int) -> None:
        assert 0 <= index < self.length, 'Invalid index'
        assert self.captures[index, 0] < 13, 'Too many captures'
        self.captures[index, self.captures[index, 0]] = capture
        self.captures[index, 0] += 1
        if self.captures[index, 0] == 1:
            self.num_captures += 1

    def clear(self) -> None:
        self.length = 0
        self.num_captures = 0

    def copy(self) -> _CheckersMoves:
        moves = _CheckersMoves()
        moves.starts = self.starts.copy()
        moves.ends = self.ends.copy()
        moves.captures = self.captures.copy()
        moves.length = self.length
        moves.num_captures = self.num_captures
        return moves

    def pop(self) -> bool:
        if self.length == 0:
            return False
        if self.captures[self.length - 1, 0] > 0:
            self.num_captures -= 1
        self.length -= 1
        return True


BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# File masks to prevent wrapping
A_FILE = np.uint64(0x0101010101010101)  # leftmost column bits set
H_FILE = np.uint64(0x8080808080808080)  # rightmost column bits set

# Mask for 64-bit
FULL_64 = np.uint64((1 << 64) - 1)

# Starting positions (bitboards)
# Black piece:
BLACK_MEN_START = np.uint64(
    (1 << 1)
    | (1 << 3)
    | (1 << 5)
    | (1 << 7)
    | (1 << 8)
    | (1 << 10)
    | (1 << 12)
    | (1 << 14)
    | (1 << 17)
    | (1 << 19)
    | (1 << 21)
    | (1 << 23)
)

# White piece:
WHITE_MEN_START = np.uint64(
    (1 << 40)
    | (1 << 42)
    | (1 << 44)
    | (1 << 46)
    | (1 << 49)
    | (1 << 51)
    | (1 << 53)
    | (1 << 55)
    | (1 << 56)
    | (1 << 58)
    | (1 << 60)
    | (1 << 62)
)

# Promotion rows:
BLACK_PROMOTION_ROW = np.uint64(0xFF00000000000000)  # top row (row 7 in index terms, bits 56-63)
WHITE_PROMOTION_ROW = np.uint64(0x00000000000000FF)  # bottom row (row 0 in index terms, bits 0-7)


DR_SHIFT = -(BOARD_SIZE + 1)
DL_SHIFT = -(BOARD_SIZE - 1)
UR_SHIFT = BOARD_SIZE - 1
UL_SHIFT = BOARD_SIZE + 1


@njit
def _bit_set(bitboard: np.uint64, index: int) -> bool:
    return (bitboard & (np.uint64(1) << np.uint64(index))) != 0


@njit
def _down_right(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~H_FILE) << np.uint64(BOARD_SIZE + 1))


@njit
def _down_left(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~A_FILE) << np.uint64(BOARD_SIZE - 1))


@njit
def _up_right(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~H_FILE) >> np.uint64(BOARD_SIZE - 1))


@njit
def _up_left(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~A_FILE) >> np.uint64(BOARD_SIZE + 1))


@njit
def _least_significant_bit(bitboard: np.uint64) -> int:
    int_bb = int(bitboard)
    b = 0
    while int_bb:
        b += 1
        int_bb >>= 1
    return b - 1
    # return (int_bb & -int_bb).bit_length() - 1


@njit
def _bitboard_to_list(bitboard: np.uint64) -> List[int]:
    indices = []
    working = bitboard
    while working:
        index = _least_significant_bit(working)
        indices.append(index)
        working &= ~(np.uint64(1) << np.uint64(index))
    return indices


@njit
def _decode_moves(starts: np.uint64, ends: np.uint64, shift: int, moves: _CheckersMoves):
    for end_square in _bitboard_to_list(ends):
        start_square = end_square + shift
        if 0 <= start_square < BOARD_SQUARES and _bit_set(starts, start_square):
            moves.append(start_square, end_square)


@njit
def _mask_from_shift(mask: np.uint64, shift: int) -> np.uint64:
    if shift == DL_SHIFT:
        return _down_left(mask)
    elif shift == DR_SHIFT:
        return _down_right(mask)
    elif shift == UL_SHIFT:
        return _up_left(mask)
    elif shift == UR_SHIFT:
        return _up_right(mask)
    else:
        raise ValueError('Invalid shift')


@njit
def _piece_normal_moves(pieces: np.uint64, empty: np.uint64, shift: int, moves: _CheckersMoves):
    piece_moves = _mask_from_shift(pieces, shift) & empty
    _decode_moves(pieces, piece_moves, shift, moves)


@njit
def _black_piece_normal_moves(pieces: np.uint64, empty: np.uint64, moves: _CheckersMoves):
    _piece_normal_moves(pieces, empty, DL_SHIFT, moves)
    _piece_normal_moves(pieces, empty, DR_SHIFT, moves)


@njit
def _white_piece_normal_moves(pieces: np.uint64, empty: np.uint64, moves: _CheckersMoves):
    _piece_normal_moves(pieces, empty, UL_SHIFT, moves)
    _piece_normal_moves(pieces, empty, UR_SHIFT, moves)


@njit
def _piece_single_captures(pieces: np.uint64, enemy: np.uint64, empty: np.uint64, shift: int, moves: _CheckersMoves):
    step = _mask_from_shift(pieces, shift) & enemy
    cap = _mask_from_shift(step, shift) & empty

    previous_moves = moves.length

    _decode_moves(pieces, cap, 2 * shift, moves)

    for i in range(previous_moves, moves.length):
        moves.add_capture(i, moves.starts[i] - shift)


@njit
def _black_piece_single_captures(pieces: np.uint64, enemy: np.uint64, empty: np.uint64, moves: _CheckersMoves):
    _piece_single_captures(pieces, enemy, empty, DL_SHIFT, moves)
    _piece_single_captures(pieces, enemy, empty, DR_SHIFT, moves)


@njit
def _white_piece_single_captures(pieces: np.uint64, enemy: np.uint64, empty: np.uint64, moves: _CheckersMoves):
    _piece_single_captures(pieces, enemy, empty, UL_SHIFT, moves)
    _piece_single_captures(pieces, enemy, empty, UR_SHIFT, moves)


DIRECTIONS = ((DR_SHIFT, (1, 1)), (DL_SHIFT, (-1, 1)), (UR_SHIFT, (1, -1)), (UL_SHIFT, (-1, -1)))


@njit
def _king_moves(occupied: np.uint64, enemy: np.uint64, start: int, moves: _CheckersMoves):
    y, x = divmod(start, BOARD_SIZE)
    for shift, (dx, dy) in DIRECTIONS:
        pos = start
        px, py = x, y
        while True:
            pos -= shift
            px += dx
            py += dy
            if px < 0 or px >= BOARD_SIZE or py < 0 or py >= BOARD_SIZE:
                break
            if _bit_set(occupied, pos):
                if _bit_set(enemy, pos):
                    # capture
                    if 0 <= px + dx < BOARD_SIZE and 0 <= py + dy < BOARD_SIZE and not _bit_set(occupied, pos - shift):
                        occupied &= ~(np.uint64(1) << np.uint64(pos))
                        occupied &= ~(np.uint64(1) << np.uint64(start))
                        occupied |= np.uint64(1) << np.uint64(pos - shift)
                        enemy &= ~(np.uint64(1) << np.uint64(pos))
                        previous_moves = moves.length
                        _king_moves(occupied, enemy, pos - shift, moves)
                        occupied |= np.uint64(1) << np.uint64(start)
                        occupied |= np.uint64(1) << np.uint64(pos)
                        occupied &= ~(np.uint64(1) << np.uint64(pos - shift))
                        enemy |= np.uint64(1) << np.uint64(pos)

                        for i in range(previous_moves, moves.length):
                            moves.add_capture(i, pos)
                            moves.starts[i] = start

                        if moves.length == previous_moves:
                            move_index = moves.append(start, pos - shift)
                            moves.add_capture(move_index, pos)
                # else: blocked by a piece
                break
            # empty square
            moves.append(start, pos)


@njit
def _check_further_capture(
    start: int,
    end: int,
    shift: int,
    dx: int,
    dy: int,
    enemy: np.uint64,
    empty: np.uint64,
    moves: _CheckersMoves,
) -> int:
    ey, ex = divmod(end, BOARD_SIZE)
    if (
        ey + 2 * dy >= 0  # in bounds
        and ey + 2 * dy < BOARD_SIZE  # in bounds
        and ex + 2 * dx >= 0  # in bounds
        and ex + 2 * dx < BOARD_SIZE  # in bounds
        and _bit_set(enemy, end - shift)  # next square is enemy
        and _bit_set(empty, end - 2 * shift)  # square after enemy is empty
    ):
        move_index = moves.append(start, end - 2 * shift)
        moves.add_capture(move_index, end - shift)
        return move_index
    return -1


@njit
def _expand_single_captures(
    moves: _CheckersMoves,
    enemy: np.uint64,
    empty: np.uint64,
    shift_left: int,
    shift_right: int,
    dy: int,
):
    assert moves.num_captures > 0, 'No captures to expand'
    assert moves.length == moves.num_captures, 'Not all moves are captures'

    outstanding_captures = moves.copy()
    moves.clear()

    while outstanding_captures.length > 0:
        move_index = outstanding_captures.length - 1
        start, end = outstanding_captures.starts[move_index], outstanding_captures.ends[move_index]
        captures = outstanding_captures.captures[move_index]
        outstanding_captures.pop()

        capture_right = _check_further_capture(
            start,
            end,
            shift_right,
            1,
            dy,
            enemy,
            empty,
            outstanding_captures,
        )
        capture_left = _check_further_capture(
            start,
            end,
            shift_left,
            -1,
            dy,
            enemy,
            empty,
            outstanding_captures,
        )

        if capture_right == -1 and capture_left == -1:
            # no further captures possible, add to all_captures
            move_index = moves.append(start, end)
            moves.captures[move_index] = captures
            continue

        if capture_right != -1:
            capture_right_capture = outstanding_captures.captures[capture_right][1]
            outstanding_captures.captures[capture_right] = captures
            outstanding_captures.add_capture(capture_right, capture_right_capture)
        if capture_left != -1:
            capture_left_capture = outstanding_captures.captures[capture_left][1]
            outstanding_captures.captures[capture_left] = captures
            outstanding_captures.add_capture(capture_left, capture_left_capture)


@njit
def _black_moves(pieces: np.uint64, kings: np.uint64, enemy: np.uint64, empty: np.uint64, moves: _CheckersMoves):
    _black_piece_single_captures(pieces, enemy, empty, moves)
    if moves.num_captures > 0:
        _expand_single_captures(moves, enemy, empty, DL_SHIFT, DR_SHIFT, -1)

    for start in _bitboard_to_list(kings):
        _king_moves(~empty, enemy, start, moves)

    if moves.num_captures > 0:
        return

    _black_piece_normal_moves(pieces, empty, moves)


def _white_moves(pieces: np.uint64, kings: np.uint64, enemy: np.uint64, empty: np.uint64, moves: _CheckersMoves):
    _white_piece_single_captures(pieces, enemy, empty, moves)
    if moves.num_captures > 0:
        _expand_single_captures(moves, enemy, empty, UL_SHIFT, UR_SHIFT, 1)

    for start in _bitboard_to_list(kings):
        _king_moves(~empty, enemy, start, moves)

    if moves.num_captures > 0:
        return

    _white_piece_normal_moves(pieces, empty, moves)


class Piece(Enum):
    BLACK_KING = 2
    BLACK_PIECE = 1
    WHITE_KING = -2
    WHITE_PIECE = -1


class CheckersBoard(Board[CheckersMove]):
    def __init__(self) -> None:
        super().__init__()
        self.black_pieces: np.uint64 = BLACK_MEN_START
        self.black_kings: np.uint64 = np.uint64(0)
        self.white_pieces: np.uint64 = WHITE_MEN_START
        self.white_kings: np.uint64 = np.uint64(0)

    @property
    def board_dimensions(self) -> Tuple[int, int]:
        return BOARD_SIZE, BOARD_SIZE

    def get_cell(self, row: int, col: int) -> Optional[Piece]:
        index = row * BOARD_SIZE + col
        if _bit_set(self.black_pieces, index):
            return Piece.BLACK_PIECE
        if _bit_set(self.white_pieces, index):
            return Piece.WHITE_PIECE
        if _bit_set(self.black_kings, index):
            return Piece.BLACK_KING
        if _bit_set(self.white_kings, index):
            return Piece.WHITE_KING
        return None

    def get_valid_moves(self) -> List[CheckersMove]:
        """
        Returns a list of (start, end) moves. If captures exist, return only captures.
        """
        return [(start, end) for start, end, _ in self._get_valid_moves()]

    @Board._cache()
    def _get_valid_moves(self) -> _ValidCheckersMovesList:
        pieces = self._friendly_piece()
        kings = self._friendly_kings()
        opp = self._opponent_pieces()
        empty = self._empty()

        moves = _CheckersMoves()

        if self.current_player == 1:  # Black
            _black_moves(pieces, kings, opp, empty, moves)
        else:  # White
            _white_moves(pieces, kings, opp, empty, moves)

        return [
            (moves.starts[i], moves.ends[i], list(moves.captures[i, 1 : moves.captures[i, 0]]))
            for i in range(moves.length)
        ]

    def make_move(self, move: CheckersMove) -> None:
        """
        Apply a move to the board. If it is a capture move, remove the captured piece.
        Also handle promotion if a man reaches the last row.
        Switch current player.
        Invalidate game over cache.
        """
        start, end = move
        assert 0 <= start < BOARD_SQUARES and 0 <= end < BOARD_SQUARES, 'Invalid move'

        # Remove captured piece if any:
        # Do this first, before moving the piece, to get the correct valid moves
        for s, e, captures in self._get_valid_moves():
            if s == start and e == end:
                for capture in captures:
                    capture_mask = np.uint64(1 << capture)
                    self.white_pieces &= ~capture_mask
                    self.white_kings &= ~capture_mask
                    self.black_pieces &= ~capture_mask
                    self.black_kings &= ~capture_mask
                break

        start_mask = np.uint64(1 << start)
        end_mask = np.uint64(1 << end)

        # Move piece:
        if self.current_player == 1:
            # Black to move
            if self.black_pieces & start_mask:
                self.black_pieces &= ~start_mask
                self.black_pieces |= end_mask

                if end_mask & BLACK_PROMOTION_ROW:
                    self.black_pieces &= ~end_mask
                    self.black_kings |= end_mask

            elif self.black_kings & start_mask:
                self.black_kings &= ~start_mask
                self.black_kings |= end_mask
        else:
            # White to move
            if self.white_pieces & start_mask:
                self.white_pieces &= ~start_mask
                self.white_pieces |= end_mask

                if end_mask & WHITE_PROMOTION_ROW:
                    self.white_pieces &= ~end_mask
                    self.white_kings |= end_mask

            elif self.white_kings & start_mask:
                self.white_kings &= ~start_mask
                self.white_kings |= end_mask

        # Switch player
        self._switch_player()

        # Invalidate game over cache
        self._invalidate_cache()

    @Board._cache()
    def check_winner(self) -> Optional[Player]:
        black_pieces = self.black_pieces | self.black_kings
        white_pieces = self.white_pieces | self.white_kings

        if black_pieces == 0:
            # White wins
            return -1
        if white_pieces == 0:
            # Black wins
            return 1

        # Check if current player has moves
        if not self.get_valid_moves():
            # Current player can't move, so opponent wins
            return -self.current_player

        return None

    def is_game_over(self) -> bool:
        return self.check_winner() is not None

    def copy(self) -> CheckersBoard:
        game = CheckersBoard()
        game.black_pieces = self.black_pieces
        game.black_kings = self.black_kings
        game.white_pieces = self.white_pieces
        game.white_kings = self.white_kings
        game.current_player = self.current_player
        return game

    def _occupied(self) -> np.uint64:
        return self.black_pieces | self.black_kings | self.white_pieces | self.white_kings

    def _empty(self) -> np.uint64:
        return ~self._occupied() & FULL_64

    def _opponent_pieces(self) -> np.uint64:
        if self.current_player == 1:  # black to move
            return self.white_pieces | self.white_kings
        else:
            return self.black_pieces | self.black_kings

    def _friendly_piece(self) -> np.uint64:
        return self.black_pieces if self.current_player == 1 else self.white_pieces

    def _friendly_kings(self) -> np.uint64:
        return self.black_kings if self.current_player == 1 else self.white_kings


if __name__ == '__main__':
    import time

    board = CheckersBoard()

    start = time.time()
    board.get_valid_moves()
    print('First compilation time:', time.time() - start)

    start = time.time()
    while not board.is_game_over():
        board.get_valid_moves()
        board.make_move(board.get_valid_moves()[0])
    print('Iteration time:', time.time() - start)
