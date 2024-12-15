from __future__ import annotations
from enum import Enum
from typing import Callable, Optional, List, Tuple

import numpy as np

from AIZeroConnect4Bot.src.games.Game import Board, Player

CheckersMove = Tuple[int, int]
_CheckersMove = Tuple[int, int, List[int]]  # start, end, captures

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

# TODO remove
ILLEGAL_CELLS = [i + j * BOARD_SIZE for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if (i + j) % 2 == 0]


DR_SHIFT = -(BOARD_SIZE + 1)
DL_SHIFT = -(BOARD_SIZE - 1)
UR_SHIFT = BOARD_SIZE - 1
UL_SHIFT = BOARD_SIZE + 1


def _bit_set(bitboard: np.uint64, index: int) -> bool:
    return (bitboard & (np.uint64(1) << np.uint64(index))) != 0


def _down_right(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~H_FILE) << np.uint64(BOARD_SIZE + 1))


def _down_left(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~A_FILE) << np.uint64(BOARD_SIZE - 1))


def _up_right(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~H_FILE) >> np.uint64(BOARD_SIZE - 1))


def _up_left(bitboard: np.uint64) -> np.uint64:
    return np.uint64((bitboard & ~A_FILE) >> np.uint64(BOARD_SIZE + 1))


def _least_significant_bit(bitboard: np.uint64) -> int:
    int_bb = int(bitboard)
    return (int_bb & -int_bb).bit_length() - 1


def _bitboard_to_list(bitboard: np.uint64) -> List[int]:
    indices = []
    working = bitboard
    while working:
        index = _least_significant_bit(working)
        indices.append(index)
        working &= ~(np.uint64(1) << np.uint64(index))
    return indices


def _decode_moves(starts: np.uint64, ends: np.uint64, shift: int) -> List[_CheckersMove]:
    moves: List[_CheckersMove] = []
    for end_square in _bitboard_to_list(ends):
        start_square = end_square + shift
        if 0 <= start_square < BOARD_SQUARES and _bit_set(starts, start_square):
            moves.append((start_square, end_square, []))
    return moves


def _piece_normal_moves(
    pieces: np.uint64,
    empty: np.uint64,
    shift: int,
    left_func: Callable[[np.uint64], np.uint64],
    right_func: Callable[[np.uint64], np.uint64],
) -> List[_CheckersMove]:
    piece_left = left_func(pieces) & empty
    moves_left = _decode_moves(pieces, piece_left, shift=shift)

    piece_right = right_func(pieces) & empty
    moves_right = _decode_moves(pieces, piece_right, shift=shift)

    return moves_left + moves_right


def _black_piece_normal_moves(pieces: np.uint64, empty: np.uint64) -> List[_CheckersMove]:
    piece_dl = _down_left(pieces) & empty
    moves_dl = _decode_moves(pieces, piece_dl, shift=DL_SHIFT)

    piece_dr = _down_right(pieces) & empty
    moves_dr = _decode_moves(pieces, piece_dr, shift=DR_SHIFT)

    return moves_dl + moves_dr


def _black_piece_single_captures(pieces: np.uint64, enemy: np.uint64, empty: np.uint64) -> List[_CheckersMove]:
    # Down-left capture
    dl_step = _down_left(pieces) & enemy
    dl_cap = _down_left(dl_step) & empty
    moves_dl = _decode_moves(pieces, dl_cap, shift=2 * DL_SHIFT)
    for s, _, captures in moves_dl:
        captures.append(s - DL_SHIFT)

    # Down-right capture
    dr_step = _down_right(pieces) & enemy
    dr_cap = _down_right(dr_step) & empty
    moves_dr = _decode_moves(pieces, dr_cap, shift=2 * DR_SHIFT)
    for s, _, captures in moves_dr:
        captures.append(s - DR_SHIFT)

    return moves_dl + moves_dr


DIRECTIONS = ((DR_SHIFT, (1, 1)), (DL_SHIFT, (-1, 1)), (UR_SHIFT, (1, -1)), (UL_SHIFT, (-1, -1)))


def _king_moves(occupied: np.uint64, enemy: np.uint64, start: int) -> Tuple[List[_CheckersMove], List[_CheckersMove]]:
    normal_moves: List[_CheckersMove] = []
    captures: List[_CheckersMove] = []

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
                        _, next_captures = _king_moves(occupied, enemy, pos - shift)
                        occupied |= np.uint64(1) << np.uint64(start)
                        occupied |= np.uint64(1) << np.uint64(pos)
                        occupied &= ~(np.uint64(1) << np.uint64(pos - shift))
                        enemy |= np.uint64(1) << np.uint64(pos)
                        for s, e, capt in next_captures:
                            captures.append((start, e, [pos] + capt))
                        if not next_captures:
                            captures.append((start, pos - shift, [pos]))
                # else: blocked by a piece
                break
            # empty square
            normal_moves.append((start, pos, []))

    return normal_moves, captures


def _black_moves(pieces: np.uint64, kings: np.uint64, enemy: np.uint64, empty: np.uint64) -> List[_CheckersMove]:
    all_moves: List[_CheckersMove] = []
    all_captures: List[_CheckersMove] = []
    # Black piece moves (down-left and down-right)
    piece_captures = _black_piece_single_captures(pieces, enemy, empty)
    print('Black Piece captures:', piece_captures)
    if piece_captures:
        outstanding_captures = piece_captures
        while outstanding_captures:
            start, end, captures = outstanding_captures.pop()
            print('Start:', start, 'End:', end, 'Captures:', captures, 'is king:', _bit_set(kings, start))

            ey, ex = divmod(end, BOARD_SIZE)
            added = False

            if (
                ey >= 2  # in bounds
                and ex < BOARD_SIZE - 2  # in bounds
                and _bit_set(enemy, end - DR_SHIFT)  # next square is enemy
                and _bit_set(empty, end - 2 * DR_SHIFT)  # square after enemy is empty
            ):
                print('1', start, end - 2 * DR_SHIFT, captures + [end - DR_SHIFT], ex, ey)
                outstanding_captures.append((start, end - 2 * DR_SHIFT, captures + [end - DR_SHIFT]))
                added = True
            if (
                ey >= 2  # in bounds
                and ex >= 2  # in bounds
                and _bit_set(enemy, end - DL_SHIFT)  # next square is enemy
                and _bit_set(empty, end - 2 * DL_SHIFT)  # square after enemy is empty
            ):
                print('2', start, end - 2 * DL_SHIFT, captures + [end - DL_SHIFT], ex, ey)
                outstanding_captures.append((start, end - 2 * DL_SHIFT, captures + [end - DL_SHIFT]))
                added = True

            if not added:  # no further jump captures possible
                all_captures.append((start, end, captures))

    # Black king moves
    for start in _bitboard_to_list(kings):
        normal_moves, king_captures = _king_moves(~empty, enemy, start)
        all_captures += king_captures
        all_moves += normal_moves

    if all_captures:
        return all_captures

    all_moves += _black_piece_normal_moves(pieces, empty)
    return all_moves


def _white_piece_normal_moves(pieces: np.uint64, empty: np.uint64) -> List[_CheckersMove]:
    piece_ul = _up_left(pieces) & empty
    moves_ul = _decode_moves(pieces, piece_ul, shift=UL_SHIFT)

    piece_ur = _up_right(pieces) & empty
    moves_ur = _decode_moves(pieces, piece_ur, shift=UR_SHIFT)

    return moves_ul + moves_ur


def _white_piece_single_captures(pieces: np.uint64, enemy: np.uint64, empty: np.uint64) -> List[_CheckersMove]:
    # Up-left capture
    ul_step = _up_left(pieces) & enemy
    ul_cap = _up_left(ul_step) & empty
    moves_ul = _decode_moves(pieces, ul_cap, shift=2 * UL_SHIFT)
    for s, _, captures in moves_ul:
        captures.append(s - UL_SHIFT)

    # Up-right capture
    ur_step = _up_right(pieces) & enemy
    ur_cap = _up_right(ur_step) & empty
    moves_ur = _decode_moves(pieces, ur_cap, shift=2 * UR_SHIFT)
    for s, _, captures in moves_ur:
        captures.append(s - UR_SHIFT)

    return moves_ul + moves_ur


def _white_moves(pieces: np.uint64, kings: np.uint64, enemy: np.uint64, empty: np.uint64) -> List[_CheckersMove]:
    all_moves: List[_CheckersMove] = []
    all_captures: List[_CheckersMove] = []
    # White piece moves (up-left and up-right)
    piece_captures = _white_piece_single_captures(pieces, enemy, empty)
    print('White Piece captures:', piece_captures)
    if piece_captures:
        outstanding_captures = piece_captures
        while outstanding_captures:
            start, end, captures = outstanding_captures.pop()
            print('Start:', start, 'End:', end, 'Captures:', captures, 'is king:', _bit_set(kings, start))

            ey, ex = divmod(end, BOARD_SIZE)
            added = False

            if (
                ey < BOARD_SIZE - 2  # in bounds
                and ex < BOARD_SIZE - 2  # in bounds
                and _bit_set(enemy, end - UR_SHIFT)  # next square is enemy
                and _bit_set(empty, end - 2 * UR_SHIFT)  # square after enemy is empty
            ):
                print('1', start, end - 2 * UR_SHIFT, captures + [end - UR_SHIFT], ex, ey)
                outstanding_captures.append((start, end - 2 * UR_SHIFT, captures + [end - UR_SHIFT]))
                added = True
            if (
                ey < BOARD_SIZE - 2  # in bounds
                and ex >= 2  # in bounds
                and _bit_set(enemy, end - UL_SHIFT)  # next square is enemy
                and _bit_set(empty, end - 2 * UL_SHIFT)  # square after enemy is empty
            ):
                print('2', start, end - 2 * UL_SHIFT, captures + [end - UL_SHIFT], ex, ey)
                outstanding_captures.append((start, end - 2 * UL_SHIFT, captures + [end - UL_SHIFT]))
                added = True

            if not added:  # no further jump captures possible
                all_captures.append((start, end, captures))

    # White king moves
    for start in _bitboard_to_list(kings):
        normal_moves, king_captures = _king_moves(~empty, enemy, start)
        all_captures += king_captures
        all_moves += normal_moves

    if all_captures:
        return all_captures

    all_moves += _white_piece_normal_moves(pieces, empty)
    return all_moves


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
    def _get_valid_moves(self) -> List[_CheckersMove]:
        pieces = self._friendly_piece()
        kings = self._friendly_kings()
        opp = self._opponent_pieces()
        empty = self._empty()

        def _inner():
            if self.current_player == 1:  # Black
                return _black_moves(pieces, kings, opp, empty)
            else:  # White
                return _white_moves(pieces, kings, opp, empty)

        moves = _inner()
        for s, e, captures in moves:
            if s in ILLEGAL_CELLS or e in ILLEGAL_CELLS or s < 0 or e < 0 or s >= BOARD_SQUARES or e >= BOARD_SQUARES:
                print(f'Illegal move: {s} -> {e} with captures {captures}')
                print('Piece:', self.get_cell(s // BOARD_SIZE, s % BOARD_SIZE))
                print('Start:', s // BOARD_SIZE, s % BOARD_SIZE)
                print('End:', e // BOARD_SIZE, e % BOARD_SIZE)
                print('Current board:')
                for i in range(BOARD_SIZE):
                    for j in range(BOARD_SIZE):
                        cell = self.get_cell(i, j)
                        if i * BOARD_SIZE + j == s:
                            print(' S', end='   ')
                        elif i * BOARD_SIZE + j == e:
                            print(' E', end='   ')
                        elif cell is None:
                            print('  ', end='   ')
                        elif cell.value < 0:
                            print(cell.value, end='   ')
                        else:
                            print(f' {cell.value}', end='   ')
                    print()
                    for j in range(BOARD_SIZE):
                        print(f'{i * BOARD_SIZE + j:2}', end='   ')
                    print()
                    print()
                exit()
        print('Valid moves:')
        from pprint import pprint

        pprint(moves)

        return moves

    def make_move(self, move: CheckersMove) -> None:
        """
        Apply a move to the board. If it is a capture move, remove the captured piece.
        Also handle promotion if a man reaches the last row.
        Switch current player.
        Invalidate game over cache.
        """
        start, end = move
        assert 0 <= start < BOARD_SQUARES and 0 <= end < BOARD_SQUARES, 'Invalid move'
        assert start not in ILLEGAL_CELLS and end not in ILLEGAL_CELLS, 'Illegal cell'

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

        # Remove captured piece if any:
        for s, e, captures in self._get_valid_moves():
            if s == start and e == end:
                for capture in captures:
                    capture_mask = np.uint64(1 << capture)
                    self.white_pieces &= ~capture_mask
                    self.white_kings &= ~capture_mask
                    self.black_pieces &= ~capture_mask
                    self.black_kings &= ~capture_mask
                break

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
