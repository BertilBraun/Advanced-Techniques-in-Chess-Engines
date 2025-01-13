from __future__ import annotations
from enum import Enum
from typing import Callable, Optional, List, Tuple

from src.games.Game import Board, Player


CheckersMove = Tuple[int, int]
_CheckersMove = Tuple[int, int, List[int]]  # start, end, captures

# TODO far from 100% Core usage during move generation - why?

uint64 = int

BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# File masks to prevent wrapping
A_FILE: uint64 = 0x0101010101010101  # leftmost column bits set
H_FILE: uint64 = 0x8080808080808080  # rightmost column bits set

# Mask for 64-bit
FULL_64: uint64 = (1 << 64) - 1

# Starting positions (bitboards)
# Black piece:
BLACK_MEN_START: uint64 = (
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
WHITE_MEN_START: uint64 = (
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
BLACK_PROMOTION_ROW: uint64 = 0xFF00000000000000  # top row (row 7 in index terms, bits 56-63)
WHITE_PROMOTION_ROW: uint64 = 0x00000000000000FF  # bottom row (row 0 in index terms, bits 0-7)


DR_SHIFT = -(BOARD_SIZE + 1)
DL_SHIFT = -(BOARD_SIZE - 1)
UR_SHIFT = BOARD_SIZE - 1
UL_SHIFT = BOARD_SIZE + 1


def _bit_set(bitboard: uint64, index: int) -> bool:
    return (bitboard & ((1) << (index))) != 0


def _down_right(bitboard: uint64) -> uint64:
    return (bitboard & ~H_FILE) << (BOARD_SIZE + 1)


def _down_left(bitboard: uint64) -> uint64:
    return (bitboard & ~A_FILE) << (BOARD_SIZE - 1)


def _up_right(bitboard: uint64) -> uint64:
    return (bitboard & ~H_FILE) >> (BOARD_SIZE - 1)


def _up_left(bitboard: uint64) -> uint64:
    return (bitboard & ~A_FILE) >> (BOARD_SIZE + 1)


def _least_significant_bit(bitboard: uint64) -> int:
    return (bitboard & -bitboard).bit_length() - 1


def _bitboard_to_list(bitboard: uint64) -> List[int]:
    indices = []
    working = bitboard
    while working:
        index = _least_significant_bit(working)
        indices.append(index)
        working &= ~((1) << (index))
    return indices


def _decode_moves(starts: uint64, ends: uint64, shift: int) -> List[_CheckersMove]:
    moves: List[_CheckersMove] = []
    for end_square in _bitboard_to_list(ends):
        start_square = end_square + shift
        if 0 <= start_square < BOARD_SQUARES and _bit_set(starts, start_square):
            moves.append((start_square, end_square, []))
    return moves


MaskFunc = Callable[[uint64], uint64]


def _piece_normal_moves(
    pieces: uint64,
    empty: uint64,
    shift: int,
    mask_func: MaskFunc,
) -> List[_CheckersMove]:
    piece_moves = mask_func(pieces) & empty
    return _decode_moves(pieces, piece_moves, shift)


def _black_piece_normal_moves(pieces: uint64, empty: uint64) -> List[_CheckersMove]:
    dl_moves = _piece_normal_moves(pieces, empty, DL_SHIFT, _down_left)
    dr_moves = _piece_normal_moves(pieces, empty, DR_SHIFT, _down_right)
    return dl_moves + dr_moves


def _white_piece_normal_moves(pieces: uint64, empty: uint64) -> List[_CheckersMove]:
    ul_moves = _piece_normal_moves(pieces, empty, UL_SHIFT, _up_left)
    ur_moves = _piece_normal_moves(pieces, empty, UR_SHIFT, _up_right)
    return ul_moves + ur_moves


def _piece_single_captures(
    pieces: uint64, enemy: uint64, empty: uint64, shift: int, mask_func: MaskFunc
) -> List[_CheckersMove]:
    step = mask_func(pieces) & enemy
    cap = mask_func(step) & empty
    moves = _decode_moves(pieces, cap, 2 * shift)
    for s, _, captures in moves:
        captures.append(s - shift)
    return moves


def _black_piece_single_captures(pieces: uint64, enemy: uint64, empty: uint64) -> List[_CheckersMove]:
    dl_captures = _piece_single_captures(pieces, enemy, empty, DL_SHIFT, _down_left)
    dr_captures = _piece_single_captures(pieces, enemy, empty, DR_SHIFT, _down_right)
    return dl_captures + dr_captures


def _white_piece_single_captures(pieces: uint64, enemy: uint64, empty: uint64) -> List[_CheckersMove]:
    ul_captures = _piece_single_captures(pieces, enemy, empty, UL_SHIFT, _up_left)
    ur_captures = _piece_single_captures(pieces, enemy, empty, UR_SHIFT, _up_right)
    return ul_captures + ur_captures


DIRECTIONS = ((DR_SHIFT, (1, 1)), (DL_SHIFT, (-1, 1)), (UR_SHIFT, (1, -1)), (UL_SHIFT, (-1, -1)))


def _king_moves(occupied: uint64, enemy: uint64, start: int) -> Tuple[List[_CheckersMove], List[_CheckersMove]]:
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
                        occupied &= ~((1) << (pos))
                        occupied &= ~((1) << (start))
                        occupied |= (1) << (pos - shift)
                        enemy &= ~((1) << (pos))
                        _, next_captures = _king_moves(occupied, enemy, pos - shift)
                        occupied |= (1) << (start)
                        occupied |= (1) << (pos)
                        occupied &= ~((1) << (pos - shift))
                        enemy |= (1) << (pos)
                        for s, e, capt in next_captures:
                            captures.append((start, e, [pos] + capt))
                        if not next_captures:
                            captures.append((start, pos - shift, [pos]))
                # else: blocked by a piece
                break
            # empty square
            normal_moves.append((start, pos, []))

    return normal_moves, captures


def _check_further_capture(
    start: int,
    end: int,
    shift: int,
    dx: int,
    dy: int,
    enemy: uint64,
    empty: uint64,
) -> Optional[_CheckersMove]:
    ey, ex = divmod(end, BOARD_SIZE)
    if (
        ey + 2 * dy >= 0  # in bounds
        and ey + 2 * dy < BOARD_SIZE  # in bounds
        and ex + 2 * dx >= 0  # in bounds
        and ex + 2 * dx < BOARD_SIZE  # in bounds
        and _bit_set(enemy, end - shift)  # next square is enemy
        and _bit_set(empty, end - 2 * shift)  # square after enemy is empty
    ):
        return start, end - 2 * shift, [end - shift]
    return None


def _expand_single_captures(
    piece_captures: List[_CheckersMove],
    enemy: uint64,
    empty: uint64,
    shift_left: int,
    shift_right: int,
    dy: int,
) -> List[_CheckersMove]:
    outstanding_captures: List[_CheckersMove] = piece_captures
    all_captures: List[_CheckersMove] = []

    while outstanding_captures:
        start, end, captures = outstanding_captures.pop()

        capture_right = _check_further_capture(start, end, shift_right, 1, dy, enemy, empty)
        capture_left = _check_further_capture(start, end, shift_left, -1, dy, enemy, empty)

        if not capture_right and not capture_left:
            # no further captures possible, add to all_captures
            all_captures.append((start, end, captures))
            continue

        if capture_right:
            capture_right[2].extend(captures)
            outstanding_captures.append(capture_right)
        if capture_left:
            capture_left[2].extend(captures)
            outstanding_captures.append(capture_left)

    return all_captures


def _black_moves(pieces: uint64, kings: uint64, enemy: uint64, empty: uint64) -> List[_CheckersMove]:
    piece_captures = _black_piece_single_captures(pieces, enemy, empty)
    if piece_captures:
        all_captures = _expand_single_captures(piece_captures, enemy, empty, DL_SHIFT, DR_SHIFT, 1)
    else:
        all_captures: List[_CheckersMove] = []

    all_moves: List[_CheckersMove] = []
    for start in _bitboard_to_list(kings):
        normal_moves, king_captures = _king_moves(~empty, enemy, start)
        all_captures += king_captures
        all_moves += normal_moves

    if all_captures:
        return all_captures

    all_moves += _black_piece_normal_moves(pieces, empty)
    return all_moves


def _white_moves(pieces: uint64, kings: uint64, enemy: uint64, empty: uint64) -> List[_CheckersMove]:
    piece_captures = _white_piece_single_captures(pieces, enemy, empty)
    if piece_captures:
        all_captures = _expand_single_captures(piece_captures, enemy, empty, UL_SHIFT, UR_SHIFT, -1)
    else:
        all_captures: List[_CheckersMove] = []

    all_moves: List[_CheckersMove] = []
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
        self.black_pieces: uint64 = BLACK_MEN_START
        self.black_kings: uint64 = 0
        self.white_pieces: uint64 = WHITE_MEN_START
        self.white_kings: uint64 = 0

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

        if self.current_player == 1:  # Black
            return _black_moves(pieces, kings, opp, empty)
        else:  # White
            return _white_moves(pieces, kings, opp, empty)

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
                    capture_mask = 1 << capture
                    self.white_pieces &= ~capture_mask
                    self.white_kings &= ~capture_mask
                    self.black_pieces &= ~capture_mask
                    self.black_kings &= ~capture_mask
                break

        start_mask = 1 << start
        end_mask = 1 << end

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
                assert False, 'Invalid move'
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

            else:
                assert False, 'Invalid move'

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
        game._copy_cache(self)  # Copy cached results
        return game

    def _occupied(self) -> uint64:
        return self.black_pieces | self.black_kings | self.white_pieces | self.white_kings

    def _empty(self) -> uint64:
        return ~self._occupied() & FULL_64

    def _opponent_pieces(self) -> uint64:
        if self.current_player == 1:  # black to move
            return self.white_pieces | self.white_kings
        else:
            return self.black_pieces | self.black_kings

    def _friendly_piece(self) -> uint64:
        return self.black_pieces if self.current_player == 1 else self.white_pieces

    def _friendly_kings(self) -> uint64:
        return self.black_kings if self.current_player == 1 else self.white_kings

    def quick_hash(self) -> int:
        return hash((self.black_kings, self.black_pieces, self.white_kings, self.white_pieces))

    def __repr__(self) -> str:
        # Print the board, with black pieces as 'b', white pieces as 'w', kings as 'B' and 'W'
        rows = []
        for row in range(BOARD_SIZE):
            cells = []
            for col in range(BOARD_SIZE):
                piece = self.get_cell(row, col)
                if piece is None:
                    cells.append('.')
                elif piece == Piece.BLACK_PIECE:
                    cells.append('b')
                elif piece == Piece.WHITE_PIECE:
                    cells.append('w')
                elif piece == Piece.BLACK_KING:
                    cells.append('B')
                elif piece == Piece.WHITE_KING:
                    cells.append('W')
            rows.append(' '.join(cells))

        return '\n'.join(rows)
