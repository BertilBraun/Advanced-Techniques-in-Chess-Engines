from __future__ import annotations
from typing import Optional, List, Tuple

from AIZeroConnect4Bot.src.games.Game import Board

CheckersMove = Tuple[int, int]

# File masks to prevent wrapping
A_FILE = 0x0101010101010101  # leftmost column bits set
H_FILE = 0x8080808080808080  # rightmost column bits set

# Mask for 64-bit
FULL_64 = (1 << 64) - 1

# Starting positions (bitboards)
# Black men:
BLACK_MEN_START = (
    (1 << 0)
    | (1 << 2)
    | (1 << 4)
    | (1 << 6)
    | (1 << 9)
    | (1 << 11)
    | (1 << 13)
    | (1 << 15)
    | (1 << 16)
    | (1 << 18)
    | (1 << 20)
    | (1 << 22)
)

# White men:
WHITE_MEN_START = (
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
BLACK_PROMOTION_ROW = 0xFF00000000000000  # top row (row 7 in index terms, bits 56-63)
WHITE_PROMOTION_ROW = 0x00000000000000FF  # bottom row (row 0 in index terms, bits 0-7)


class CheckersBoard(Board[CheckersMove]):
    def __init__(self) -> None:
        super().__init__()
        self.black_pieces: int = BLACK_MEN_START
        self.black_kings: int = 0
        self.white_pieces: int = WHITE_MEN_START
        self.white_kings: int = 0

    @property
    def board_dimensions(self) -> Tuple[int, int]:
        return 8, 8

    @Board._cache()
    def get_valid_moves(self) -> List[CheckersMove]:
        """
        Returns a list of (start, end) moves. If captures exist, return only captures.
        """
        # TODO possibly currently wrong
        men = self._friendly_men()
        kings = self._friendly_kings()
        opp = self._opponent_pieces()
        empty = self._empty()

        if self.current_player == 1:  # Black
            captures = self._black_captures(men, kings, opp, empty)
            if captures:
                return captures
            return self._black_normal_moves(men, kings, empty)
        else:  # White
            captures = self._white_captures(men, kings, opp, empty)
            if captures:
                return captures
            return self._white_normal_moves(men, kings, empty)

    def make_move(self, move: CheckersMove) -> None:
        """
        Apply a move to the board. If it is a capture move, remove the captured piece.
        Also handle promotion if a man reaches the last row.
        Switch current player.
        Invalidate game over cache.
        """
        # TODO possibly currently wrong

        start, end = move
        start_mask = 1 << start
        end_mask = 1 << end

        # Determine if capture:
        diff = end - start
        if abs(diff) > 10:  # More than a step, likely a capture (14 or 18)
            is_capture = True
            # intermediate square:
            mid = (start + end) // 2
            mid_mask = 1 << mid
        else:
            is_capture = False
            mid_mask = 0

        # Move piece:
        moved_piece = None
        if self.current_player == 1:
            # Black to move
            if self.black_pieces & start_mask:
                self.black_pieces &= ~start_mask
                self.black_pieces |= end_mask
                moved_piece = 'black_man'
            elif self.black_kings & start_mask:
                self.black_kings &= ~start_mask
                self.black_kings |= end_mask
                moved_piece = 'black_king'
        else:
            # White to move
            if self.white_pieces & start_mask:
                self.white_pieces &= ~start_mask
                self.white_pieces |= end_mask
                moved_piece = 'white_man'
            elif self.white_kings & start_mask:
                self.white_kings &= ~start_mask
                self.white_kings |= end_mask
                moved_piece = 'white_king'

        # Remove captured piece if any:
        if is_capture:
            # Captured piece could be in either opponent men or kings
            if self.current_player == 1:
                # Removing white piece
                if self.white_pieces & mid_mask:
                    self.white_pieces &= ~mid_mask
                else:
                    self.white_kings &= ~mid_mask
            else:
                # Removing black piece
                if self.black_pieces & mid_mask:
                    self.black_pieces &= ~mid_mask
                else:
                    self.black_kings &= ~mid_mask

        # Promotion:
        if moved_piece == 'black_man':
            # If black reaches last row (top row?), black moves from top to bottom in this setup:
            # Actually black starts at top (0) and moves down, so bottom row is bits 56-63.
            # Promotion when black reaches row 7 (indices 56-63)
            if end_mask & BLACK_PROMOTION_ROW:
                self.black_pieces &= ~end_mask
                self.black_kings |= end_mask
        elif moved_piece == 'white_man':
            # White moves up, so promotion row is top row (bits 0-7)
            if end_mask & WHITE_PROMOTION_ROW:
                self.white_pieces &= ~end_mask
                self.white_kings |= end_mask

        # Switch player
        self._switch_player()

        # Invalidate game over cache
        self._invalidate_cache()

    @Board._cache()
    def check_winner(self) -> Optional[int]:
        black_pieces = self.black_pieces | self.black_kings
        white_pieces = self.white_pieces | self.white_kings

        if black_pieces == 0:
            # White wins
            return -1
        if white_pieces == 0:
            # Black wins
            return 1

        # Check if current player has moves
        moves = self.get_valid_moves()
        if not moves:
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

    def _occupied(self) -> int:
        return self.black_pieces | self.black_kings | self.white_pieces | self.white_kings

    def _empty(self) -> int:
        return ~self._occupied() & FULL_64

    def _opponent_pieces(self) -> int:
        if self.current_player == 1:  # black to move
            return self.white_pieces | self.white_kings
        else:
            return self.black_pieces | self.black_kings

    def _friendly_men(self) -> int:
        return self.black_pieces if self.current_player == 1 else self.white_pieces

    def _friendly_kings(self) -> int:
        return self.black_kings if self.current_player == 1 else self.white_kings

    def _black_normal_moves(self, men: int, kings: int, empty: int) -> List[CheckersMove]:
        moves = []
        # Black men move down-left (<<7) and down-right (<<9)
        men_dl = ((men & ~H_FILE) << 7) & empty
        men_dr = ((men & ~A_FILE) << 9) & empty

        # Black kings can also move up-left (>>9), up-right (>>7)
        # plus the same down moves as men
        kings_dl = ((kings & ~H_FILE) << 7) & empty
        kings_dr = ((kings & ~A_FILE) << 9) & empty
        kings_ul = ((kings & ~H_FILE) >> 9) & empty
        kings_ur = ((kings & ~A_FILE) >> 7) & empty

        # Decode moves (for each set bit in men_dl, find start and end)
        moves += self._decode_single_step_moves(men, men_dl, shift=-7)
        moves += self._decode_single_step_moves(men, men_dr, shift=-9)
        moves += self._decode_single_step_moves(kings, kings_dl, shift=-7)
        moves += self._decode_single_step_moves(kings, kings_dr, shift=-9)
        moves += self._decode_single_step_moves(kings, kings_ul, shift=9)
        moves += self._decode_single_step_moves(kings, kings_ur, shift=7)

        return moves

    def _white_normal_moves(self, men: int, kings: int, empty: int) -> List[CheckersMove]:
        moves = []
        # White men move up-left (>>9) and up-right (>>7)
        men_ul = ((men & ~H_FILE) >> 9) & empty
        men_ur = ((men & ~A_FILE) >> 7) & empty

        # White kings can move in all four directions:
        kings_dl = ((kings & ~H_FILE) << 7) & empty
        kings_dr = ((kings & ~A_FILE) << 9) & empty
        kings_ul = ((kings & ~H_FILE) >> 9) & empty
        kings_ur = ((kings & ~A_FILE) >> 7) & empty

        moves += self._decode_single_step_moves(men, men_ul, shift=9)
        moves += self._decode_single_step_moves(men, men_ur, shift=7)
        moves += self._decode_single_step_moves(kings, kings_dl, shift=-7)
        moves += self._decode_single_step_moves(kings, kings_dr, shift=-9)
        moves += self._decode_single_step_moves(kings, kings_ul, shift=9)
        moves += self._decode_single_step_moves(kings, kings_ur, shift=7)

        return moves

    def _black_captures(self, men: int, kings: int, opp: int, empty: int) -> List[CheckersMove]:
        captures = []
        # Black men captures:
        # down-left capture: men must not be on H-file for first step
        # start -> (start<<7) opponent -> (start<<14) empty
        # Careful with masking at each step:
        dl_step1 = ((men & ~H_FILE) << 7) & opp
        dl_cap = ((dl_step1 & ~H_FILE) << 7) & empty
        captures += self._decode_capture_moves(men, dl_cap, intermediate_shift=-7, final_shift=-14)

        # down-right capture:
        dr_step1 = ((men & ~A_FILE) << 9) & opp
        dr_cap = ((dr_step1 & ~A_FILE) << 9) & empty
        captures += self._decode_capture_moves(men, dr_cap, intermediate_shift=-9, final_shift=-18)

        # Kings can capture in all four directions:
        # Up-left (>>9 then >>9)
        kul_step1 = ((kings & ~H_FILE) >> 9) & opp
        kul_cap = ((kul_step1 & ~H_FILE) >> 9) & empty
        captures += self._decode_capture_moves(kings, kul_cap, intermediate_shift=9, final_shift=18)

        # Up-right (>>7 then >>7)
        kur_step1 = ((kings & ~A_FILE) >> 7) & opp
        kur_cap = ((kur_step1 & ~A_FILE) >> 7) & empty
        captures += self._decode_capture_moves(kings, kur_cap, intermediate_shift=7, final_shift=14)

        # Down-left for kings (similar to men but kings bitboard):
        kdl_step1 = ((kings & ~H_FILE) << 7) & opp
        kdl_cap = ((kdl_step1 & ~H_FILE) << 7) & empty
        captures += self._decode_capture_moves(kings, kdl_cap, intermediate_shift=-7, final_shift=-14)

        # Down-right for kings
        kdr_step1 = ((kings & ~A_FILE) << 9) & opp
        kdr_cap = ((kdr_step1 & ~A_FILE) << 9) & empty
        captures += self._decode_capture_moves(kings, kdr_cap, intermediate_shift=-9, final_shift=-18)

        return captures

    def _white_captures(self, men: int, kings: int, opp: int, empty: int) -> List[CheckersMove]:
        captures = []
        # White men captures:
        # up-left: (men>>9)&opp then (>>9)&empty
        ul_step1 = ((men & ~H_FILE) >> 9) & opp
        ul_cap = ((ul_step1 & ~H_FILE) >> 9) & empty
        captures += self._decode_capture_moves(men, ul_cap, intermediate_shift=9, final_shift=18)

        # up-right:
        ur_step1 = ((men & ~A_FILE) >> 7) & opp
        ur_cap = ((ur_step1 & ~A_FILE) >> 7) & empty
        captures += self._decode_capture_moves(men, ur_cap, intermediate_shift=7, final_shift=14)

        # Kings captures (all four directions, same as black kings):
        # Down-left
        kdl_step1 = ((kings & ~H_FILE) << 7) & opp
        kdl_cap = ((kdl_step1 & ~H_FILE) << 7) & empty
        captures += self._decode_capture_moves(kings, kdl_cap, intermediate_shift=-7, final_shift=-14)

        # Down-right
        kdr_step1 = ((kings & ~A_FILE) << 9) & opp
        kdr_cap = ((kdr_step1 & ~A_FILE) << 9) & empty
        captures += self._decode_capture_moves(kings, kdr_cap, intermediate_shift=-9, final_shift=-18)

        # Up-left
        kul_step1 = ((kings & ~H_FILE) >> 9) & opp
        kul_cap = ((kul_step1 & ~H_FILE) >> 9) & empty
        captures += self._decode_capture_moves(kings, kul_cap, intermediate_shift=9, final_shift=18)

        # Up-right
        kur_step1 = ((kings & ~A_FILE) >> 7) & opp
        kur_cap = ((kur_step1 & ~A_FILE) >> 7) & empty
        captures += self._decode_capture_moves(kings, kur_cap, intermediate_shift=7, final_shift=14)

        return captures

    def _decode_single_step_moves(self, starts: int, ends: int, shift: int) -> List[CheckersMove]:
        # For each bit in ends set, find corresponding start
        # If end = start << n, then start = end >> n (for positive n), or start = end << (-n) for negative n.
        moves = []
        while ends:
            end_square = (ends & (-ends)).bit_length() - 1  # index of least significant set bit
            end_mask = 1 << end_square
            # reverse the shift
            if shift > 0:
                start_square = end_square + shift
            else:
                start_square = end_square - (-shift)

            if (starts & (1 << start_square)) != 0:
                moves.append((start_square, end_square))
            ends &= ~end_mask
        return moves

    def _decode_capture_moves(
        self, starts: int, ends: int, intermediate_shift: int, final_shift: int
    ) -> List[CheckersMove]:
        # For captures:
        # final position = start + final_shift
        # intermediate (captured piece) = start + intermediate_shift
        # We know ends are final landing squares. We need to re-derive start.
        moves = []
        while ends:
            end_square = (ends & (-ends)).bit_length() - 1
            end_mask = 1 << end_square
            start_square = end_square + final_shift
            if (starts & (1 << start_square)) != 0:
                moves.append((start_square, end_square))
            ends &= ~end_mask
        return moves
