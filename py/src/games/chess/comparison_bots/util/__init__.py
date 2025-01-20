from src.games.chess.comparison_bots.util.piece_tables import *
from src.games.chess.comparison_bots.util.transposition_entry import *
from src.games.chess.comparison_bots.util.extended_move import *


def get_board_hash(board: Board) -> int:
    return hash(board._transposition_key())


def get_number_of_set_bits(bitboard: int) -> int:
    """Returns the number of bits that are set to 1 in the given bitboard."""
    return bin(bitboard).count('1')


MAX_TIME_TO_THINK = 0.2  # 200ms of thinking time
