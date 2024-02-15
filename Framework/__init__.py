from chess import *

from Framework.BitboardHelper import *
from Framework.ChessBot import ChessBot  # noqa: F401
from Framework.ExtendedMove import ExtendedMove, get_legal_moves  # noqa: F401


def get_board_hash(board: Board) -> int:
    return hash(board._transposition_key())
