from typing import Literal
from src.settings_common import *

CURRENT_GAME: Literal['tictactoe', 'connect4', 'hex', 'checkers', 'chess'] = 'chess'

if CURRENT_GAME == 'tictactoe':
    from src.games.tictactoe.TicTacToeSettings import *
elif CURRENT_GAME == 'connect4':
    from src.games.connect4.Connect4Settings import *
elif CURRENT_GAME == 'checkers':
    from src.games.checkers.CheckersSettings import *
elif CURRENT_GAME == 'hex':
    from src.games.hex.HexSettings import *
elif CURRENT_GAME == 'chess':
    from src.games.chess.ChessSettings import *
else:
    raise ValueError(f'Game {CURRENT_GAME} not supported. Supported games: hex, connect4, checkers, chess, tictactoe')
