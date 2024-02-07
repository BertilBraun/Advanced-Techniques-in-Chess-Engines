import random
from chess import Board, Move

from Framework.ChessBot import ChessBot

class RandomBot(ChessBot):
    def __init__(self) -> None:
        """Initializes the random player."""
        super().__init__("Random")
    
    def think(self, board: Board) -> Move:
        """Selects a random legal move."""
        legal_moves = list(board.legal_moves)
        
        return random.choice(legal_moves)