
from abc import ABC, abstractmethod
from chess import Board, Move

class ChessBot(ABC):
    def __init__(self, name: str) -> None:
        """Initializes the bot with a name."""
        self.name = name

    @abstractmethod
    def think(self, board: Board) -> Move:
        """This method is called when it's the bot's turn to move. It should return the move that the bot wants to make."""
        raise NotImplementedError("Subclasses must implement this method")