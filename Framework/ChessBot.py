
import time

from abc import ABC, abstractmethod
from chess import Board, Move

class ChessBot(ABC):
    def __init__(self, name: str) -> None:
        """Initializes the bot with a name."""
        self.name = name
        self.start_time = 0.0

    @abstractmethod
    def think(self, board: Board) -> Move:
        """This method is called when it's the bot's turn to move. It should return the move that the bot wants to make."""
        raise NotImplementedError("Subclasses must implement this method")
    
    @property
    def time_elapsed(self) -> float:
        """Returns the time elapsed since the bot started thinking."""
        return time.time() - self.start_time
    
    def restart_clock(self) -> None:
        """Restarts the clock for the bot."""
        self.start_time = time.time()
    
    def time_remaining(self) -> float:
        """
        Determines the time remaining for the bot to think.
        
        :return: The time remaining in seconds.
        """
        TIME_TO_THINK = 5 # seconds
        return TIME_TO_THINK - self.time_elapsed