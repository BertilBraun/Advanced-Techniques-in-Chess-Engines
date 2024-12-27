from abc import ABC, abstractmethod
import time

from src.settings import CURRENT_BOARD, CURRENT_GAME_MOVE


class Bot(ABC):
    def __init__(self, name: str, max_time_to_think: float) -> None:
        """Initializes the bot with a name."""
        self.name = name
        self.start_time = 0.0
        self.max_time_to_think = max_time_to_think

    @abstractmethod
    def think(self, board: CURRENT_BOARD) -> CURRENT_GAME_MOVE:
        """This method is called when it's the bot's turn to move. It should return the move that the bot wants to make."""
        raise NotImplementedError('Subclasses must implement this method')

    @property
    def time_elapsed(self) -> float:
        """Returns the time elapsed since the bot started thinking."""
        return time.time() - self.start_time

    @property
    def time_remaining(self) -> float:
        """
        Determines the time remaining for the bot to think.

        :return: The time remaining in seconds.
        """
        return self.max_time_to_think - self.time_elapsed

    @property
    def time_is_up(self) -> bool:
        """Determines if the bot has run out of time to think."""
        return self.time_remaining <= 0

    def restart_clock(self) -> None:
        """Restarts the clock for the bot."""
        self.start_time = time.time()
