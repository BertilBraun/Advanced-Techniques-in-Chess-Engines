import time
import chess
from typing import Optional
from abc import ABC, abstractmethod


def check_winner(board: chess.Board) -> Optional[chess.Color]:
    """
    Checks the winner of the game based on the board state.

    :param board: The chess board.
    :return: The color of the winning player or None if it's a draw.
    """
    result = board.result()
    if result == '1-0':
        return chess.WHITE
    elif result == '0-1':
        return chess.BLACK
    else:
        return None


class Bot(ABC):
    def __init__(self, name: str, max_time_to_think: float) -> None:
        """Initializes the bot with a name."""
        self.name = name
        self.start_time = 0.0
        self.max_time_to_think = max_time_to_think

    @abstractmethod
    def think(self, board: chess.Board) -> chess.Move:
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
