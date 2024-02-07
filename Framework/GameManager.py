import time
import chess
from dataclasses import dataclass

from Framework.ChessBot import ChessBot
from Framework.GameResult import GameResult


@dataclass
class ThinkingTime:
    white: float = 0.0
    black: float = 0.0
    
    def update(self, color: chess.Color, time: float) -> None:
        if color == chess.WHITE:
            self.white += time
        else:
            self.black += time


class GameManager:
    def __init__(self, white: ChessBot, black: ChessBot) -> None:
        """Initializes the game manager with two players."""
        self.board = chess.Board()
        self.white = white
        self.black = black
        self.thinking_time = ThinkingTime()


    def play_game(self) -> tuple[GameResult, ThinkingTime]:
        """Manages the gameplay loop until the game is over or a player quits."""
        while not self.board.is_game_over():
            current_player = self.white if self.board.turn == chess.WHITE else self.black
            
            start_time = time.time()
            move = current_player.think(self.board)
            end_time = time.time()
            
            self.thinking_time.update(self.board.turn, end_time - start_time)
            
            self.board.push(move)
        
        return GameResult.from_board(self.board), self.thinking_time
