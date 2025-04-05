import chess
import AlphaZeroCpp

from src.eval.Bot import Bot
from src.settings import TRAINING_ARGS
from src.util.save_paths import model_save_path


class AlphaZeroBot(Bot):
    def __init__(self, iteration: int, max_time_to_think: float = 1.0, network_eval_only: bool = False) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)
        
        self.iteration = iteration
        self.max_time_to_think = max_time_to_think
        self.network_eval_only = network_eval_only
        
    def think(self, board: chess.Board) -> chess.Move:
        return AlphaZeroCpp.eval_board_iterate(model_save_path(self.iteration, TRAINING_ARGS.save_path), board.fen(), self.network_eval_only, self.max_time_to_think)
