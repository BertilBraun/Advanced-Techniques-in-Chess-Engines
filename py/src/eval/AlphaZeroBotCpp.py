from __future__ import annotations

import chess

from src.eval.Bot import Bot
from src.eval.InteractiveEngine import AnalysisMode, InteractiveEngine, InteractiveGame
from src.settings import CurrentBoard, CurrentGameMove, PLAY_C_PARAM
from src.util.log import log


class AlphaZeroBot(Bot):
    def __init__(
        self,
        current_model_path: str,
        device_id: int,
        max_time_to_think: float = 1.0,
        network_eval_only: bool = False,
    ) -> None:
        super().__init__("AlphaZeroBot", max_time_to_think)
        if not network_eval_only and not 1.0 <= max_time_to_think < 31.0:
            raise ValueError("MCTS max_time_to_think must be between 1 and 30 seconds")
        self.engine = InteractiveEngine(
            model_path=current_model_path,
            device_id=device_id,
            parallel_searches=64,
            c_param=PLAY_C_PARAM,
            inference_workers=2,
            outstanding_batches_per_worker=2,
        )
        self.network_eval_only = network_eval_only
        self.time_limit_seconds = int(max_time_to_think)
        self.game: InteractiveGame | None = None

    @staticmethod
    def _history(board: CurrentBoard) -> tuple[str, tuple[str, ...]]:
        native_board = board.board
        return native_board.root().fen(), tuple(
            move.uci() for move in native_board.move_stack
        )

    def _synchronize_game(self, board: CurrentBoard) -> InteractiveGame:
        starting_fen, moves_uci = self._history(board)
        if (
            self.game is None
            or self.game.starting_fen != starting_fen
            or moves_uci[: len(self.game.moves_uci)] != self.game.moves_uci
        ):
            self.game = self.engine.new_game(starting_fen, moves_uci)
            return self.game

        for move_uci in moves_uci[len(self.game.moves_uci) :]:
            self.game.apply_move(move_uci)
        return self.game

    def think(self, board: CurrentBoard) -> CurrentGameMove:
        game = self._synchronize_game(board)
        mode = AnalysisMode.POLICY if self.network_eval_only else AnalysisMode.MCTS
        time_limit_seconds = None if self.network_eval_only else self.time_limit_seconds
        result = game.analyze(mode=mode, time_limit_seconds=time_limit_seconds)
        move = chess.Move.from_uci(result.chosen_move_uci)
        if move not in board.get_valid_moves():
            raise ValueError(
                f"Engine returned illegal move {move.uci()} in FEN {board.board.fen()}"
            )

        game.apply_move(result.chosen_move_uci)
        log("---------------------- Alpha Zero Best Move ----------------------")
        log("Mode:", mode.value)
        log("Best move:", result.chosen_move_uci)
        log("Root value:", result.value)
        log("Completed searches:", result.searches)
        log("Maximum depth:", result.maximum_depth)
        log("Elapsed milliseconds:", result.elapsed_milliseconds)
        log("Principal variation:", result.principal_variation)
        log("Candidates:", result.candidates)
        log("------------------------------------------------------------------")
        return move
