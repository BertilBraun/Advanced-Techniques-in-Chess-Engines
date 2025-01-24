from chess import *  # type: ignore
from src.eval.Bot import Bot

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.comparison_bots.util import *


class HandcraftedBotV2(Bot):
    def __init__(self) -> None:
        super().__init__('HandcraftedBotV2', max_time_to_think=MAX_TIME_TO_THINK)
        self.transposition_table: list[float | None] = [None] * 2**16

    async def think(self, board: ChessBoard) -> Move:
        """
        Determine the best move given the current board state, using iterative deepening.

        :param board: The current board state.
        :return: The best move as determined by the engine.
        """
        self.best_move = Move.null()
        depth = 1

        while not self.time_is_up:
            self.negamax(board.board, depth, -float('inf'), float('inf'), board.board.turn)
            depth += 1

        return self.best_move

    def negamax(self, board: Board, depth: int, alpha: float, beta: float, color: Color) -> float:
        """
        The core search function using alpha-beta pruning within the Negamax framework.

        :param board: The current board state.
        :param depth: The current depth of the search.
        :param alpha: The alpha value for alpha-beta pruning.
        :param beta: The beta value for alpha-beta pruning.
        :param color: The color of the current player.
        :return: The best evaluation score for the current player.
        """
        key = get_board_hash(board) % len(self.transposition_table)

        # Check if the position is in the transposition table
        tt_entry = self.transposition_table[key]
        if tt_entry is not None:
            return tt_entry

        if self.time_is_up or depth == 0 or board.is_game_over():
            return self.evaluate_board(board, color)

        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, not color)
            board.pop()
            if score > alpha:
                alpha = score
                self.best_move = move
            if alpha >= beta:
                break

        # Store the position in the transposition table
        self.transposition_table[key] = alpha

        return alpha

    def evaluate_board(self, board: Board, color: Color) -> float:
        """
        Evaluates the given board state and returns a score from the perspective of the current player.

        :param board: The board state to evaluate.
        :param color: The color of the current player.
        :return: The evaluation score.
        """
        score = 0

        for piece in board.piece_map().values():
            score += self.piece_value(piece, color)

        return score

    def piece_value(self, piece: Piece, color: Color) -> float:
        """
        Determines the value of the given piece from the perspective of the current player.

        :param piece: The piece to evaluate.
        :param color: The color of the current player.
        :return: The value of the piece.
        """
        if piece.color == color:
            return PIECE_VALUES[piece.piece_type]
        else:
            return -PIECE_VALUES[piece.piece_type]
