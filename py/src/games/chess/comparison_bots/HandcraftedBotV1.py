from chess import *  # type: ignore
from src.eval.Bot import Bot

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.comparison_bots.util import *


class HandcraftedBotV1(Bot):
    """Description:
    A simple chess bot that uses the Negamax algorithm with alpha-beta pruning to search the game tree. The bot uses a simple evaluation function that assigns a value to each piece and sums the values for each player. The bot searches to a fixed depth and selects the move with the highest evaluation score.
    """

    def __init__(self) -> None:
        super().__init__('HandcraftedBotV1', max_time_to_think=MAX_TIME_TO_THINK)

    def think(self, board: ChessBoard) -> Move:
        """
        Determine the best move given the current board state.

        :param board: The current board state.
        :return: The best move as determined by the engine.
        """
        self.best_move = Move.null()

        self.negamax(board.board, 3, -float('inf'), float('inf'), board.board.turn)

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
        if depth == 0 or board.is_game_over():
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

        return alpha

    def evaluate_board(self, board: Board, color: Color) -> float:
        """
        Evaluates the given board state and returns a score from the perspective of the current player.
        Checks for checkmate, stalemate, and check situations to prioritize these conditions in the evaluation.
        """
        # Check for checkmate and stalemate conditions
        if board.is_checkmate():
            # If the current player is in checkmate, return a very low score
            # If the opponent is in checkmate, return a very high score
            return -float('inf') if board.turn == color else float('inf')

        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            # In case of a draw condition, return a neutral score
            return 0

        if board.is_check():
            # Adjust score slightly for being in check - this could be more sophisticated
            # For example, penalizing checks more severely if they lead to material loss or checkmate threats
            check_penalty = -50  # Example penalty value for being in check
            return check_penalty if board.turn == color else -check_penalty

        # Continue with normal evaluation if none of the above conditions are met
        score = 0

        # Piece values and positions
        for square, piece in board.piece_map().items():
            score += self.piece_value(square, piece, color)

        # Further evaluation components like mobility, pawn structure, etc., go here

        return score

    def piece_value(self, square: Square, piece: Piece, color: Color) -> float:
        """
        Determines the value of the given piece from the perspective of the current player.
        """
        value = PIECE_VALUES[piece.piece_type]

        # Apply piece-square table adjustments
        piece_square_table_square = square if piece.color == WHITE else square_mirror(square)
        piece_square_table_x = square_file(piece_square_table_square)
        piece_square_table_y = square_rank(piece_square_table_square)

        value += PIECE_SQUARE_TABLES[piece.piece_type][piece_square_table_y][piece_square_table_x]

        return value if piece.color == color else -value
