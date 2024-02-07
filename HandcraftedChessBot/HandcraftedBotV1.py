import time
from Framework import *


PIECE_VALUES = {
    PAWN: 1,
    KNIGHT: 3,
    BISHOP: 3,
    ROOK: 5,
    QUEEN: 9,
    KING: 0
}


class HandcraftedBotV1(ChessBot):
    """Description:
    A simple chess bot that uses the Negamax algorithm with alpha-beta pruning to search the game tree. The bot uses a simple evaluation function that assigns a value to each piece and sums the values for each player. The bot searches to a fixed depth and selects the move with the highest evaluation score.
    """
    
    def __init__(self) -> None:
        super().__init__("HandcraftedBotV1")
        
    def think(self, board: Board) -> Move:
        """
        Determine the best move given the current board state.
        
        :param board: The current board state.
        :return: The best move as determined by the engine.
        """
        self.start_time = time.time()
        self.best_move = Move.null()

        self.negamax(board, 3, -float('inf'), float('inf'), board.turn)
        
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
            score = -self.negamax(board, depth-1, -beta, -alpha, not color)
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
        
        :param board: The board state to evaluate.
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
