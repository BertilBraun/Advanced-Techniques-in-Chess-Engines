# This bot is heavily based on the King Gambot IV by toanth (https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results/blob/main/Bots/Bot_628.cs)

import numpy as np


from chess import *  # type: ignore
from src.eval.Bot import Bot

from py.src.games.chess.comparison_bots.util import *


class HandcraftedBotV3(Bot):
    def __init__(self) -> None:
        super().__init__('HandcraftedBotV3', max_time_to_think=MAX_TIME_TO_THINK)

        self.transposition_table = [TranspositionEntry()] * 2**16
        self.best_root_move = Move.null()

    def think(self, board: Board) -> Move:
        self.initialize_search_parameters()
        self.iterative_deepening_search(board)
        return self.best_root_move

    def initialize_search_parameters(self) -> None:
        self.best_root_move = Move.null()
        self.history = np.zeros((2, 7, 64), dtype=int)
        self.killers = [ExtendedMove.null() for _ in range(256)]

    def iterative_deepening_search(self, board: Board) -> None:
        alpha, beta = -30000, 30000
        depth = 1
        while not self.time_is_up and depth < 64:
            score = self.negamax(board, depth, alpha, beta, 0, False)
            alpha, beta = self.adjust_search_window(alpha, beta, score)
            depth += 1

    def adjust_search_window(self, alpha: int, beta: int, score: int) -> tuple[int, int]:
        new_alpha = min(alpha, score) - 20
        new_beta = max(beta, score) + 20
        return new_alpha, new_beta

    def negamax(self, board: Board, depth: int, alpha: int, beta: int, half_ply: int, allow_nmp: bool) -> int:
        print(f'Depth: {depth}')
        if board.is_repetition(2):
            print('Repetition')
            return 0

        transposition_index = self.get_transposition_index(board)
        tt_entry = self.transposition_table[transposition_index]

        if self.should_stop_search(tt_entry, transposition_index, alpha, beta):
            print('TT cutoff')
            return tt_entry.score

        moves, move_scores = self.generate_and_score_moves_sorted(board, half_ply, tt_entry.move)
        if depth <= 0 or board.is_game_over():
            print('Leaf node')
            return move_scores[0] if moves else 0

        best_score, alpha = self.search_moves(board, moves, move_scores, depth, alpha, beta, half_ply, allow_nmp)

        self.update_transposition_table(transposition_index, best_score, depth, alpha, beta, tt_entry.move)
        print(f'Best score: {best_score}')
        return best_score

    def get_transposition_index(self, board: Board) -> int:
        # Calculate and return the transposition table index for the current board state
        return get_board_hash(board) % len(self.transposition_table)

    def is_tt_entry_reliable(self, tt_entry: TranspositionEntry, transposition_index: int) -> bool:
        # Determine if the transposition table entry is reliable for use in the current search
        return tt_entry.hash == transposition_index

    def should_stop_search(self, tt_entry: TranspositionEntry, transposition_index: int, alpha: int, beta: int) -> bool:
        trust_tt = self.is_tt_entry_reliable(tt_entry, transposition_index)

        # Determine if the current search can be stopped early based on transposition table data and search parameters
        if not trust_tt:
            return False
        if tt_entry.flag == TranspositionFlag.EXACT:
            return True
        if tt_entry.flag == TranspositionFlag.LOWER_BOUND and tt_entry.score < alpha:
            return True
        if tt_entry.flag == TranspositionFlag.UPPER_BOUND and tt_entry.score > beta:
            return True
        return False

    def generate_and_score_moves_sorted(
        self, board: Board, half_ply: int, best_move_from_tt: Move
    ) -> tuple[list[ExtendedMove], list[int]]:
        # Generate legal moves and score them for move ordering
        moves = get_legal_moves(board)
        move_scores = [self.score_move(move, board, half_ply, best_move_from_tt) for move in moves]

        sorted_indices = np.argsort(-np.array(move_scores))

        return [moves[i] for i in sorted_indices], [move_scores[i] for i in sorted_indices]

    def score_move(self, move: ExtendedMove, board: Board, half_ply: int, best_move_from_tt: Move) -> int:
        # Score a given move for move ordering.
        if move == best_move_from_tt:
            return 2_000_000_000  # High priority for TT suggested move
        elif move.is_capture:
            # Prioritize captures by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            return 1_000_000_000 + move.move_piece_type * 10 - move.capture_piece_type if move.capture_piece_type else 0
        elif move in (self.killers[half_ply], self.killers[half_ply + 1]):
            return 500_000_000  # High priority for killer moves
        else:
            # Use historical heuristics for other moves
            return self.history[board.turn][move.move_piece_type][move.to_square]

    def search_moves(
        self,
        board: Board,
        moves: list[ExtendedMove],
        move_scores: list[int],
        depth: int,
        alpha: int,
        beta: int,
        half_ply: int,
        allow_nmp: bool,
    ) -> tuple[int, int]:
        # Search through all generated moves, applying negamax search and updating alpha. Return the best score and alpha.
        best_score = -999999
        for move, score in zip(moves, move_scores):
            if self.time_is_up:  # Check if the search should be terminated early
                break

            board.push(move.move)

            # Apply LMR (Late Move Reductions) based on certain conditions
            reduction = 1 if (score < 500_000_000 and len(moves) > 4) else 0  # Example condition for reduction
            if reduction:
                provisional_score = -self.negamax(board, depth - 1 - reduction, -beta, -alpha, half_ply + 1, allow_nmp)
                if provisional_score > alpha:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, half_ply + 1, allow_nmp)
                else:
                    score = provisional_score
            else:
                score = -self.negamax(board, depth - 1, -beta, -alpha, half_ply + 1, allow_nmp)

            board.pop()

            if score >= beta:
                return score, beta  # Beta cutoff
            if score > best_score:
                best_score = score
                self.best_root_move = move.move
                if score > alpha:
                    alpha = score  # Update alpha if a better move is found

        return best_score, alpha

    def update_transposition_table(
        self, transposition_index: int, best_score: int, depth: int, alpha: int, beta: int, best_move: Move
    ):
        # Update the transposition table with the results of the current search
        self.transposition_table[transposition_index] = TranspositionEntry(
            hash=transposition_index,
            move=best_move,
            score=best_score,
            depth=depth,
            flag=self.determine_flag_for_tt_entry(best_score, alpha, beta),
        )

    def determine_flag_for_tt_entry(self, best_score: int, alpha: int, beta: int) -> TranspositionFlag:
        # Determine the flag for the transposition table entry based on the search results.
        if best_score <= alpha:
            return TranspositionFlag.UPPER_BOUND
        elif best_score >= beta:
            return TranspositionFlag.LOWER_BOUND
        return TranspositionFlag.EXACT
