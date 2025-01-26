# This bot is heavily based on the King Gambot IV by toanth (https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results/blob/main/Bots/Bot_628.cs)

from math import log
import random

from chess import *  # type: ignore
from src.eval.Bot import Bot

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.comparison_bots.util import *


GAME_PHASE_WEIGHTS = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 4, QUEEN: 5, KING: 0}

MID_GAME_PIECE_SCORES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}
END_GAME_PIECE_SCORES = {PAWN: 200, KNIGHT: 280, BISHOP: 320, ROOK: 600, QUEEN: 900, KING: 0}

MID_GAME_PIECE_SQUARE_TABLES = {
    PAWN: [
        [0, 0, 76, 60, 142, 11, 19, 25],
        [68, 104, 162, 145, 165, 212, 122, 130],
        [113, 150, 156, 180, 215, 255, 172, 162],
        [98, 111, 136, 165, 137, 170, 116, 136],
        [83, 102, 115, 114, 126, 122, 127, 92],
        [64, 86, 103, 104, 118, 105, 109, 77],
        [57, 60, 82, 93, 93, 97, 89, 88],
        [4, 67, 45, 63, 65, 81, 65, 32],
    ],
    KNIGHT: [
        [60, 35, 30, 0, 3, 25, 92, 37],
        [65, 100, 76, 65, 87, 133, 90, 95],
        [77, 105, 122, 129, 128, 152, 142, 97],
        [64, 79, 107, 125, 114, 111, 77, 75],
        [60, 74, 82, 103, 105, 80, 82, 61],
        [73, 83, 80, 83, 83, 82, 78, 90],
        [76, 78, 88, 65, 74, 92, 105, 79],
        [46, 70, 57, 52, 53, 49, 67, 65],
    ],
    BISHOP: [
        [133, 137, 147, 153, 178, 198, 210, 212],
        [82, 73, 100, 122, 103, 135, 136, 185],
        [59, 77, 79, 94, 117, 124, 182, 120],
        [34, 48, 56, 75, 76, 78, 82, 74],
        [10, 15, 27, 45, 49, 33, 52, 26],
        [1, 19, 25, 31, 39, 37, 70, 40],
        [0, 18, 33, 36, 41, 47, 64, 7],
        [18, 23, 34, 56, 53, 45, 35, 0],
    ],
    ROOK: [
        [117, 153, 193, 217, 225, 255, 244, 192],
        [128, 106, 113, 102, 105, 182, 147, 240],
        [131, 128, 135, 146, 160, 192, 213, 185],
        [116, 117, 120, 122, 127, 139, 142, 147],
        [112, 117, 114, 121, 122, 125, 132, 136],
        [113, 124, 116, 119, 121, 123, 140, 133],
        [112, 119, 130, 136, 130, 144, 154, 168],
        [108, 98, 111, 125, 114, 102, 98, 86],
    ],
    QUEEN: [
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255],
        [250, 250, 250, 250, 250, 250, 250, 250],
        [200, 200, 200, 200, 200, 200, 200, 200],
        [100, 100, 100, 100, 100, 100, 100, 100],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [0, 0, 0, 0, 5, 0, 0, 0],
    ],
    KING: [
        [-20, -10, 0, 0, 0, 0, -10, -20],
        [0, 10, 10, 10, 10, 10, 10, 0],
        [10, 20, 20, 20, 20, 20, 20, 10],
        [20, 30, 30, 40, 40, 30, 30, 20],
        [30, 40, 40, 50, 50, 40, 40, 30],
        [30, 40, 40, 50, 50, 40, 40, 30],
        [30, 40, 40, 50, 50, 40, 40, 30],
        [30, 40, 40, 50, 50, 40, 40, 30],
    ],
}

END_GAME_PIECE_SQUARE_TABLES = {
    PAWN: [
        [34, 83, 98, 99, 91, 87, 95, 0],
        [90, 105, 103, 113, 101, 86, 95, 70],
        [96, 109, 133, 129, 108, 103, 99, 80],
        [109, 136, 144, 149, 149, 140, 130, 100],
        [112, 123, 148, 151, 153, 141, 124, 109],
        [94, 116, 127, 144, 142, 121, 112, 96],
        [84, 107, 115, 118, 116, 117, 103, 97],
        [79, 60, 99, 105, 102, 100, 64, 68],
    ],
    KNIGHT: [
        [22, 38, 35, 47, 45, 35, 24, 22],
        [12, 32, 39, 41, 37, 23, 39, 13],
        [43, 32, 44, 35, 38, 42, 27, 37],
        [37, 58, 47, 61, 55, 48, 53, 35],
        [33, 49, 62, 52, 52, 56, 47, 25],
        [31, 43, 50, 50, 56, 50, 36, 25],
        [26, 23, 27, 42, 47, 33, 35, 9],
        [8, 28, 6, 35, 31, 34, 14, 0],
    ],
    BISHOP: [
        [27, 33, 41, 38, 30, 22, 16, 12],
        [30, 50, 48, 39, 40, 34, 27, 2],
        [35, 38, 38, 35, 23, 15, 5, 6],
        [39, 36, 44, 38, 25, 18, 15, 11],
        [34, 38, 41, 36, 34, 33, 20, 20],
        [29, 27, 27, 31, 29, 20, 1, 4],
        [23, 24, 27, 30, 21, 17, 3, 14],
        [16, 29, 36, 33, 27, 23, 19, 0],
    ],
    ROOK: [
        [172, 172, 183, 175, 164, 162, 145, 171],
        [145, 188, 223, 244, 255, 219, 243, 166],
        [154, 172, 206, 210, 233, 207, 156, 166],
        [160, 186, 202, 221, 239, 216, 209, 189],
        [158, 187, 193, 215, 211, 199, 190, 177],
        [151, 149, 186, 177, 186, 186, 162, 149],
        [146, 147, 135, 144, 155, 127, 98, 55],
        [139, 137, 139, 128, 146, 146, 117, 105],
    ],
    QUEEN: [
        [0, 50, 57, 93, 81, 88, 84, 8],
        [84, 110, 119, 107, 122, 133, 132, 105],
        [98, 115, 133, 140, 142, 137, 138, 113],
        [88, 121, 137, 150, 150, 146, 138, 116],
        [76, 108, 132, 148, 148, 137, 124, 108],
        [70, 94, 116, 128, 129, 121, 102, 89],
        [51, 79, 93, 105, 108, 98, 79, 60],
        [20, 36, 56, 76, 49, 74, 46, 17],
    ],
    KING: [
        [-50, -30, -30, -30, -30, -30, -30, -50],
        [-30, -30, 0, 0, 0, 0, -30, -30],
        [-30, -10, 20, 30, 30, 20, -10, -30],
        [-30, -10, 30, 40, 40, 30, -10, -30],
        [-30, -10, 30, 40, 40, 30, -10, -30],
        [-30, -10, 20, 30, 30, 20, -10, -30],
        [-30, -20, -10, 0, 0, -10, -20, -30],
        [-50, -40, -30, -20, -20, -30, -40, -50],
    ],
}


class HandcraftedBotV4(Bot):
    def __init__(self) -> None:
        super().__init__('HandcraftedBotV4', max_time_to_think=MAX_TIME_TO_THINK)

        self.transposition_table = [TranspositionEntry()] * 0x10_0000  # 0x80_0000 in the original bot
        self.best_root_move = Move.null()

    def think(self, board: ChessBoard) -> Move:
        self.initialize_search_parameters()
        self.iterative_deepening_search(board.board)
        if self.best_root_move == Move.null():
            print('WARNING: No move found, returning random move')
            return random.choice(board.get_valid_moves())
        return self.best_root_move

    def initialize_search_parameters(self) -> None:
        self.best_root_move = Move.null()
        self.history = [[[0 for _ in range(64)] for _ in range(7)] for _ in range(2)]  # shape: (2, 7, 64)
        self.killers = [ExtendedMove.null() for _ in range(256)]  # shape: (256)

    def iterative_deepening_search(self, board: Board) -> None:
        alpha, beta = -30000, 30000
        depth = 1
        while not self.time_is_up and depth < 64:
            score = self.negamax(board, depth, alpha, beta, 0, False)
            if score == alpha and score == beta:
                depth += 1
            alpha, beta = self.adjust_search_window(alpha, beta, score)

    def adjust_search_window(self, alpha: int, beta: int, score: int) -> tuple[int, int]:
        new_alpha = min(alpha, score) - 20
        new_beta = max(beta, score) + 20
        return new_alpha, new_beta

    def negamax(
        self, board: Board, depth: int, alpha: int, beta: int, half_move_count: int, allow_null_move_pruning: bool
    ) -> int:
        # Half move count is the number of plays (either by white or black)
        if board.is_repetition(2):
            return 0

        tt_index = self.get_transposition_table_index(board)
        tt_entry = self.transposition_table[tt_index]

        baseline_evaluation_score = self.calculate_baseline_evaluation_score(board, tt_entry, tt_index)

        if board.is_check():
            depth += 1

        best_score = -32000
        # Quiescence search is a special kind of search that only considers captures and checks
        in_quiescence_search = depth <= 0

        if in_quiescence_search:
            best_score = baseline_evaluation_score
            alpha = max(alpha, best_score)

            if alpha >= beta:
                return baseline_evaluation_score

        if self.should_use_tt_entry_score_as_search_result(depth, alpha, beta, tt_index, tt_entry):
            return tt_entry.score

        if depth > 3 and tt_entry.move == Move.null():
            depth -= 1

        if self.is_allowed_pruning(board, alpha, beta):
            if not in_quiescence_search and depth < 5 and baseline_evaluation_score >= beta + 64 * depth:
                return baseline_evaluation_score

            if depth >= 4 and allow_null_move_pruning and baseline_evaluation_score >= beta:
                board.push(Move.null())
                played_move_score = -self.negamax(
                    board, depth - 3 - depth // 4, -beta, -alpha, half_move_count + 2, False
                )
                board.pop()

                if played_move_score >= beta:
                    return played_move_score

        sorted_moves = self.calculate_sorted_moves(board, half_move_count, tt_entry)

        if not sorted_moves:
            if in_quiescence_search:
                return best_score
            elif board.is_check():
                return -30000 + half_move_count
            else:
                return 0

        best_move = tt_entry.move

        for move_index, move in enumerate(sorted_moves):
            is_move_uninteresting = move.score > -250_000_000

            if self.stop_evaluating_moves(
                board, depth, alpha, beta, baseline_evaluation_score, best_score, move_index, is_move_uninteresting
            ):
                break

            if self.time_is_up:
                return 30999

            played_move_score = self.search_extend(
                board, depth, alpha, beta, move_index, move, is_move_uninteresting, half_move_count
            )

            best_score = max(best_score, played_move_score)
            if played_move_score < alpha:
                continue

            best_move = move.move
            alpha = played_move_score

            if half_move_count == 0:
                self.best_root_move = best_move

            if played_move_score < beta:
                continue

            if move.is_capture:
                break

            if move != self.killers[half_move_count]:
                self.killers[half_move_count + 1] = self.killers[half_move_count]
                self.killers[half_move_count] = move

            self.history[board.turn][move.move_piece_type][move.to_square] += 2**depth

            break

        self.save_to_transposition_table(depth, alpha, beta, tt_index, best_score, best_move)

        return best_score

    def search_extend(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        move_index: int,
        move: ExtendedMove,
        is_move_uninteresting: bool,
        half_move_count: int,
    ) -> int:
        def search(minus_new_alpha: int, reduction: int, allow_null_move_pruning: bool) -> int:
            return -self.negamax(
                board,
                depth - reduction,
                -minus_new_alpha,
                -alpha,
                half_move_count + 2,
                allow_null_move_pruning,
            )

        passed_pawn_extension = move.move_piece_type == PAWN and square_rank(move.to_square) % 5 == 1

        reduction = self.calculate_reduction_value(depth, alpha, beta, move_index, is_move_uninteresting)

        board.push(move.move)

        if move_index == 0:
            played_move_score = search(beta, 1 - passed_pawn_extension, True)
        else:
            played_move_score = search(alpha - 1, reduction - passed_pawn_extension, True)
            if alpha < played_move_score < beta:
                played_move_score = search(beta, 1 - passed_pawn_extension, True)

        board.pop()

        return played_move_score

    def save_to_transposition_table(
        self, depth: int, alpha: int, beta: int, tt_index: int, best_score: int, best_move: Move
    ) -> None:
        self.transposition_table[tt_index] = TranspositionEntry(
            hash=tt_index,
            move=best_move,
            score=best_score,
            depth=depth,
            flag=self.determine_flag_for_tt_entry(best_score, alpha, beta),
        )

    def calculate_reduction_value(
        self, depth: int, alpha: int, beta: int, move_index: int, is_move_uninteresting: bool
    ) -> int:
        if move_index < (4 - self.is_non_principal_variation_node(alpha, beta)) or depth <= 3 or is_move_uninteresting:
            return 1

        reduction_value = min(
            int(1.0 + log(depth) * log(move_index) / 2.36) + (not self.is_non_principal_variation_node(alpha, beta)),
            depth,
        )
        return reduction_value

    def stop_evaluating_moves(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        baseline_evaluation_score: int,
        best_score: int,
        move_index: int,
        is_move_uninteresting: bool,
    ) -> bool:
        return (
            depth <= 5
            and best_score > -29000
            and self.is_allowed_pruning(board, alpha, beta)
            and (
                is_move_uninteresting
                and baseline_evaluation_score + 300 + 64 * depth < alpha
                or move_index > 7 + depth**2
            )
        )

    def is_allowed_pruning(self, board: Board, alpha: int, beta: int) -> bool:
        return self.is_non_principal_variation_node(alpha, beta) and not board.is_check()

    def should_use_tt_entry_score_as_search_result(
        self, depth: int, alpha: int, beta: int, tt_index: int, tt_entry: TranspositionEntry
    ) -> bool:
        if (
            tt_entry.depth < depth
            or not self.trust_tt_entry(tt_entry, tt_index)
            or not self.is_non_principal_variation_node(alpha, beta)
        ):
            return False

        if tt_entry.score >= beta:
            return tt_entry.flag != TranspositionFlag.LOWER_BOUND

        return tt_entry.flag != TranspositionFlag.EXACT

    def is_non_principal_variation_node(self, alpha: int, beta: int) -> bool:
        # Principal variation is the sequence of moves that the engine considers the best, a non principal variation node is a node that is not one of the best moves
        return alpha + 1 >= beta

    def get_transposition_table_index(self, board: Board) -> int:
        return get_board_hash(board) % len(self.transposition_table)

    def calculate_move_scores(
        self, turn: Color, half_move_count: int, tt_entry: TranspositionEntry, legal_moves: list[ExtendedMove]
    ) -> list[int]:
        return [self.calculate_move_score(turn, half_move_count, tt_entry, move) for move in legal_moves]

    def calculate_move_score(
        self, turn: Color, half_move_count: int, tt_entry: TranspositionEntry, move: ExtendedMove
    ) -> int:
        if move == tt_entry.move:
            return 2_000_000_000
        elif move.is_capture:
            return move.capture_piece_type * 268_435_456 - move.move_piece_type  # type: ignore
        elif move == self.killers[half_move_count] or move == self.killers[half_move_count + 1]:
            return 250_000_000
        else:
            return self.history[turn][move.move_piece_type][move.to_square]

    def calculate_sorted_moves(
        self, board: Board, half_move_count: int, tt_entry: TranspositionEntry
    ) -> list[ExtendedMove]:
        legal_moves = get_legal_moves(board)

        move_scores = self.calculate_move_scores(board.turn, half_move_count, tt_entry, legal_moves)

        for move, move_score in zip(legal_moves, move_scores):
            move.score = move_score

        return list(sorted(legal_moves, key=lambda x: x.score, reverse=True))

    def calculate_baseline_evaluation_score(self, board: Board, tt_entry: TranspositionEntry, tt_index: int) -> int:
        if self.trust_tt_entry(tt_entry, tt_index):
            return tt_entry.score

        game_phase = 0  # opening, midgame, endgame (0-32)
        mid_game_score = 7
        end_game_score = 7

        for color in (board.turn, not board.turn):
            (
                num_doubled_pawns,
                game_phase_increment,
                mid_game_increment,
                end_game_increment,
            ) = self.calculate_piece_scores(board, color)

            game_phase += game_phase_increment
            mid_game_score += mid_game_increment
            end_game_score += end_game_increment

            mid_game_score = num_doubled_pawns * 9 - mid_game_score
            end_game_score = num_doubled_pawns * 32 - end_game_score

        baseline_evaluation_score = (mid_game_score * game_phase + end_game_score * (32 - game_phase)) // 32
        return baseline_evaluation_score

    def calculate_piece_scores(self, board: Board, color: Color) -> tuple[int, int, int, int]:
        game_phase_increment = mid_game_increment = end_game_increment = 0

        for piece in PIECE_TYPES:
            for square in board.pieces(piece, color):
                # Apply piece-square table adjustments
                piece_square_table_square = square if color == BLACK else square_mirror(square)
                piece_square_table_x = square_file(piece_square_table_square)
                piece_square_table_y = square_rank(piece_square_table_square)

                mid_game_increment += MID_GAME_PIECE_SQUARE_TABLES[piece][piece_square_table_y][piece_square_table_x]
                end_game_increment += END_GAME_PIECE_SQUARE_TABLES[piece][piece_square_table_y][piece_square_table_x]

                mid_game_increment += MID_GAME_PIECE_SCORES[piece]
                end_game_increment += END_GAME_PIECE_SCORES[piece]

                game_phase_increment += GAME_PHASE_WEIGHTS[piece]

        pawns = board.pieces_mask(PAWN, color)
        num_doubled_pawns = get_number_of_set_bits(pawns & pawns << 8)

        if 0x0101_0101_0101_0101 << self.get_king_file(board, color) & pawns:  # TODO what is this
            mid_game_increment -= 40

        return num_doubled_pawns, game_phase_increment, mid_game_increment, end_game_increment

    def get_king_file(self, board: Board, color: Color) -> int:
        return square_file(board.king(color) or 0)

    def trust_tt_entry(self, tt_entry: TranspositionEntry, tt_index: int) -> bool:
        return tt_entry.hash == tt_index

    def determine_flag_for_tt_entry(self, best_score: int, alpha: int, beta: int) -> TranspositionFlag:
        # Determine the flag for the transposition table entry based on the search results.
        if best_score <= alpha:
            return TranspositionFlag.UPPER_BOUND
        elif best_score >= beta:
            return TranspositionFlag.LOWER_BOUND
        return TranspositionFlag.EXACT
