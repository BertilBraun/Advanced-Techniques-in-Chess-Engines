from __future__ import annotations
import random
import time

import chess
import numpy as np
from dataclasses import dataclass

from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.util import lerp
from src.mcts.MCTS import MCTS, action_probabilities
from src.mcts.MCTSNode import MCTSNode
from src.cluster.InferenceClient import InferenceClient
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import CURRENT_GAME, CurrentBoard, CurrentGame, CurrentGameMove, log_text
from src.Encoding import get_board_result_score
from src.train.TrainingArgs import SelfPlayParams
from src.util.log import log
from src.util.timing import reset_times, timeit


@dataclass(frozen=True)
class SelfPlayGameMemory:
    board: CurrentBoard
    visit_counts: list[tuple[int, int]]
    result_score: float


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.played_moves: list[CurrentGameMove] = []
        self.already_expanded_node: MCTSNode | None = None
        self.start_generation_time = time.time()

    def expand(self, move: CurrentGameMove) -> SelfPlayGame:
        new_game = self.copy()
        new_game.board.make_move(move)
        new_game.played_moves.append(move)
        return new_game

    def copy(self) -> SelfPlayGame:
        game = SelfPlayGame()
        game.board = self.board.copy()
        game.memory = self.memory.copy()
        game.played_moves = self.played_moves.copy()
        game.start_generation_time = self.start_generation_time
        return game

    def __hash__(self) -> int:
        return self.board.quick_hash()


class SelfPlay:
    def __init__(self, client: InferenceClient, args: SelfPlayParams) -> None:
        self.client = client
        self.args = args

        self.self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0

        self.mcts = MCTS(self.client, self.args.mcts)

    def update_iteration(self, iteration: int) -> None:
        if len(self.dataset) > 0:
            log(f'Warning: Dataset should be empty when updating iteration. Discarding {len(self.dataset)} samples.')
        self.iteration = iteration
        self.dataset = SelfPlayDataset()
        self.client.update_iteration(iteration)

    def self_play(self) -> None:
        mcts_results = self.mcts.search([(spg.board, spg.already_expanded_node) for spg in self.self_play_games])

        for i, (spg, mcts_result) in enumerate(zip(self.self_play_games, mcts_results)):
            if mcts_result.is_full_search:
                spg.memory.append(
                    SelfPlayGameMemory(spg.board.copy(), mcts_result.visit_counts, mcts_result.result_score)
                )

            if mcts_result.result_score < self.args.resignation_threshold:
                # Resignation if most of the mcts searches result in a loss
                self._add_training_data(spg, mcts_result.result_score, resignation=True)
                self.self_play_games[i] = SelfPlayGame()
                continue

            if len(spg.played_moves) >= 250 and CURRENT_GAME == 'chess':
                # If the game is too long, end it and add it to the dataset
                pieces = list(spg.board.board.piece_map().values())
                white_pieces = sum(1 for piece in pieces if piece.color == chess.WHITE)
                black_pieces = sum(1 for piece in pieces if piece.color == chess.BLACK)
                if white_pieces < 4 or black_pieces < 4:
                    # If there are only a few pieces left, the game is a win for the player with more pieces
                    value_map = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                        chess.KING: 0,
                    }
                    white_value = sum(value_map[piece.piece_type] for piece in pieces if piece.color == chess.WHITE)
                    black_value = sum(value_map[piece.piece_type] for piece in pieces if piece.color == chess.BLACK)

                    # Convert to result from current player's perspective
                    if spg.board.current_player == 1:  # White's perspective
                        game_outcome = 1.0 if white_value > black_value else -1.0 if black_value > white_value else 0.0
                    else:  # Black's perspective
                        game_outcome = 1.0 if black_value > white_value else -1.0 if white_value > black_value else 0.0

                    self._add_training_data(spg, game_outcome, resignation=False)
                self.dataset.stats += SelfPlayDatasetStats(num_too_long_games=1)
                self.self_play_games[i] = SelfPlayGame()
                continue

            spg_action_probabilities = action_probabilities(mcts_result.visit_counts)

            while np.sum(spg_action_probabilities) > 0:
                new_spg, move = self._sample_self_play_game(spg, spg_action_probabilities, mcts_result.children)

                is_duplicate = any(hash(game) == hash(new_spg) for game in self.self_play_games)
                is_repetition = move in spg.played_moves[-16:]
                if is_duplicate or is_repetition:
                    # don't play the same move twice in a row
                    # Already exploring this state, so remove the probability of this move and try again
                    spg_action_probabilities[CurrentGame.encode_move(move, spg.board)] = 0
                else:
                    self.self_play_games[i] = new_spg
                    break

            else:
                # No valid moves left which are not already being explored
                # Therefore simply pick the most likely move, and expand to different states from the most likely next state in the next iteration
                new_spg, _ = self._sample_self_play_game(
                    spg, action_probabilities(mcts_result.visit_counts), mcts_result.children
                )
                self.self_play_games[i] = new_spg

        reset_times()

    @timeit
    def _sample_self_play_game(
        self, current: SelfPlayGame, action_probabilities: np.ndarray, children: list[MCTSNode]
    ) -> tuple[SelfPlayGame, CurrentGameMove]:
        # Sample a move from the action probabilities then create a new game state with that move
        # If the game is over, add the game to the dataset and return a new game state, thereby initializing a new game

        # only use temperature for the first X moves, then simply use the most likely move
        # Keep exploration high for the first X moves, then play out as well as possible to reduce noise in the backpropagated final game results
        if len(current.played_moves) >= self.args.num_moves_after_which_to_play_greedy:
            move = CurrentGame.decode_move(np.argmax(action_probabilities).item(), current.board)
        else:
            move = self._sample_move(action_probabilities, current.board, self.args.temperature)

        new_spg = current.expand(move)
        # TODO encoded_move = CurrentGame.encode_move(move)
        # TODO new_spg.already_expanded_node = next(
        # TODO     child for child in children if child.encoded_move_to_get_here == encoded_move
        # TODO ).copy(parent=None)  # remove parent to avoid memory leaks

        if not new_spg.board.is_game_over():
            return new_spg, move

        # Game is over, add the game to the dataset
        result = get_board_result_score(new_spg.board)
        assert result is not None, 'Game should not be over if result is None'
        self._add_training_data(new_spg, result, resignation=False)
        return SelfPlayGame(), move

    def _sample_move(
        self, action_probabilities: np.ndarray, board: CurrentBoard, temperature: float = 1.0
    ) -> CurrentGameMove:
        assert temperature > 0, 'Temperature must be greater than 0'

        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)

        action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

        return CurrentGame.decode_move(action, board)

    @timeit
    def _add_training_data(self, spg: SelfPlayGame, game_outcome: float, resignation: bool) -> None:
        # result: 1 if current player won, -1 if current player lost, 0 if draw

        if random.random() < 0.01:
            # log a game every 1% of the time
            self._log_game(spg, game_outcome)

        self.dataset.add_generation_stats(
            game_length=len(spg.played_moves),
            generation_time=time.time() - spg.start_generation_time,
            resignation=resignation,
        )

        for mem in spg.memory[::-1]:
            turn_game_outcome = game_outcome if mem.board.current_player == spg.board.current_player else -game_outcome

            for board, visit_counts in CurrentGame.symmetric_variations(mem.board, mem.visit_counts):
                self.dataset.add_sample(
                    board.astype(np.int8).copy(),
                    self._preprocess_visit_counts(visit_counts),
                    # clamp(turn_game_outcome + self.args.result_score_weight * mem.result_score, -1, 1),
                    lerp(turn_game_outcome, mem.result_score, self.args.result_score_weight),
                )

            # TODO disabled for now, discounting only in MCTS search game_outcome *= 0.99  # discount the game outcome for each move

    def _preprocess_visit_counts(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        total_visits = sum(visit_count for _, visit_count in visit_counts)
        # Remove moves with less than 1% of the total visits
        visit_counts = [(move, count) for move, count in visit_counts if count >= total_visits * 0.01]

        return visit_counts

    def _log_game(self, spg: SelfPlayGame, result: float) -> None:
        moves = ','.join([str(CurrentGame.encode_move(move, spg.board)) for move in spg.played_moves])
        log_text(f'moves/{self.iteration}/{hash(moves)}', f'{result}:{moves}')
