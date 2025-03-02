from __future__ import annotations
import random
import time

import numpy as np
from collections import Counter
from dataclasses import dataclass

from src.util import lerp
from src.util.remove_repetitions import remove_repetitions
from src.mcts.MCTS import MCTS, action_probabilities
from src.mcts.MCTSNode import MCTSNode
from src.cluster.InferenceClient import InferenceClient
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove, log_text
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

        self.self_play_games: Counter[SelfPlayGame] = Counter()
        self.self_play_games[SelfPlayGame()] += self.args.num_parallel_games
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

        current_self_play_games = list(self.self_play_games.items())
        for (spg, count), mcts_result in zip(current_self_play_games, mcts_results):
            spg.memory.append(SelfPlayGameMemory(spg.board.copy(), mcts_result.visit_counts, mcts_result.result_score))

            if (
                mcts_result.result_score < self.args.resignation_threshold
                # TODO or len(spg.played_moves) >= 250  # some max game length
            ):
                # Resignation if most of the mcts searches result in a loss
                self.self_play_games[spg] = 0
                self._add_training_data(spg, mcts_result.result_score, resignation=True)
                self.self_play_games[SelfPlayGame()] += count
                continue

            if len(spg.played_moves) >= 250:
                # If the game is too long, end it and add it to the dataset
                self.self_play_games[spg] = 0
                log('Game too long:', len(spg.played_moves), spg.played_moves)
                self.self_play_games[SelfPlayGame()] += count
                continue

            for _ in range(count):
                self.self_play_games[spg] -= 1

                spg_action_probabilities = action_probabilities(mcts_result.visit_counts)

                while np.sum(spg_action_probabilities) > 0:
                    new_spg, move = self._sample_self_play_game(spg, spg_action_probabilities, mcts_result.children)

                    if self.self_play_games[new_spg] == 0 and move not in spg.played_moves[-16:]:
                        # don't play the same move twice in a row
                        self.self_play_games[new_spg] += 1
                        break
                    else:
                        # Already exploring this state, so remove the probability of this move and try again
                        spg_action_probabilities[CurrentGame.encode_move(move)] = 0

                else:
                    # No valid moves left which are not already being explored
                    # Therefore simply pick the most likely move, and expand to different states from the most likely next state in the next iteration
                    new_spg, _ = self._sample_self_play_game(
                        spg, action_probabilities(mcts_result.visit_counts), mcts_result.children
                    )
                    self.self_play_games[new_spg] += 1

        # remove spgs with count 0
        self.self_play_games = self.self_play_games - Counter()
        assert all(count > 0 for count in self.self_play_games.values())

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
            move = CurrentGame.decode_move(np.argmax(action_probabilities).item())
        else:
            move = self._sample_move(action_probabilities, self.args.temperature)

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

    def _sample_move(self, action_probabilities: np.ndarray, temperature: float = 1.0) -> CurrentGameMove:
        assert temperature > 0, 'Temperature must be greater than 0'

        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)

        action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

        return CurrentGame.decode_move(action)

    @timeit
    def _add_training_data(self, spg: SelfPlayGame, game_outcome: float, resignation: bool) -> None:
        # result: 1 if current player won, -1 if current player lost, 0 if draw

        if random.random() < 0.01:
            # log a game every 1% of the time
            self._log_game(spg, game_outcome)

        self.dataset.add_generation_stats(
            num_games=1,
            generation_time=time.time() - spg.start_generation_time,
            resignation=resignation,
        )

        if game_outcome == 0.0:
            # if the outcome is 0.0 (draw), remove repetitions from the moves
            indices_to_keep = remove_repetitions(spg.played_moves)
            spg.played_moves = [spg.played_moves[i] for i in indices_to_keep]
            spg.memory = [spg.memory[i] for i in indices_to_keep]

        for mem in spg.memory[::-1]:
            encoded_board = CurrentGame.get_canonical_board(mem.board)
            turn_game_outcome = game_outcome if mem.board.current_player == spg.board.current_player else -game_outcome

            for board, visit_counts in CurrentGame.symmetric_variations(encoded_board, mem.visit_counts):
                self.dataset.add_sample(
                    board.astype(np.int8).copy(),
                    self._preprocess_visit_counts(visit_counts),
                    lerp(turn_game_outcome, mem.result_score, self.args.result_score_weight),
                )

            # TODO? game_outcome *= 0.98  # discount the game outcome for each move

    def _preprocess_visit_counts(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        total_visits = sum(visit_count for _, visit_count in visit_counts)
        # Remove moves with less than 1% of the total visits
        visit_counts = [(move, count) for move, count in visit_counts if count >= total_visits * 0.01]

        return visit_counts

    def _log_game(self, spg: SelfPlayGame, result: float) -> None:
        moves = ','.join([str(CurrentGame.encode_move(move)) for move in spg.played_moves])
        log_text(f'moves/{self.iteration}/{hash(moves)}', f'{result}:{moves}')
