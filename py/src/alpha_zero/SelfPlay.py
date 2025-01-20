from __future__ import annotations
import time

import numpy as np
from collections import Counter
from dataclasses import dataclass

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove
from src.Encoding import get_board_result_score
from src.alpha_zero.train.TrainingArgs import SelfPlayParams
from src.util import lerp
from src.util.log import log
from src.util.profiler import reset_times, timeit


@dataclass(frozen=True)
class SelfPlayGameMemory:
    board: CurrentBoard
    action_probabilities: np.ndarray
    result_score: float


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.played_moves: list[CurrentGameMove] = []
        self.num_played_moves = 0
        self.start_generation_time = time.time()

    def expand(self, move: CurrentGameMove) -> SelfPlayGame:
        new_game = self.copy()
        new_game.board.make_move(move)
        new_game.num_played_moves += 1
        new_game.played_moves.append(move)
        return new_game

    def copy(self) -> SelfPlayGame:
        game = SelfPlayGame()
        game.board = self.board.copy()
        game.memory = self.memory.copy()
        game.played_moves = self.played_moves.copy()
        game.num_played_moves = self.num_played_moves
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

        self.mcts = self._get_mcts(self.iteration)

    def update_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        self.mcts = self._get_mcts(self.iteration)
        self.dataset = SelfPlayDataset()
        self.client.update_iteration(iteration)

    @timeit
    async def self_play(self) -> None:
        mcts_results = await self.mcts.search([spg.board for spg in self.self_play_games])

        current_self_play_games = list(self.self_play_games.items())
        for (spg, count), (action_probabilities, result_score) in zip(current_self_play_games, mcts_results):
            spg.memory.append(SelfPlayGameMemory(spg.board.copy(), action_probabilities.copy(), result_score))

            if result_score < self.args.resignation_threshold:
                # Resignation if most of the mcts searches result in a loss
                self.self_play_games[spg] = 0
                self._add_training_data(spg, result_score)
                self.self_play_games[SelfPlayGame()] += count
                continue

            for _ in range(count):
                self.self_play_games[spg] -= 1

                spg_action_probabilities = action_probabilities.copy()

                while np.sum(spg_action_probabilities) > 0:
                    new_spg, move = self._sample_self_play_game(spg, spg_action_probabilities)

                    if self.self_play_games[new_spg] > 0:
                        # Already exploring this state, so remove the probability of this move and try again
                        spg_action_probabilities[CurrentGame.encode_move(move)] = 0
                        continue

                    self.self_play_games[new_spg] += 1
                    break
                else:
                    # No valid moves left which are not already being explored
                    # Therefore simply pick the most likely move, and expand to different states from the most likely next state in the next iteration
                    new_spg, _ = self._sample_self_play_game(spg, action_probabilities)
                    self.self_play_games[new_spg] += 1

        # remove spgs with count 0
        self.self_play_games = self.self_play_games - Counter()
        assert all(count > 0 for count in self.self_play_games.values())

        reset_times()
        log(
            'Cache hit rate:',
            self.client.total_hits / self.client.total_evals,
            'on',
            self.client.total_evals,
            'evaluations',
        )

    @timeit
    def _sample_self_play_game(
        self, current: SelfPlayGame, action_probabilities: np.ndarray
    ) -> tuple[SelfPlayGame, CurrentGameMove]:
        # Sample a move from the action probabilities then create a new game state with that move
        # If the game is over, add the game to the dataset and return a new game state, thereby initializing a new game

        # only use temperature for the first X moves, then simply use the most likely move
        # Keep exploration high for the first X moves, then play out as well as possible to reduce noise in the backpropagated final game results
        if current.num_played_moves >= self.args.num_moves_after_which_to_play_greedy:
            move = CurrentGame.decode_move(np.argmax(action_probabilities).item())
        else:
            move = self._sample_move(action_probabilities, self.args.temperature)

        new_spg = current.expand(move)

        if not new_spg.board.is_game_over():
            return new_spg, move

        # Game is over, add the game to the dataset
        result = get_board_result_score(new_spg.board)
        assert result is not None, 'Game should not be over if result is None'
        self._add_training_data(new_spg, result)
        return SelfPlayGame(), move

    def _get_mcts(self, iteration: int) -> MCTS:
        return MCTS(
            self.client,
            MCTSArgs(
                num_searches_per_turn=self.args.mcts.num_searches_per_turn,
                num_parallel_searches=self.args.mcts.num_parallel_searches,
                dirichlet_epsilon=self.args.mcts.dirichlet_epsilon,
                dirichlet_alpha=self.args.mcts.dirichlet_alpha(iteration),
                c_param=self.args.mcts.c_param,
            ),
        )

    def _sample_move(self, action_probabilities: np.ndarray, temperature: float = 1.0) -> CurrentGameMove:
        assert temperature > 0, 'Temperature must be greater than 0'

        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)

        action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

        return CurrentGame.decode_move(action)

    def _add_training_data(self, spg: SelfPlayGame, result: float) -> None:
        # result: 1 if current player won, -1 if current player lost, 0 if draw

        log(f'Adding training data for game with result {result}')
        log(f'Game moves: {spg.played_moves}')

        self.dataset.add_generation_stats(num_games=1, generation_time=time.time() - spg.start_generation_time)

        for mem in spg.memory[::-1]:  # reverse to flip the result for the other player
            encoded_board = CurrentGame.get_canonical_board(mem.board)

            for board, probabilities in CurrentGame.symmetric_variations(encoded_board, mem.action_probabilities):
                self.dataset.add_sample(
                    board.copy().astype(np.int8),
                    probabilities.copy().astype(np.float32),
                    lerp(result, mem.result_score, self.args.result_score_weight),
                )
            result = -result
