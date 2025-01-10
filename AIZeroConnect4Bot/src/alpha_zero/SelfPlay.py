import numpy as np
from dataclasses import dataclass

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove
from src.Encoding import get_board_result_score
from src.alpha_zero.train.TrainingArgs import SelfPlayParams


@dataclass
class SelfPlayGameMemory:
    board: CurrentBoard
    action_probabilities: np.ndarray


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.num_played_moves = 0


def sample_move(
    action_probabilities: np.ndarray, num_played_moves: int = 0, temperature: float = 1.0
) -> CurrentGameMove:
    # only use temperature for the first 30 moves, then simply use the action probabilities as they are
    if num_played_moves < 30:
        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)
    else:
        temperature_action_probabilities = action_probabilities

    action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

    return CurrentGame.decode_move(action)


class SelfPlay:
    def __init__(self, client: InferenceClient, args: SelfPlayParams) -> None:
        self.client = client
        self.args = args

        self.self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0

        self.mcts = self._get_mcts(self.iteration)

    def update_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        self.mcts = self._get_mcts(self.iteration)
        self.dataset = SelfPlayDataset()

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

    async def self_play(self) -> None:
        mcts_action_probabilities = await self.mcts.search([spg.board for spg in self.self_play_games])

        for spg, action_probabilities in zip(self.self_play_games, mcts_action_probabilities):
            spg.memory.append(SelfPlayGameMemory(spg.board.copy(), action_probabilities))

            move = sample_move(action_probabilities, spg.num_played_moves, self.args.temperature)
            spg.board.make_move(move)
            spg.num_played_moves += 1

            if spg.board.is_game_over():
                self.dataset += self._get_training_data(spg)

        self.self_play_games = [spg for spg in self.self_play_games if not spg.board.is_game_over()]
        num_games_to_restart = self.args.num_parallel_games - len(self.self_play_games)
        self.self_play_games += [SelfPlayGame() for _ in range(num_games_to_restart)]

    def _get_training_data(self, spg: SelfPlayGame) -> SelfPlayDataset:
        self_play_dataset = SelfPlayDataset()

        # 1 if current player won, -1 if current player lost, 0 if draw
        result = get_board_result_score(spg.board)
        assert result is not None, 'Game is not over'

        for mem in spg.memory[::-1]:  # reverse to flip the result for the other player
            encoded_board = CurrentGame.get_canonical_board(mem.board)

            for board, probabilities in CurrentGame.symmetric_variations(encoded_board, mem.action_probabilities):
                self_play_dataset.add_sample(
                    board.copy().astype(np.int8),
                    probabilities.copy().astype(np.float32),
                    result,
                )
            result = -result

        return self_play_dataset
