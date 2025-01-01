import torch
import numpy as np
from dataclasses import dataclass

from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CURRENT_BOARD, CURRENT_GAME, CURRENT_GAME_MOVE
from src.Network import Network
from src.Encoding import get_board_result_score
from src.alpha_zero.train.TrainingArgs import TrainingArgs


@dataclass
class SelfPlayMemory:
    state: torch.Tensor
    policy_targets: torch.Tensor
    value_target: float


@dataclass
class SelfPlayGameMemory:
    board: CURRENT_BOARD
    action_probabilities: np.ndarray


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CURRENT_GAME.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.num_played_moves = 0


def sample_move(
    action_probabilities: np.ndarray, num_played_moves: int = 0, temperature: float = 1.0
) -> CURRENT_GAME_MOVE:
    # only use temperature for the first 30 moves, then simply use the action probabilities as they are
    if num_played_moves < 30:
        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)
    else:
        temperature_action_probabilities = action_probabilities

    action = np.random.choice(CURRENT_GAME.action_size, p=temperature_action_probabilities)

    return CURRENT_GAME.decode_move(action)


class SelfPlay:
    def __init__(self, model: Network, args: TrainingArgs) -> None:
        self.model = model
        self.args = args

    def self_play(self, iteration: int) -> list[SelfPlayMemory]:
        self.model.eval()

        self_play_memory: list[SelfPlayMemory] = []
        self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]

        mcts = MCTS(
            self.model,
            MCTSArgs(
                num_searches_per_turn=self.args.mcts_num_searches_per_turn,
                dirichlet_epsilon=self.args.mcts_dirichlet_epsilon,
                dirichlet_alpha=self.args.mcts_dirichlet_alpha(iteration),
                c_param=self.args.mcts_c_param,
            ),
        )

        while len(self_play_games) > 0:
            for spg, action_probabilities in zip(self_play_games, mcts.search([spg.board for spg in self_play_games])):
                spg.memory.append(SelfPlayGameMemory(spg.board.copy(), action_probabilities))

                move = sample_move(action_probabilities, spg.num_played_moves, self.args.temperature)
                spg.board.make_move(move)

                if spg.board.is_game_over():
                    self_play_memory.extend(self._get_training_data(spg))

            self_play_games = [spg for spg in self_play_games if not spg.board.is_game_over()]

        return self_play_memory

    def _get_training_data(self, spg: SelfPlayGame) -> list[SelfPlayMemory]:
        self_play_memory: list[SelfPlayMemory] = []

        # 1 if current player won, -1 if current player lost, 0 if draw
        result = get_board_result_score(spg.board)
        assert result is not None, 'Game is not over'

        for mem in spg.memory[::-1]:  # reverse to flip the result for the other player
            encoded_board = CURRENT_GAME.get_canonical_board(mem.board)

            for board, probabilities in CURRENT_GAME.symmetric_variations(encoded_board, mem.action_probabilities):
                self_play_memory.append(
                    SelfPlayMemory(
                        torch.tensor(board, dtype=torch.int8, requires_grad=False),
                        torch.tensor(probabilities, dtype=torch.float32, requires_grad=False),
                        result,
                    )
                )
            result = -result

        return self_play_memory
