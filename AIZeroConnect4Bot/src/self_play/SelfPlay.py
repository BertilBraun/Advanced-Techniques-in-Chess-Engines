# This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

from dataclasses import dataclass
import numpy as np
import torch

from AIZeroConnect4Bot.src.settings import CURRENT_GAME, CURRENT_GAME_MOVE, TORCH_DTYPE
from AIZeroConnect4Bot.src.util import lerp
from AIZeroConnect4Bot.src.Network import Network, cached_network_inference
from AIZeroConnect4Bot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroConnect4Bot.src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from AIZeroConnect4Bot.src.train.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.self_play.SelfPlayGame import SelfPlayGame, SelfPlayGameMemory


@dataclass
class SelfPlayMemory:
    state: torch.Tensor
    policy_targets: torch.Tensor
    value_target: float


class SelfPlay:
    def __init__(self, model: Network, args: TrainingArgs) -> None:
        self.model = model
        self.args = args

    def self_play(self) -> list[SelfPlayMemory]:
        self_play_memory: list[SelfPlayMemory] = []
        self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]

        self.model.eval()

        while len(self_play_games) > 0:
            self._expand_self_play_games(self_play_games)

            for spg in self_play_games:
                action_probabilities = self._get_action_probabilities(spg.root)
                spg.memory.append(SelfPlayGameMemory(spg.root.board, action_probabilities))

                move = self._sample_move(action_probabilities, spg.root)

                spg.board = spg.root.board.copy()
                spg.board.make_move(move)

                if spg.board.is_game_over():
                    self_play_memory.extend(self._get_training_data(spg))

            self_play_games = [spg for spg in self_play_games if not spg.board.is_game_over()]

        return self_play_memory

    def _get_training_data(self, spg: SelfPlayGame) -> list[SelfPlayMemory]:
        self_play_memory = []

        # 1 if current player won, -1 if current player lost, 0 if draw
        result = get_board_result_score(spg.board)
        assert result is not None, 'Game is not over'

        for mem in spg.memory[::-1]:  # reverse to flip the result for the other player
            encoded_board = CURRENT_GAME.get_canonical_board(mem.board)

            # print(
            #     f'encoded_board: {encoded_board}\n'
            #     f'mem.policy_targets: {np.round(mem.action_probabilities, 3)}\n'
            #     f'result: {result}\n'
            # )

            for board, probabilities in CURRENT_GAME.symmetric_variations(encoded_board, mem.action_probabilities):
                self_play_memory.append(
                    SelfPlayMemory(
                        torch.tensor(board.copy(), dtype=torch.int8, requires_grad=False),
                        torch.tensor(probabilities.copy(), dtype=torch.float32, requires_grad=False),
                        result,
                    )
                )
            result = -result

        return self_play_memory

    @torch.no_grad()
    def _expand_self_play_games(self, self_play_games: list[SelfPlayGame]) -> None:
        policy = self._get_policy_with_noise(self_play_games)

        for spg, spg_policy in zip(self_play_games, policy):
            moves = filter_policy_then_get_moves_and_probabilities(spg_policy, spg.board)

            spg.root = AlphaMCTSNode.root(spg.board)
            spg.root.expand(moves)
            spg.node = spg.root

        for _ in range(self.args.num_iterations_per_turn):
            for spg in self_play_games:
                spg.node = spg.get_best_child_or_back_propagate(self.args.c_param)

            expandable_nodes: list[tuple[SelfPlayGame, AlphaMCTSNode]] = [
                (spg, spg.node) for spg in self_play_games if spg.node is not None
            ]

            if len(expandable_nodes) == 0:
                continue

            boards = [CURRENT_GAME.get_canonical_board(node.board) for _, node in expandable_nodes]
            policy, value = cached_network_inference(
                self.model,
                torch.tensor(
                    np.array(boards),
                    device=self.model.device,
                    dtype=TORCH_DTYPE,
                ),
            )

            for i, (spg, node) in enumerate(expandable_nodes):
                moves = filter_policy_then_get_moves_and_probabilities(policy[i], node.board)

                node.expand(moves)
                if node.board.current_player == spg.root.board.current_player:
                    node.back_propagate(value[i])
                else:
                    node.back_propagate(-value[i])

    def _get_policy_with_noise(self, self_play_games: list[SelfPlayGame]) -> np.ndarray:
        encoded_boards = [CURRENT_GAME.get_canonical_board(spg.board) for spg in self_play_games]
        policy, _ = cached_network_inference(
            self.model,
            torch.tensor(
                np.array(encoded_boards),
                device=self.model.device,
                dtype=TORCH_DTYPE,
            ),
        )

        # Add dirichlet noise to the policy to encourage exploration
        dirichlet_noise = np.random.dirichlet(
            [self.args.dirichlet_alpha] * CURRENT_GAME.action_size,
            size=len(self_play_games),
        )
        policy = lerp(policy, dirichlet_noise, self.args.dirichlet_epsilon)
        return policy

    def _get_action_probabilities(self, root_node: AlphaMCTSNode) -> np.ndarray:
        action_probabilities = np.zeros(CURRENT_GAME.action_size, dtype=np.float32)

        for child in root_node.children:
            action_probabilities[child.move_to_get_here] = child.number_of_visits
        action_probabilities /= np.sum(action_probabilities)

        return action_probabilities

    def _sample_move(self, action_probabilities: np.ndarray, root_node: AlphaMCTSNode) -> CURRENT_GAME_MOVE:
        # only use temperature for the first 30 moves, then simply use the action probabilities as they are
        if root_node.num_played_moves < 30:
            temperature_action_probabilities = action_probabilities ** (1 / self.args.temperature)
        else:
            temperature_action_probabilities = action_probabilities

        action = np.random.choice(CURRENT_GAME.action_size, p=temperature_action_probabilities)

        return action
