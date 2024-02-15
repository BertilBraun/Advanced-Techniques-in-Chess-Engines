# This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

import torch
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from AIZeroChessBot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroChessBot.src.Network import Network
from AIZeroChessBot.src.SelfPlayGame import SelfPlayGame, SelfPlayGameMemory
from AIZeroChessBot.src.TrainingArgs import TrainingArgs
from AIZeroChessBot.src.util import lerp

from Framework import *
from AIZeroChessBot.src.MoveEncoding import (
    ACTION_SIZE,
    decode_move,
    encode_move,
    filter_policy_then_get_moves_and_probabilities,
)

from AIZeroChessBot.src.BoardEncoding import encode_boards, encode_board, board_result_to_score


@dataclass
class SelfPlayMemory:
    state: NDArray[np.float32]
    policy_targets: NDArray[np.float32]
    value_targets: float


class SelfPlay:
    def __init__(self, model: Network, args: TrainingArgs) -> None:
        self.model = model
        self.args = args

    def self_play(self) -> list[SelfPlayMemory]:
        return_memory: list[SelfPlayMemory] = []
        self_play_games = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]

        while len(self_play_games) > 0:
            self._expand_self_play_games(self_play_games)

            for i in range(len(self_play_games))[::-1]:
                spg = self_play_games[i]

                action_probabilities = self._get_action_probabilities(spg.root)
                spg.memory.append(SelfPlayGameMemory(spg.root.board, action_probabilities, spg.board.turn))

                move = self._sample_move(action_probabilities)

                spg.board = spg.board.copy(stack=False)
                spg.board.push(move)

                if spg.board.is_game_over():
                    value = -board_result_to_score(spg.board)

                    for mem in spg.memory:
                        hist_outcome = value if mem.turn == spg.board.turn else -value
                        return_memory.append(
                            SelfPlayMemory(encode_board(mem.board), mem.action_probabilities, hist_outcome)
                        )

                    del self_play_games[i]

        return return_memory

    @torch.no_grad()
    def _expand_self_play_games(self, self_play_games: list[SelfPlayGame]) -> None:
        policy = self._get_policy_with_noise(self_play_games)

        for spg, spg_policy in zip(self_play_games, policy):
            moves = filter_policy_then_get_moves_and_probabilities(spg_policy, spg.board)

            spg.root = AlphaMCTSNode(spg.board)
            spg.root.expand(moves)

        for _ in range(self.args.num_searches):
            for spg in self_play_games:
                spg.node = spg.get_best_child_or_back_propagate(self.args.c_param)

            expandable_self_play_games = [
                mappingIdx for mappingIdx, spg in enumerate(self_play_games) if spg.node is not None
            ]

            if len(expandable_self_play_games) > 0:
                # spg.node is not None because it is filtered in the expandable_self_play_games list
                boards = [self_play_games[mappingIdx].node.board for mappingIdx in expandable_self_play_games]  # type: ignore
                policy, value = self.model.inference(torch.tensor(encode_boards(boards), device=self.model.device))

            for i, mappingIdx in enumerate(expandable_self_play_games):
                # spg.node is not None because it is filtered in the expandable_self_play_games list
                node: AlphaMCTSNode = self_play_games[mappingIdx].node  # type: ignore

                moves = filter_policy_then_get_moves_and_probabilities(policy[i], node.board)

                node.expand(moves)
                node.back_propagate(value[i])

    def _get_policy_with_noise(self, self_play_games: list[SelfPlayGame]) -> NDArray[np.float32]:
        encoded_boards = encode_boards([spg.board for spg in self_play_games])
        policy, _ = self.model.inference(torch.tensor(encoded_boards, device=self.model.device))

        # Add dirichlet noise to the policy to encourage exploration
        dirichlet_noise = np.random.dirichlet([self.args.dirichlet_alpha] * ACTION_SIZE)
        policy = lerp(policy, dirichlet_noise, self.args.dirichlet_epsilon)
        return policy

    def _get_action_probabilities(self, root_node: AlphaMCTSNode) -> NDArray[np.float32]:
        action_probabilities = np.zeros(ACTION_SIZE, dtype=np.float32)

        for child in root_node.children:
            action_probabilities[encode_move(child.move_to_get_here)] = child.number_of_visits
        action_probabilities /= np.sum(action_probabilities)

        return action_probabilities

    def _sample_move(self, action_probabilities: NDArray[np.float32]) -> Move:
        temperature_action_probabilities = action_probabilities ** (1 / self.args.temperature)
        # Divide temperature_action_probabilities with its sum in case of an error
        action = np.random.choice(ACTION_SIZE, p=temperature_action_probabilities)

        move = decode_move(action)
        return move
