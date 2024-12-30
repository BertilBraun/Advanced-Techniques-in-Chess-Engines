# This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

from dataclasses import dataclass
import numpy as np
import torch

from src.settings import CURRENT_GAME, CURRENT_GAME_MOVE, TORCH_DTYPE
from src.util import lerp
from src.Network import Network, cached_network_inference
from src.AlphaMCTSNode import AlphaMCTSNode
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.train.TrainingArgs import TrainingArgs
from src.self_play.SelfPlayGame import SelfPlayGame, SelfPlayGameMemory


@dataclass
class SelfPlayMemory:
    state: torch.Tensor
    policy_targets: torch.Tensor
    value_target: float


class SelfPlay:
    def __init__(self, model: Network, args: TrainingArgs) -> None:
        self.model = model
        self.args = args

    def self_play(self, iteration: int) -> list[SelfPlayMemory]:
        self_play_memory: list[SelfPlayMemory] = []
        self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]

        self.model.eval()

        while len(self_play_games) > 0:
            self._expand_self_play_games(self_play_games, iteration)
            iteration += 1  # TODO remove this

            for spg in self_play_games:
                action_probabilities = self._get_action_probabilities(spg.root)
                spg.memory.append(SelfPlayGameMemory(spg.root.board, action_probabilities))

                move = self._sample_move(action_probabilities, spg.root)

                spg.board = spg.root.board.copy()
                spg.board.make_move(move)

                if spg.board.is_game_over():
                    self_play_memory.extend(self._get_training_data(spg))
                    exit()  # TODO remove this

            self_play_games = [spg for spg in self_play_games if not spg.board.is_game_over()]

        return self_play_memory

    def _get_training_data(self, spg: SelfPlayGame) -> list[SelfPlayMemory]:
        self_play_memory = []

        # 1 if current player won, -1 if current player lost, 0 if draw
        result = get_board_result_score(spg.board)
        assert result is not None, 'Game is not over'

        self.analyze(spg)

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

    def analyze(self, spg):
        database = {}
        with open('tictactoe_database.txt', 'r') as f:
            for line in f:
                # (0, 0, 0, 0, 0, 0, 0, 0, 0);(0, 1, 2, 3, 4, 5, 6, 7, 8);0
                board, valid_moves, result = line.split(';')
                board = eval(board)
                valid_moves = eval(valid_moves)
                database[board] = (valid_moves, int(result))

        for i, mem in enumerate(spg.memory[::-1]):  # reverse to flip the result for the other player
            if i >= len(spg.memory) - 1:
                continue
            encoded_board = CURRENT_GAME.get_canonical_board(mem.board)
            board = tuple((encoded_board[0] - encoded_board[1]).reshape(-1).tolist())
            valid_moves, db_result = database[board]

            moves = filter_policy_then_get_moves_and_probabilities(mem.action_probabilities, mem.board)

            if max(moves, key=lambda x: x[1])[0] not in valid_moves:
                if (encoded_board[0] + encoded_board[1]).sum() >= 3:
                    print('Moves:', list(sorted(moves, key=lambda x: x[1], reverse=True)))
                    print('Valid moves from db:', valid_moves)
                    print('Result:', str(result).strip(), 'DB Result:', db_result)
                    print('Curr Board:\n', np.array(board).reshape(3, 3))
                    # input()
            else:
                print('Move is valid and in DB')

    def dump_search_tree_to_string(self, node: AlphaMCTSNode) -> str:
        res = ''
        res += node.graph_id(self.args) + '\n'
        for child in node.children:
            child.init()
        for child in sorted(node.children, key=lambda x: x.move_to_get_here, reverse=True):
            res += self.dump_search_tree_to_string(child)
        return res

    @torch.no_grad()
    def _expand_self_play_games(self, self_play_games: list[SelfPlayGame], iteration: int) -> None:
        policy = self._get_policy_with_noise(self_play_games, iteration)

        for spg, spg_policy in zip(self_play_games, policy):
            moves = filter_policy_then_get_moves_and_probabilities(spg_policy, spg.board)

            spg.root = AlphaMCTSNode.root(spg.board)
            spg.root.expand(moves)

        f = open(f'search_tree_{iteration}.txt', 'w')
        for _ in range(self.args.num_searches_per_turn):
            f.write('------------------------------------\n')
            f.write(self.dump_search_tree_to_string(self_play_games[0].root))
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
                spg.back_propagate(value[i], node)
        spg.root.show_graph(self.args, iteration)  # TODO remove this

    def _get_policy_with_noise(self, self_play_games: list[SelfPlayGame], iteration: int) -> np.ndarray:
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
        np.random.seed(42)  # TODO remove this
        dirichlet_noise = np.random.dirichlet(
            [self.args.dirichlet_alpha(iteration)] * CURRENT_GAME.action_size,
            # TODO readd this size=len(self_play_games),
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
            temperature_action_probabilities /= np.sum(temperature_action_probabilities)
        else:
            assert False  # TODO remove
            temperature_action_probabilities = action_probabilities

        np.random.seed(42)  # TODO remove this
        action = np.random.choice(CURRENT_GAME.action_size, p=temperature_action_probabilities)

        return CURRENT_GAME.decode_move(action)
