import torch
import numpy as np

from src.settings import CurrentBoard, CurrentGame, TORCH_DTYPE
from src.util import lerp
from src.Network import Network, cached_network_inference
from src.mcts.MCTSNode import MCTSNode
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.mcts.MCTSArgs import MCTSArgs


class MCTS:
    def __init__(self, model: Network, args: MCTSArgs) -> None:
        self.model = model
        self.args = args

    @torch.no_grad()
    def search(self, boards: list[CurrentBoard]) -> list[np.ndarray]:
        policy = self._get_policy_with_noise(boards)

        nodes: list[MCTSNode] = []
        for board, spg_policy in zip(boards, policy):
            moves = filter_policy_then_get_moves_and_probabilities(spg_policy, board)

            root = MCTSNode.root(board)
            root.expand(moves)
            nodes.append(root)

        for _ in range(self.args.num_searches_per_turn):
            expandable_nodes: list[MCTSNode] = []
            for root in nodes:
                node = self._get_best_child_or_back_propagate(root, self.args.c_param)
                if node is not None:
                    expandable_nodes.append(node)

            if len(expandable_nodes) == 0:
                continue

            encoded_boards = [CurrentGame.get_canonical_board(node.board) for node in expandable_nodes]
            policy, value = cached_network_inference(
                self.model,
                torch.tensor(
                    np.array(encoded_boards),
                    device=self.model.device,
                    dtype=TORCH_DTYPE,
                ),
            )

            for i, node in enumerate(expandable_nodes):
                moves = filter_policy_then_get_moves_and_probabilities(policy[i], node.board)

                node.expand(moves)
                node.back_propagate(value[i])

        return [self._get_action_probabilities(root) for root in nodes]

    def _get_policy_with_noise(self, boards: list[CurrentBoard]) -> np.ndarray:
        encoded_boards = [CurrentGame.get_canonical_board(board) for board in boards]
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
            [self.args.dirichlet_alpha] * CurrentGame.action_size,
            size=len(boards),
        )
        policy = lerp(policy, dirichlet_noise, self.args.dirichlet_epsilon)
        return policy

    def _get_action_probabilities(self, root_node: MCTSNode) -> np.ndarray:
        action_probabilities = np.zeros(CurrentGame.action_size, dtype=np.float32)

        for child in root_node.children:
            action_probabilities[child.move_to_get_here] = child.number_of_visits
        action_probabilities /= np.sum(action_probabilities)

        return action_probabilities

    def _get_best_child_or_back_propagate(self, root: MCTSNode, c_param: float) -> MCTSNode | None:
        node = root

        while node.is_fully_expanded:
            node = node.best_child(c_param)

        if node.is_terminal_node:
            result = get_board_result_score(node.board)
            assert result is not None
            node.back_propagate(-result)
            return None

        return node
