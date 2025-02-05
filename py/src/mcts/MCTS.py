import numpy as np
from dataclasses import dataclass

from src.util import lerp
from src.mcts.MCTSNode import MCTSNode
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CurrentBoard, CurrentGame
from src.cluster.InferenceClient import InferenceClient
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.util.tensorboard import log_scalar
from src.util.timing import timeit


@dataclass
class MCTSResult:
    result_score: float
    visit_counts: list[tuple[int, int]]
    children: list[MCTSNode]


def action_probabilities(visit_counts: list[tuple[int, int]]) -> np.ndarray:
    action_probabilities = np.zeros(CurrentGame.action_size, dtype=np.float32)
    for move, visit_count in visit_counts:
        action_probabilities[move] = visit_count
    action_probabilities /= np.sum(action_probabilities)
    return action_probabilities


class MCTS:
    def __init__(self, client: InferenceClient, args: MCTSArgs) -> None:
        self.client = client
        self.args = args

    @timeit
    def search(self, inputs: list[tuple[CurrentBoard, MCTSNode | None]]) -> list[MCTSResult]:
        if not inputs:
            return []

        # TODO Already expanded nodes currently dont properly work. Issues:
        #  - The addition of noise to the policies does not seem to retroactively affect the exploration of other nodes enough
        #  - The node will get visited an additional X times, but should be visited a total of X times, i.e. X - node.number_of_visits times

        policies = self._get_policy_with_noise([board for board, node in inputs if node is None])

        roots: list[MCTSNode] = []
        policies_index = 0

        for board, node in inputs:
            if node is not None:
                assert node.parent is None, 'Node must be a root node (be careful with memory leaks)'
                # Add noise to the children policies, then reuse the node
                node.children_policies = self._add_noise(node.children_policies)
                roots.append(node)
                continue

            moves = filter_policy_then_get_moves_and_probabilities(policies[policies_index], board)
            policies_index += 1

            root = MCTSNode.root(board)
            root.expand(moves)
            roots.append(root)

        for _ in range(self.args.num_searches_per_turn // self.args.num_parallel_searches):
            self.parallel_iterate(roots)

        # for root in roots:
        #     print(repr(root))
        #     num_placed_stones = np.sum(root.board.board != 0)
        #     from src.util.mcts_graph import draw_mcts_graph
        #
        #     draw_mcts_graph(root, f'mcts_{num_placed_stones}.png')

        def dfs(node: MCTSNode) -> int:
            # depth dfs
            if not node or not node.is_fully_expanded:
                return 0
            return 1 + max(dfs(child) for child in node.children)

        depths = [dfs(root) for root in roots]
        average_depth = sum(depths) / len(depths)

        log_scalar('dataset/average_search_depth', average_depth)

        def entropy(node: MCTSNode) -> float:
            # calculate the entropy of the nodes visit counts
            node_number_of_visits = node.number_of_visits
            return -sum(
                visit_count / node_number_of_visits * np.log2(visit_count / node_number_of_visits)
                for visit_count in node.children_number_of_visits
                if visit_count > 0
            )

        entropies = [entropy(root) for root in roots]
        average_entropy = sum(entropies) / len(entropies)

        log_scalar('dataset/average_search_entropy', average_entropy)

        # print(repr(roots[0]))

        return [
            MCTSResult(
                root.result_score,
                [(child.encoded_move_to_get_here, child.number_of_visits) for child in root.children],
                root.children,
            )
            for root in roots
        ]

    @timeit
    def parallel_iterate(self, roots: list[MCTSNode]) -> None:
        nodes = self._get_nodes_to_expand(roots)

        if not nodes:
            return

        results = self.client.inference_batch([node.board for node in nodes])

        self._apply_results(nodes, results)

    @timeit
    def _get_nodes_to_expand(self, roots: list[MCTSNode]) -> list[MCTSNode]:
        nodes: list[MCTSNode] = []

        for root in roots:
            node = self._get_best_child_or_back_propagate(root, self.args.c_param, self.args.min_visit_count)
            if node is not None:
                node.update_virtual_losses(1)
                nodes.append(node)

        return nodes

    @timeit
    def _apply_results(self, nodes: list[MCTSNode], results: list[tuple[np.ndarray, float]]) -> None:
        for node, (policy, value) in zip(nodes, results):
            moves = filter_policy_then_get_moves_and_probabilities(policy, node.board)
            node.expand(moves)
            node.back_propagate(value)

            node.update_virtual_losses(-1)

    def _get_policy_with_noise(self, boards: list[CurrentBoard]) -> list[np.ndarray]:
        results = self.client.inference_batch(boards)

        return [self._add_noise(policy) for policy, _ in results]

    def _add_noise(self, policy: np.ndarray) -> np.ndarray:
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(policy))
        return lerp(policy, noise, self.args.dirichlet_epsilon)

    def _get_best_child_or_back_propagate(
        self, root: MCTSNode, c_param: float, min_visit_count: int
    ) -> MCTSNode | None:
        node = root

        while node.is_fully_expanded:
            node = node.best_child(c_param, min_visit_count)
            min_visit_count = 0  # Only the root children have a min_visit_count

        if node.is_terminal_node:
            result = get_board_result_score(node.board)
            assert result is not None
            node.back_propagate(-result)
            return None

        return node
