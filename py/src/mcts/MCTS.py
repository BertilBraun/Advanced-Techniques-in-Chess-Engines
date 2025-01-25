import numpy as np
from dataclasses import dataclass

from src.util import lerp
from src.mcts.MCTSNode import MCTSNode
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CurrentBoard, CurrentGame
from src.cluster.InferenceClient import InferenceClient
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
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
    async def search(self, inputs: list[tuple[CurrentBoard, MCTSNode | None]]) -> list[MCTSResult]:
        policies = await self._get_policy_with_noise([board for board, node in inputs if node is None])

        roots: list[MCTSNode] = []
        policies_index = 0

        for board, node in inputs:
            if node is not None:
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
            await self.parallel_iterate(roots)

        # for root in roots:
        #     log(repr(root))
        #     num_placed_stones = np.sum(root.board.board != 0)
        #     draw_mcts_graph(root, f'mcts_{num_placed_stones}.png')

        return [
            MCTSResult(
                root.result_score,
                [(CurrentGame.encode_move(child.move_to_get_here), child.number_of_visits) for child in root.children],
                root.children,
            )
            for root in roots
        ]

    @timeit
    async def parallel_iterate(self, roots: list[MCTSNode]) -> None:
        await self.client.run_batch(
            [self.iterate(root) for root in roots for _ in range(self.args.num_parallel_searches)]
        )

    @timeit
    async def iterate(self, root: MCTSNode) -> None:
        if not (node := self._get_best_child_or_back_propagate(root, self.args.c_param, self.args.min_visit_count)):
            return

        node.update_virtual_losses(1)

        policy, value = await self.client.inference(node.board)

        moves = filter_policy_then_get_moves_and_probabilities(policy, node.board)
        node.expand(moves)
        node.back_propagate(value)

        node.update_virtual_losses(-1)

    async def _get_policy_with_noise(self, boards: list[CurrentBoard]) -> list[np.ndarray]:
        results = await self.client.run_batch([self.client.inference(board) for board in boards])

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
