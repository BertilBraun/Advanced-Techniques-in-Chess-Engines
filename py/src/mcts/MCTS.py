import numpy as np

from src.util import lerp
from src.mcts.MCTSNode import MCTSNode
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CurrentBoard, CurrentGame
from src.cluster.InferenceClient import InferenceClient
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.util.log import log
from src.util.mcts_graph import draw_mcts_graph
from src.util.profiler import timeit


class MCTS:
    def __init__(self, client: InferenceClient, args: MCTSArgs) -> None:
        self.client = client
        self.args = args

    @timeit
    async def search(self, boards: list[CurrentBoard]) -> list[tuple[np.ndarray, float]]:
        policies = await self._get_policy_with_noise(boards)

        roots: list[MCTSNode] = []
        for board, spg_policy in zip(boards, policies):
            moves = filter_policy_then_get_moves_and_probabilities(spg_policy, board)

            root = MCTSNode.root(board)
            root.expand(moves)
            roots.append(root)

        for _ in range(self.args.num_searches_per_turn // self.args.num_parallel_searches):
            await self.parallel_iterate(roots)

        # for root in roots:
        #     log(repr(root))
        #     num_placed_stones = np.sum(root.board.board != 0)
        #     draw_mcts_graph(root, f'mcts_{num_placed_stones}.png')

        return [(self._get_action_probabilities(root), root.result_score) for root in roots]

    @timeit
    async def parallel_iterate(self, roots: list[MCTSNode]) -> None:
        await self.client.run_batch(
            [self.iterate(root) for root in roots for _ in range(self.args.num_parallel_searches)]
        )

    @timeit
    async def iterate(self, root: MCTSNode) -> None:
        if not (node := self._get_best_child_or_back_propagate(root, self.args.c_param)):
            return

        node.update_virtual_losses(1)

        policy, value = await self.client.inference(node.board)

        moves = filter_policy_then_get_moves_and_probabilities(policy, node.board)
        node.expand(moves)
        node.back_propagate(value)

        node.update_virtual_losses(-1)

    async def _get_policy_with_noise(self, boards: list[CurrentBoard]) -> np.ndarray:
        results = await self.client.run_batch([self.client.inference(board) for board in boards])

        policies = np.array([policy for policy, _ in results], dtype=np.float32)

        # Add dirichlet noise to the policy to encourage exploration
        dirichlet_noise = np.random.dirichlet(
            [self.args.dirichlet_alpha] * CurrentGame.action_size,
            size=len(boards),
        )
        policies = lerp(policies, dirichlet_noise, self.args.dirichlet_epsilon)
        return policies

    def _get_action_probabilities(self, root_node: MCTSNode) -> np.ndarray:
        action_probabilities = np.zeros(CurrentGame.action_size, dtype=np.float32)

        for child in root_node.children:
            action_probabilities[CurrentGame.encode_move(child.move_to_get_here)] = child.number_of_visits

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
