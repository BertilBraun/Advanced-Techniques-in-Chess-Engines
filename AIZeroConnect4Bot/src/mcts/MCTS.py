import asyncio
import numpy as np

from src.cluster.InferenceClient import InferenceClient
from src.settings import CurrentBoard, CurrentGame
from src.util import lerp
from src.mcts.MCTSNode import MCTSNode
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.mcts.MCTSArgs import MCTSArgs
from src.util.log import log


class MCTS:
    def __init__(self, client: InferenceClient, args: MCTSArgs) -> None:
        self.client = client
        self.args = args

    async def iterate(self, root: MCTSNode) -> None:
        if not (node := self._get_best_child_or_back_propagate(root, self.args.c_param)):
            return

        node.update_virtual_losses(1)

        policy, value = await self.client.inference(node.board)

        moves = filter_policy_then_get_moves_and_probabilities(policy, node.board)
        node.expand(moves)
        node.back_propagate(value)

        node.update_virtual_losses(-1)

    async def search(self, boards: list[CurrentBoard]) -> list[np.ndarray]:
        policies = await self._get_policy_with_noise(boards)
        log('Got policies')

        nodes: list[MCTSNode] = []
        for board, spg_policy in zip(boards, policies):
            moves = filter_policy_then_get_moves_and_probabilities(spg_policy, board)

            root = MCTSNode.root(board)
            root.expand(moves)
            nodes.append(root)

        for _ in range(self.args.num_searches_per_turn // self.args.num_parallel_searches):
            log('Next search iteration')
            await asyncio.gather(
                *[self.iterate(root) for _ in range(self.args.num_parallel_searches) for root in nodes]
            )

        log('Search done, getting action probabilities')
        return [self._get_action_probabilities(root) for root in nodes]

    async def _get_policy_with_noise(self, boards: list[CurrentBoard]) -> np.ndarray:
        results = await asyncio.gather(*[self.client.inference(board) for board in boards])
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
