import numpy as np

from src.cluster.InferenceServerProcess import start_inference_server
from src.util.log import log
from src.eval.Bot import Bot
from src.mcts.MCTSNode import MCTSNode
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.settings import CurrentBoard, CurrentGameMove, PLAY_C_PARAM


class AlphaZeroBot(Bot):
    def __init__(self, iteration: int, max_time_to_think: float) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)

        inference_client, stop_inference_server = start_inference_server(iteration)
        self.inference_client = inference_client
        self.stop_inference_server = stop_inference_server

    async def think(self, board: CurrentBoard) -> CurrentGameMove:
        root = MCTSNode.root(board)

        for _ in range(2**16 - 1):
            # ensure, that the max number of visits of a node does not exceed the capacity of an uint16
            await self.iterate(root)  # TODO could these just run in parallel? Virtual loss probably
            if self.time_is_up:
                break

        best_move_index = np.argmax(root.children_number_of_visits)
        best_child = root.children[best_move_index]

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Best child index:', best_move_index)
        log('Child number of visits:', root.children_number_of_visits)
        log(f'Best child has {best_child.number_of_visits} visits')
        log(f'Best child has {best_child.result_score:.4f} result_score')
        log(f'Best child has {best_child.policy:.4f} policy')
        log('Child moves:', [child.move_to_get_here for child in root.children])
        log('Child visits:', [child.number_of_visits for child in root.children])
        log('Child result_scores:', [round(child.result_score, 2) for child in root.children])
        log('Child policies:', [round(child.policy, 2) for child in root.children])
        log('------------------------------------------------------------------')

        return best_child.move_to_get_here

    async def iterate(self, root: MCTSNode) -> None:
        current_node = root
        while not current_node.is_terminal_node and current_node.is_fully_expanded:
            current_node = current_node.best_child(PLAY_C_PARAM)

        if current_node.is_terminal_node:
            result = get_board_result_score(current_node.board)
            assert result is not None, 'Game is not over'
            result = -result
        else:
            moves_with_scores, result = await self.evaluation(current_node.board)
            current_node.expand(moves_with_scores)

        current_node.back_propagate(result)

    async def evaluation(self, board: CurrentBoard) -> tuple[list[tuple[CurrentGameMove, float]], float]:
        policy, value = await self.inference_client.inference([board])

        moves = filter_policy_then_get_moves_and_probabilities(policy[0], board)

        return moves, value[0]
