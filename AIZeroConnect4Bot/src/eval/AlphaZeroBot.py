import numpy as np
import torch

from src.util.log import log
from src.eval.Bot import Bot
from src.AlphaMCTSNode import AlphaMCTSNode
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.Network import Network, cached_network_inference
from src.settings import CURRENT_BOARD, CURRENT_GAME, CURRENT_GAME_MOVE, PLAY_C_PARAM, TORCH_DTYPE


class AlphaZeroBot(Bot):
    def __init__(self, network_model_file_path: str | Network | None, max_time_to_think: float) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)
        if isinstance(network_model_file_path, Network):
            self.model = network_model_file_path
        else:
            self.model = Network()
            if network_model_file_path is not None:
                self.model.load_state_dict(
                    torch.load(
                        network_model_file_path,
                        map_location=self.model.device,
                        weights_only=True,
                    )
                )
        self.model.eval()

    def think(self, board: CURRENT_BOARD) -> CURRENT_GAME_MOVE:
        # board.board = np.array([1, 0, 1, 0, -1, 0, 0, 0, -1])
        root = AlphaMCTSNode.root(board)

        for _ in range(1_000_000):
            self.iterate(root)
            if self.time_is_up:
                break

        # root.show_graph()

        best_child_index = np.argmax(root.children_number_of_visits)
        best_child = root.children[best_child_index]

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Best child index:', best_child_index)
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

    def iterate(self, root: AlphaMCTSNode) -> None:
        current_node = root
        while not current_node.is_terminal_node and current_node.is_fully_expanded:
            current_node = current_node.best_child(PLAY_C_PARAM)

        # TODO if not current_node.is_terminal_node:
        # TODO     print('Currently at node:')
        # TODO     current_node.board.display()

        if current_node.is_terminal_node:
            result = get_board_result_score(current_node.board)
            assert result is not None, 'Game is not over'
            result = -result
        else:
            moves_with_scores, result = self.evaluation(current_node.board)
            current_node.expand(moves_with_scores)

        # TODO if not current_node.is_terminal_node:
        # TODO     print('Result:', result)
        # TODO     print('Child visits:', [child.number_of_visits for child in current_node.children])

        current_node.back_propagate(result)

    @torch.no_grad()
    def evaluation(self, board: CURRENT_BOARD) -> tuple[list[tuple[CURRENT_GAME_MOVE, float]], float]:
        policy, value = cached_network_inference(
            self.model,
            torch.tensor(
                np.array([CURRENT_GAME.get_canonical_board(board)]),
                device=self.model.device,
                dtype=TORCH_DTYPE,
            ),
        )
        # TODO remove this
        # Test, wheather the search is working
        # policy = np.full(policy.shape, 1 / policy.shape[1])
        # value = np.full(value.shape, 0.0)

        moves = filter_policy_then_get_moves_and_probabilities(policy[0], board)

        return moves, value[0]
