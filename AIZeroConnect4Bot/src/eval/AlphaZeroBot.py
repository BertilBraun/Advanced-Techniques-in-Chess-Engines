import numpy as np
import torch

from src.util.compile import try_compile
from src.util.log import log
from src.eval.Bot import Bot
from src.mcts.MCTSNode import MCTSNode
from src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from src.Network import Network, cached_network_inference
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove, PLAY_C_PARAM, TORCH_DTYPE, TRAINING_ARGS


class AlphaZeroBot(Bot):
    def __init__(self, network_model_file_path: str | Network | None, max_time_to_think: float) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)
        if isinstance(network_model_file_path, Network):
            self.model = network_model_file_path
        else:
            self.model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size, device=None)
            self.model = try_compile(self.model)
            if network_model_file_path is not None:
                self.model.load_state_dict(
                    torch.load(
                        network_model_file_path,
                        map_location=self.model.device,
                        weights_only=False,
                    )
                )
        self.model.eval()

    def think(self, board: CurrentBoard) -> CurrentGameMove:
        root = MCTSNode.root(board)

        for _ in range(2**16 - 1):
            # ensure, that the max number of visits of a node does not exceed the capacity of an uint16
            self.iterate(root)
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

    def iterate(self, root: MCTSNode) -> None:
        current_node = root
        while not current_node.is_terminal_node and current_node.is_fully_expanded:
            current_node = current_node.best_child(PLAY_C_PARAM)

        if current_node.is_terminal_node:
            result = get_board_result_score(current_node.board)
            assert result is not None, 'Game is not over'
            result = -result
        else:
            moves_with_scores, result = self.evaluation(current_node.board)
            current_node.expand(moves_with_scores)

        current_node.back_propagate(result)

    @torch.no_grad()
    def evaluation(self, board: CurrentBoard) -> tuple[list[tuple[CurrentGameMove, float]], float]:
        policy, value = cached_network_inference(
            self.model,
            torch.tensor(
                np.array([CurrentGame.get_canonical_board(board)]),
                device=self.model.device,
                dtype=TORCH_DTYPE,
            ),
        )

        moves = filter_policy_then_get_moves_and_probabilities(policy[0], board)

        return moves, value[0]
