import numpy as np
import torch

from AIZeroConnect4Bot.src.util.log import log
from AIZeroConnect4Bot.src.eval.Bot import Bot
from AIZeroConnect4Bot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroConnect4Bot.src.Encoding import filter_policy_then_get_moves_and_probabilities, get_board_result_score
from AIZeroConnect4Bot.src.games.Game import Board
from AIZeroConnect4Bot.src.Network import Network, cached_network_inference
from AIZeroConnect4Bot.src.settings import CURRENT_GAME, CURRENT_GAME_MOVE, TORCH_DTYPE


class AlphaZeroBot(Bot):
    def __init__(self, network_model_file_path: str | None, max_time_to_think: float) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)
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

    def think(self, board: Board[CURRENT_GAME_MOVE]) -> CURRENT_GAME_MOVE:
        root = AlphaMCTSNode.root(board)

        for i in range(10000):
            self.iterate(root)
            if self.time_is_up:
                log(f'AlphaZeroBot has thought for {self.time_elapsed:.2f} seconds and {i+1} iterations')
                break

        best_child_index = np.argmax(root.children_number_of_visits)
        best_child = root.children[best_child_index]

        log('---------------------- Alpha Zero Best Move ----------------------')
        log(f'Best child has {best_child.number_of_visits:.4f} visits')
        log(f'Best child has {best_child.result_score:.4f} result_score')
        log(f'Best child has {best_child.policy:.4f} policy')
        log('------------------------------------------------------------------')

        return best_child.move_to_get_here

    def iterate(self, root: AlphaMCTSNode) -> None:
        current_node = root
        while not current_node.is_terminal_node and current_node.is_fully_expanded:
            current_node = current_node.best_child()

        if current_node.is_terminal_node:
            result = get_board_result_score(current_node.board)
            assert result is not None, 'Game is not over'
        else:
            moves_with_scores, result = self.evaluation(current_node.board)
            current_node.expand(moves_with_scores)

        current_node.back_propagate(result)

    @torch.no_grad()
    def evaluation(self, board: Board[CURRENT_GAME_MOVE]) -> tuple[list[tuple[CURRENT_GAME_MOVE, float]], float]:
        policy, value = cached_network_inference(
            self.model,
            torch.tensor(
                np.array([CURRENT_GAME.get_canonical_board(board)]),
                device=self.model.device,
                dtype=TORCH_DTYPE,
            ),
        )

        moves = filter_policy_then_get_moves_and_probabilities(policy[0], board)

        # print(
        #    f'Evaluated board:\n{CURRENT_GAME.get_canonical_board(board)}\nPolicy: {np.round(policy[0], 3)}\nValue: {value[0]}\nMoves: {moves}'
        # )
        return moves, value[0]
