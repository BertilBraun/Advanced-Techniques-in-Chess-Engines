import torch

from AIZeroChessBot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroChessBot.src.BoardEncoding import encode_boards
from AIZeroChessBot.src.MoveEncoding import filter_policy_then_get_moves_and_probabilities
from AIZeroChessBot.src.Network import Network

from Framework import *


class AlphaZeroBot(ChessBot):
    def __init__(self, network_model_file_path) -> None:
        super().__init__('Alpha MCTS Bot')
        self.model = Network()
        self.model.load_state_dict(torch.load(network_model_file_path))

    def think(self, board: Board) -> Move:
        root = AlphaMCTSNode(board, 0.0, Move.null(), None)

        while not self.time_is_up:
            self.iterate(root)

        return root.best_child(c_param=0.0).move_to_get_here

    def iterate(self, root: AlphaMCTSNode) -> None:
        current_node = root

        while not current_node.is_terminal_node:
            if current_node.is_fully_expanded:
                current_node = current_node.best_child()
            else:
                moves_with_scores, result = self.evaluation(current_node.board)
                current_node.expand(moves_with_scores)
                current_node.back_propagate(result)
                return

    @torch.no_grad()
    def evaluation(self, board: Board) -> tuple[list[tuple[Move, float]], float]:
        policy, value = self.model.inference(torch.tensor(encode_boards([board]), device=self.model.device))

        moves = filter_policy_then_get_moves_and_probabilities(policy[0], board)

        return moves, value[0]
