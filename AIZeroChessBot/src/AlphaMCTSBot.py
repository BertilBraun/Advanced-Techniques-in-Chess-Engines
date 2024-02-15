import torch

from AIZeroChessBot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroChessBot.src.BoardEncoding import encode_boards
from AIZeroChessBot.src.MoveEncoding import filter_policy_then_get_moves_and_probabilities
from AIZeroChessBot.src.Network import Network

from Framework import *


# TODO exploration vs exploitation - how does that work here? How do we balance the two?
# TODO how to do the self-play part of the training?
# TODO why is there a State and a Node class? What is the difference between the two?
# TODO base the architecture on the AlphaZero paper but with less layers

"""
https://en.wikipedia.org/wiki/AlphaZero

AlphaZero only searches ~80.000 positions per second, while Stockfish searches ~70.000.000 positions per second.

Training
AlphaZero was trained solely via self-play, using 5,000 first-generation TPUs to generate the games and 64 second-generation TPUs to train the neural networks. In parallel, the in-training AlphaZero was periodically matched against its benchmark (Stockfish, Elmo, or AlphaGo Zero) in brief one-second-per-move games to determine how well the training was progressing. DeepMind judged that AlphaZero's performance exceeded the benchmark after around four hours of training for Stockfish, two hours for Elmo, and eight hours for AlphaGo Zero.
"""


class AlphaMCTSBot(ChessBot):
    def __init__(self, network_model_file: str) -> None:
        super().__init__('Alpha MCTS Bot')
        self.model = Network()
        self.model.load_state_dict(torch.load(network_model_file))

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
