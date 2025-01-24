import numpy as np

from src.Encoding import filter_policy_then_get_moves_and_probabilities
from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.util.log import log
from src.eval.Bot import Bot
from src.mcts.MCTSNode import MCTSNode
from src.settings import TRAINING_ARGS, CurrentBoard, CurrentGame, CurrentGameMove, PLAY_C_PARAM


class AlphaZeroBot(Bot):
    def __init__(self, iteration: int, max_time_to_think: float = 1.0, network_eval_only: bool = False) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)

        self.inference_client = InferenceClient(0, TRAINING_ARGS)
        self.inference_client.update_iteration(iteration)

        self.mcts_args = MCTSArgs(
            # ensure, that the max number of visits of a node does not exceed the capacity of an uint16
            num_searches_per_turn=2**16 - 1,
            num_parallel_searches=TRAINING_ARGS.self_play.mcts.num_parallel_searches,
            dirichlet_alpha=0.5,  # irrelevant
            dirichlet_epsilon=0.0,
            c_param=PLAY_C_PARAM,
            min_visit_count=0,
        )
        self.mcts = MCTS(self.inference_client, self.mcts_args)

        self.network_eval_only = network_eval_only

        # run some inferences to compile and warm up the model
        board = CurrentBoard()
        for _ in range(10):
            self.inference_client._model_inference([CurrentGame.get_canonical_board(board)])
            board.make_move(board.get_valid_moves()[0])

    async def think(self, board: CurrentBoard) -> CurrentGameMove:
        if self.network_eval_only:
            results = await self.inference_client.run_batch([self.inference_client.inference(board)])
            policy, value = results[0]
            moves = filter_policy_then_get_moves_and_probabilities(policy, board)
            move = max(moves, key=lambda move: move[1])

            log('---------------------- Alpha Zero Best Move ----------------------')
            log('Best move:', move[0])
            log('Best move probability:', move[1])
            log('Move probabilities:', moves)
            log('Result value:', value)
            log('------------------------------------------------------------------')

            return move[0]

        root = MCTSNode.root(board)

        for _ in range(self.mcts_args.num_searches_per_turn // self.mcts_args.num_parallel_searches):
            await self.mcts.parallel_iterate([root])
            if self.time_is_up:
                break

        best_move_index = np.argmax(root.children_number_of_visits)
        best_child = root.children[best_move_index]

        def max_depth(node: MCTSNode) -> int:
            if not node.children:
                return 0
            return 1 + max(max_depth(child) for child in node.children)

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Total number of searches:', root.number_of_visits)
        log('Max depth:', max_depth(root))
        log('Best child index:', best_move_index)
        log(f'Best child has {best_child.number_of_visits} visits')
        log(f'Best child has {best_child.result_score:.4f} result_score')
        log('Child moves:', [child.move_to_get_here for child in root.children])
        log('Child visits:', root.children_number_of_visits)
        log('Child result_scores:', np.round(root.children_result_scores, 2))
        log('Child priors:', np.round(root.children_policies, 2))
        log('------------------------------------------------------------------')

        return best_child.move_to_get_here
