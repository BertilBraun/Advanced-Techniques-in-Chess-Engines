import numpy as np

from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.util.log import log
from src.eval.Bot import Bot
from src.mcts.MCTSNode import MCTSNode
from src.settings import TRAINING_ARGS, CurrentBoard, CurrentGame, CurrentGameMove, PLAY_C_PARAM


class AlphaZeroBot(Bot):
    def __init__(self, iteration: int, max_time_to_think: float, network_eval_only: bool) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)

        BATCH_SIZE = 16

        self.inference_client = InferenceClient(0, TRAINING_ARGS)
        self.inference_client.update_iteration(iteration)

        self.mcts_args = MCTSArgs(
            # ensure, that the max number of visits of a node does not exceed the capacity of an uint16
            num_searches_per_turn=2**16 - 1,
            num_parallel_searches=BATCH_SIZE,
            dirichlet_alpha=0.5,  # irrelevant
            dirichlet_epsilon=0.0,
            c_param=PLAY_C_PARAM,
        )
        self.mcts = MCTS(self.inference_client, self.mcts_args)

        self.network_eval_only = network_eval_only

    async def think(self, board: CurrentBoard) -> CurrentGameMove:
        if self.network_eval_only:
            results = await self.inference_client.run_batch([self.inference_client.inference(board)])
            policy, _ = results[0]
            return CurrentGame.decode_move(np.argmax(policy).item())

        root = MCTSNode.root(board)

        for _ in range(self.mcts_args.num_searches_per_turn // self.mcts_args.num_parallel_searches):
            await self.mcts.parallel_iterate([root])
            if self.time_is_up:
                break

        best_move_index = np.argmax(root.children_number_of_visits)
        best_child = root.children[best_move_index]

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Best child index:', best_move_index)
        log('Child number of visits:', root.children_number_of_visits)
        log(f'Best child has {best_child.number_of_visits} visits')
        log(f'Best child has {best_child.result_score:.4f} result_score')
        log('Child moves:', [child.move_to_get_here for child in root.children])
        log('Child visits:', [child.number_of_visits for child in root.children])
        log('Child result_scores:', [round(child.result_score, 2) for child in root.children])
        log('------------------------------------------------------------------')

        return best_child.move_to_get_here
