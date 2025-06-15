import numpy as np

from src.self_play.SelfPlayCpp import new_game
from src.util.log import log
from src.eval.Bot import Bot
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove, PLAY_C_PARAM
from AlphaZeroCpp import INVALID_NODE, MCTS, InferenceClientParams, MCTSParams, MCTSNode


class AlphaZeroBot(Bot):
    def __init__(
        self, current_model_path: str, device_id: int, max_time_to_think: float = 1.0, network_eval_only: bool = False
    ) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)

        NUM_PARALLEL_SEARCHES = 16

        mcts_args = MCTSParams(
            num_parallel_searches=NUM_PARALLEL_SEARCHES,
            dirichlet_alpha=0.5,  # irrelevant
            dirichlet_epsilon=0.0,
            c_param=PLAY_C_PARAM,
            min_visit_count=2,
            node_reuse_discount=1.0,
            num_threads=1,
        )
        inference_args = InferenceClientParams(
            device_id=device_id,
            currentModelPath=current_model_path,
            maxBatchSize=NUM_PARALLEL_SEARCHES,
        )

        self.mcts = MCTS(inference_args, mcts_args)

        self.network_eval_only = network_eval_only

        # run some inferences to compile and warm up the model
        for _ in range(5):
            self.mcts.search([(new_game().board.board.fen(), INVALID_NODE, 64) for _ in range(4)])

    def think(self, board: CurrentBoard) -> CurrentGameMove:
        if self.network_eval_only:
            encoded_moves, value = self.mcts.inference(board.board.fen())
            encoded_move, policy = max(encoded_moves, key=lambda move: move[1])
            move = CurrentGame.decode_move(encoded_move, board)

            log('---------------------- Alpha Zero Best Move ----------------------')
            log('Best move:', move)
            log('Best move probability:', policy)
            log('Move probabilities:', encoded_moves)
            log('Result value:', value)
            log('------------------------------------------------------------------')

            return move

        board_fen = board.board.fen()
        num_searches = 512

        res = self.mcts.search([(board_fen, INVALID_NODE, num_searches)])
        root = self.mcts.get_node(res.results[0].children[0]).parent
        assert root is not None, 'MCTS search returned no parent node'

        while not self.time_is_up:
            num_searches += 512
            res = self.mcts.search([(board_fen, root.id, num_searches)])

        # visits are a list of (Encoded Move Id, Visit Count)
        visits = res.results[0].visits
        best_move_index = np.argmax([visit[1] for visit in visits])
        best_move = CurrentGame.decode_move(visits[best_move_index][0], board)

        legal_moves = board.get_valid_moves()
        while best_move not in legal_moves:
            raise Exception(f'Best move {best_move} is not legal, trying next best move...')
            visits[best_move_index] = (-1, 0)
            best_move_index = np.argmax([visit[1] for visit in visits])
            best_move = CurrentGame.decode_move(visits[best_move_index][0], board)

        best_child = self.mcts.get_node(res.results[0].children[best_move_index])

        def max_depth(node: MCTSNode) -> int:
            if not node.children:
                return 0
            return 1 + max(max_depth(child) for child in node.children)

        children = [child for child in root.children if child.visits > 0]

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Total number of searches:', root.visits)
        log(f'Results of the search: {res.results[0].result:.4f}')
        log('Max depth:', max_depth(root))
        log('Best move:', best_move)
        log('Best child index:', best_move_index)
        log(f'Best child has {best_child.visits} visits')
        log(f'Best child has {best_child.result / best_child.visits:.4f} result_score')
        log('Child moves:', [child.move for child in children])
        log('Child visits:', [child.visits for child in children])
        log('Child result_scores:', [round(child.result / child.visits, 2) for child in children])
        log('Child priors:', [round(child.policy, 2) for child in children])
        log('Current board FEN:', board_fen)
        print(repr(board))
        log('------------------------------------------------------------------')

        return best_move
