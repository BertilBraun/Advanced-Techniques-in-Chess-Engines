import multiprocessing
from typing import Optional

from src.self_play.SelfPlayCpp import new_game
from src.util.log import log
from src.eval.Bot import Bot
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove, PLAY_C_PARAM
from AlphaZeroCpp import EvalMCTSNode, new_eval_root, EvalMCTS, InferenceClientParams, EvalMCTSParams, MCTSParams, MCTS


NUM_SEARCHES_PER_TURN = 512


class AlphaZeroBot(Bot):
    def __init__(
        self, current_model_path: str, device_id: int, max_time_to_think: float = 1.0, network_eval_only: bool = False
    ) -> None:
        super().__init__('AlphaZeroBot', max_time_to_think)

        mcts_args = EvalMCTSParams(c_param=PLAY_C_PARAM, num_threads=multiprocessing.cpu_count())
        self.inference_args = InferenceClientParams(
            device_id=device_id,
            currentModelPath=current_model_path,
            maxBatchSize=256,
        )

        self.mcts = EvalMCTS(self.inference_args, mcts_args)

        self.network_eval_only = network_eval_only

        # run some inferences to compile and warm up the model
        for _ in range(5):
            self.mcts.eval_search(new_eval_root(new_game().board.board.fen()), 64)

        self.last_root: Optional[EvalMCTSNode] = None

    def think(self, board: CurrentBoard) -> CurrentGameMove:
        if self.network_eval_only:
            network_only_mcts = MCTS(
                self.inference_args,
                MCTSParams(
                    num_fast_searches=1,
                    num_full_searches=1,
                    num_parallel_searches=1,
                    c_param=PLAY_C_PARAM,
                    num_threads=multiprocessing.cpu_count(),
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.0,
                    min_visit_count=1,
                ),
            )
            encoded_moves, value = network_only_mcts.inference(board.board.fen())
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

        root = None
        if self.last_root is not None:
            # We must look for the children of the children of the last root since 2 plies were already played
            # If the fen is the same, we can reuse the last root
            for child in self.last_root.children:
                for i, grandchild in enumerate(child.children):
                    if grandchild.fen == board_fen:
                        log('Reusing last root for the same position')
                        root = child.make_new_root(i)
                        break

        if self.last_root is None:
            root = new_eval_root(board_fen)

        assert root is not None, 'Root node must be initialized'

        res = self.mcts.eval_search(root, NUM_SEARCHES_PER_TURN)

        while not self.time_is_up:
            res = self.mcts.eval_search(root, NUM_SEARCHES_PER_TURN)

        best_child = root.best_child(PLAY_C_PARAM)
        best_move = CurrentGame.decode_move(best_child.encoded_move, board)

        if best_move not in board.get_valid_moves():
            raise Exception(f'Best move {best_move} is not legal in FEN {board.board.fen()}\n{repr(board)}')

        children = [child for child in root.children if child.visits > 0]

        log('---------------------- Alpha Zero Best Move ----------------------')
        log('Total number of searches:', root.visits)
        log(f'Results of the search: {res.result:.4f}')
        log('Max depth:', root.max_depth)
        log('Best move:', best_move)
        log(f'Best child has {best_child.visits} visits')
        log(f'Best child has {best_child.result_sum / best_child.visits:.4f} result_score')
        log('Child moves:', [child.move for child in children])
        log('Child visits:', [child.visits for child in children])
        log('Child result_scores:', [round(child.result_sum / child.visits, 2) for child in children])
        log('Child priors:', [round(child.policy, 2) for child in children])
        log('Current board FEN:', board.board.fen())
        print(repr(board))
        log('------------------------------------------------------------------')

        return best_move
