import random
from typing import Iterable
import numpy as np
from dataclasses import dataclass

from src.train.TrainingArgs import MCTSParams
from src.util import lerp
from src.mcts.MCTSNode import MCTSNode
from src.settings import CURRENT_GAME, CurrentBoard, CurrentGame
from src.cluster.InferenceClient import InferenceClient
from src.Encoding import MoveScore, get_board_result_score
from src.util.tensorboard import log_scalar
from src.util.timing import timeit


@dataclass
class MCTSResult:
    result_score: float
    visit_counts: list[tuple[int, int]]
    children: list[MCTSNode]
    is_full_search: bool


def action_probabilities(visit_counts: Iterable[tuple[int, int]]) -> np.ndarray:
    action_probabilities = np.zeros(CurrentGame.action_size, dtype=np.float32)
    for move, visit_count in visit_counts:
        action_probabilities[move] = visit_count
    # total_visit_count = np.sum(action_probabilities)
    # action_probabilities[action_probabilities < total_visit_count * 0.01] = 0
    action_probabilities /= np.sum(action_probabilities)
    return action_probabilities  # TODO does the softmax application help?
    # NOTE: Currently, the training targets are too spikey, which means only a single move is even explored.

    # Set 0 probabilities to -inf to mask them out in the softmax
    action_probabilities[action_probabilities == 0] = -1e10

    # Multiply by X to make the softmax more peaky, which makes the Training Targets more peaky
    action_probabilities *= 8
    action_probabilities = np.exp(action_probabilities) / np.sum(np.exp(action_probabilities))
    return action_probabilities


class MCTS:
    def __init__(self, client: InferenceClient, args: MCTSParams) -> None:
        self.client = client
        self.args = args

    @timeit
    def search(
        self, inputs: list[tuple[CurrentBoard, MCTSNode | None]], should_run_full_search: list[bool] | None = None
    ) -> list[MCTSResult]:
        if not inputs:
            return []

        if should_run_full_search is None:
            should_run_full_search = [True] * len(inputs)

        assert len(inputs) == len(should_run_full_search), 'Inputs and should_run_full_search must have the same length'

        # TODO Already expanded nodes currently dont properly work. Issues:
        #  - The addition of noise to the policies does not seem to retroactively affect the exploration of other nodes enough
        #  - The node will get visited an additional X times, but should be visited a total of X times, i.e. X - node.number_of_visits times // "Playout Cap Oscillation" as per the KataGo paper.
        # TODO: Fix:Playout-cap oscillation - cap the cumulative visits a root may receive; when the cap is reached, halve all N counts (or clear them) and continue - One for child: N //= 2 per move
        # If fastplay_frequency > 0, tree search is modified as follows:
        #   - Each move is either a "low-readout" fast move, or a full, slow move.
        #   The percent of fast moves corresponds to "fastplay_frequency"
        #   - A "fast" move will:
        #     - Reuse the tree
        #     - Not mix noise in at root
        #     - Only perform 'fastplay_readouts' readouts.
        #     - Not be used as a training target.
        #   - A "slow" move will:
        #     - Clear the tree (*not* the cache).
        #     - Mix in dirichlet noise
        #     - Perform 'num_readouts' readouts.
        #     - Be noted in the Game object, to be written as a training example.
        assert all(node is None for _, node in inputs), 'Already expanded nodes are not supported anymore'

        moves = self._get_policy_with_noise([board for board, node in inputs if node is None])

        roots: list[MCTSNode] = []
        moves_index = 0

        for board, node in inputs:
            if node is not None:
                assert node.parent is None, 'Node must be a root node (be careful with memory leaks)'
                # Add noise to the children policies, then reuse the node
                node.children_policies = lerp(
                    node.children_policies,
                    np.random.dirichlet([self.args.dirichlet_alpha] * len(node.children_policies)),
                    self.args.dirichlet_epsilon,
                )
                roots.append(node)
                continue

            root = MCTSNode.root(board)
            root.expand(moves[moves_index])
            roots.append(root)

            moves_index += 1

        num_iterations_for_full_search = self.args.num_searches_per_turn // self.args.num_parallel_searches
        num_iterations_for_fast_search = num_iterations_for_full_search // 4

        assert num_iterations_for_fast_search > 0, 'num_iterations_for_fast_search must be greater than 0'

        full_roots = [root for root, full_search in zip(roots, should_run_full_search) if full_search]
        fast_roots = [root for root, full_search in zip(roots, should_run_full_search) if not full_search]

        for _ in range(self.args.min_visit_count):
            for root in full_roots:
                nodes: list[MCTSNode] = []
                for node in root.children:
                    if node.board is not None:
                        node = self._get_best_child_or_back_propagate(node, self.args.c_param)
                        if node is None:
                            continue

                    node._maybe_init_board()
                    if node.is_terminal_node:
                        result = get_board_result_score(node.board)
                        assert result is not None
                        node.back_propagate(result)
                    else:
                        nodes.append(node)

                results = self.client.inference_batch([node.board for node in nodes])

                for node, (moves, value) in zip(nodes, results):
                    node.expand(moves)
                    node.back_propagate(value)

        for _ in range(num_iterations_for_fast_search):
            self.parallel_iterate(fast_roots + full_roots)

        for _ in range(num_iterations_for_full_search - num_iterations_for_fast_search):
            self.parallel_iterate(full_roots)

        # for root in roots:
        #     print(repr(root))
        #     num_placed_stones = np.sum(root.board.board != 0)
        #     from src.util.mcts_graph import draw_mcts_graph
        #
        #     draw_mcts_graph(root, f'mcts_{num_placed_stones}.png')

        assert all(np.all(node.children_virtual_losses == 0) for node in roots), (
            'Virtual losses should be 0 after the search'
        )

        def dfs(node: MCTSNode) -> int:
            # depth dfs
            if not node or not node.is_fully_expanded:
                return 0
            return 1 + max(dfs(child) for child in node.children)

        depths = [dfs(root) for root in roots]
        average_depth = sum(depths) / len(depths)

        log_scalar('dataset/average_search_depth', average_depth)

        def entropy(node: MCTSNode) -> float:
            # calculate the entropy of the nodes visit counts
            node_number_of_visits = node.number_of_visits
            return -sum(
                visit_count / node_number_of_visits * np.log2(visit_count / node_number_of_visits)
                for visit_count in node.children_number_of_visits
                if visit_count > 0
            )

        entropies = [entropy(root) for root in roots]
        average_entropy = sum(entropies) / len(entropies)

        log_scalar('dataset/average_search_entropy', average_entropy)

        # KL divergence between the children policies and the visit counts
        def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
            """
            p, q: 1-D arrays containing non-negative numbers.
            Returns KL(p || q) with safe epsilon smoothing.
            """
            p = p.astype(np.float64) + eps
            q = q.astype(np.float64) + eps
            p /= p.sum()
            q /= q.sum()
            return np.sum(p * np.log(p / q))

        kl_divs = []
        for root in roots:
            if not root.is_fully_expanded or root.number_of_visits == 0:
                print('WARNING: Skipping KL divergence calculation for root with no visits or not fully expanded')
                continue

            priors = root.children_policies
            visit_counts = root.children_number_of_visits

            kl_divs.append(kl_divergence(priors, visit_counts))

        average_kl_div = sum(kl_divs) / len(kl_divs)
        log_scalar('dataset/average_search_kl_divergence', average_kl_div)

        # for root in roots:
        #     for i in range(len(root.children)):
        #         print(
        #             f'{root.children_policies[i]:.2f} -> {root.children_number_of_visits[i] / root.number_of_visits:.2f} ({divmod(root.children[i].encoded_move_to_get_here,  HEX_SIZE)})'
        #         )

        DISPLAY = False
        if DISPLAY:
            if CURRENT_GAME == 'hex':
                from src.games.hex.HexVisuals import display_node

                display_node(roots[0], inspect_or_search=False, search_function=self.search)  # type: ignore
            elif CURRENT_GAME == 'chess':
                from src.games.chess.ChessVisuals import display_node

                display_node(roots[0], inspect_or_search=False, search_function=self.search)  # type: ignore
            else:
                raise NotImplementedError(f'Visualization for {CURRENT_GAME} is not implemented')

        return [
            MCTSResult(
                root.result_score,
                [(child.encoded_move_to_get_here, child.number_of_visits) for child in root.children],
                root.children,
                is_full_search,
            )
            for root, is_full_search in zip(roots, should_run_full_search)
        ]

    @timeit
    def parallel_iterate(self, roots: list[MCTSNode]) -> None:
        nodes: list[MCTSNode] = []

        for _ in range(self.args.num_parallel_searches):
            for root in roots:
                node = self._get_best_child_or_back_propagate(root, self.args.c_param)
                if node is not None:
                    node.update_virtual_losses(1)
                    nodes.append(node)

        if not nodes:
            return

        results = self.client.inference_batch([node.board for node in nodes])

        for node, (moves, value) in zip(nodes, results):
            node.expand(moves)
            node.back_propagate(value)

            node.update_virtual_losses(-1)

    def _get_policy_with_noise(self, boards: list[CurrentBoard]) -> list[list[MoveScore]]:
        results = self.client.inference_batch(boards)

        return [self._add_noise(moves) for moves, _ in results]

    def _add_noise(self, moves: list[MoveScore]) -> list[MoveScore]:
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(moves))
        return [(move, lerp(policy, noise, self.args.dirichlet_epsilon)) for (move, policy), noise in zip(moves, noise)]

    def _get_best_child_or_back_propagate(self, root: MCTSNode, c_param: float) -> MCTSNode | None:
        node = root

        while node.is_fully_expanded:
            node = node.best_child(c_param)

        if node.is_terminal_node:
            result = get_board_result_score(node.board)
            assert result is not None
            node.back_propagate(result)
            return None

        return node


if __name__ == '__main__':
    from src.cluster.InferenceClient import InferenceClient
    from src.settings import CurrentBoard, TRAINING_ARGS
    from src.util.save_paths import model_save_path

    client = InferenceClient(0, TRAINING_ARGS.network, TRAINING_ARGS.save_path)
    client.load_model(model_save_path(0, TRAINING_ARGS.save_path))
    board = CurrentBoard()

    if CURRENT_GAME == 'hex':
        board.board = np.array(  # type: ignore
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, -1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 1, 0, -1, 0],
                [-1, 1, 1, 1, -1, 0, 0],
                [0, -1, 0, 1, -1, 0, 0],
                [0, -1, -1, 0, 0, 0, 0],
            ]
        )
    elif CURRENT_GAME == 'chess':
        for _ in range(60):
            board.make_move(random.choice(board.get_valid_moves()))

    mcts = MCTS(client, TRAINING_ARGS.self_play.mcts)
    mcts.search([(board, None)])
