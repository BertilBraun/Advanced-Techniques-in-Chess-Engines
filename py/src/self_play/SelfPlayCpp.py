from __future__ import annotations
import gc
import random
import time

import chess
import numpy as np
from dataclasses import dataclass

from src.games.Board import Player
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.util import clamp, lerp
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import CURRENT_GAME, CurrentBoard, CurrentGame, CurrentGameMove, log_text, TRAINING_ARGS
from src.Encoding import get_board_result_score
from src.train.TrainingArgs import TrainingArgs
from src.util.log import log, warn
from src.util.save_paths import model_save_path
from src.util.tensorboard import log_histogram, log_scalar
from src.util.timing import timeit

from AlphaZeroCpp import INVALID_NODE, InferenceClientParams, NodeId, MCTS, MCTSParams, MCTSResults


@dataclass(frozen=True)
class SelfPlayGameMemory:
    board: CurrentBoard
    visit_counts: list[tuple[int, int]]
    result_score: float


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.played_moves: list[CurrentGameMove] = []
        self.encoded_moves: list[int] = []
        self.already_expanded_node: NodeId = INVALID_NODE
        self.start_generation_time = time.time()

        # The move at which the player resigned, if any. None if the game is still ongoing.
        self.resigned_at_move: int | None = None
        # The player who resigned, if any. None if the game is still ongoing.
        self.resignee: Player | None = None

    def expand(self, move: CurrentGameMove) -> SelfPlayGame:
        new_game = self.copy()
        new_game.encoded_moves.append(CurrentGame.encode_move(move, new_game.board))
        new_game.board.make_move(move)
        new_game.played_moves.append(move)
        return new_game

    def copy(self) -> SelfPlayGame:
        game = SelfPlayGame()
        game.board = self.board.copy()
        game.memory = self.memory.copy()
        game.played_moves = self.played_moves.copy()
        game.encoded_moves = self.encoded_moves.copy()
        game.already_expanded_node = self.already_expanded_node
        game.start_generation_time = self.start_generation_time
        game.resigned_at_move = self.resigned_at_move
        game.resignee = self.resignee
        return game

    def __hash__(self) -> int:
        return self.board.quick_hash()

    def approximate_result_score(self) -> float:
        """Get an approximate result score for the game from the perspective of the current player."""
        # discount the score to account for uncertainty in the result
        return sign(self.board.get_approximate_result_score()) * self.board.current_player * 0.5


def sign(x: float) -> int:
    """Return the sign of x as an integer."""
    return (x > 0) - (x < 0)  # Returns 1 for positive, -1 for negative, 0 for zero


def visit_count_probabilities(visit_counts: list[tuple[int, int]], board: CurrentBoard) -> np.ndarray:
    """Convert visit counts to probabilities."""
    if not visit_counts:
        if not board.get_valid_moves() or board.is_game_over():
            raise ValueError(f'No valid moves available to sample from: {board.board.fen()}')
        return np.ones(len(visit_counts), dtype=np.float32) / len(visit_counts)

    probabilities = np.zeros(len(visit_counts), dtype=np.float32)
    for i, (_, count) in enumerate(visit_counts):
        probabilities[i] = count

    return probabilities / np.sum(probabilities)


class SelfPlayCpp:
    def __init__(self, device_id: int, args: TrainingArgs) -> None:
        self.device_id = device_id
        self.args = args.self_play
        self.save_path = args.save_path

        self.self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0

        self.mcts: MCTS | None = None
        self.num_searches_per_turn = 0

    def update_iteration(self, iteration: int) -> None:
        if len(self.dataset) > 0:
            log(f'Warning: Dataset should be empty when updating iteration. Discarding {len(self.dataset)} samples.')
        self.iteration = iteration
        self.dataset = SelfPlayDataset()

        if self.mcts is not None:
            # log inference statistics from the previous iteration
            inference_stats = self.mcts.get_inference_statistics()
            log_scalar('inference/cache_hit_rate', inference_stats.cacheHitRate, iteration - 1)
            log_scalar('inference/unique_positions', inference_stats.uniquePositions, iteration - 1)
            log_scalar('inference/cache_size_mb', inference_stats.cacheSizeMB, iteration - 1)
            log_histogram(
                'inference/nn_output_value_distribution',
                np.array(inference_stats.nnOutputValueDistribution),
                iteration - 1,
            )

            del self.mcts  # Clear the previous MCTS instance to free memory
            gc.collect()  # Force garbage collection to free memory

        self._set_mcts(iteration)

    def _set_mcts(self, iteration: int) -> None:
        """Set the MCTS parameters for the current iteration."""
        # start with 10% of the searches, scale up to 100% over the first 10% of total iterations
        self.num_searches_per_turn = int(
            lerp(
                self.args.mcts.num_searches_per_turn / 5,
                self.args.mcts.num_searches_per_turn,
                clamp(iteration * 20 / TRAINING_ARGS.num_iterations, 0.0, 1.0),
            )
        )
        self.num_searches_per_turn = self.args.mcts.num_searches_per_turn  # TODO just disable scaling for now
        assert self.num_searches_per_turn > self.args.mcts.num_parallel_searches, (
            f'Number of searches per turn ({self.num_searches_per_turn}) must be greater than number of parallel searches ({self.args.mcts.num_parallel_searches}).'
        )

        log_scalar('dataset/num_searches_per_turn', self.num_searches_per_turn, iteration)

        mcts_args = MCTSParams(
            num_parallel_searches=self.args.mcts.num_parallel_searches,
            dirichlet_alpha=self.args.mcts.dirichlet_alpha,
            dirichlet_epsilon=self.args.mcts.dirichlet_epsilon,
            c_param=self.args.mcts.c_param,
            min_visit_count=self.args.mcts.min_visit_count,
            node_reuse_discount=self.args.mcts.node_reuse_discount,
            num_threads=self.args.mcts.num_threads,
        )
        client_args = InferenceClientParams(
            self.device_id,
            currentModelPath=str(model_save_path(iteration, self.save_path).with_suffix('.jit.pt').absolute()),
            maxBatchSize=256,  # TODO: adjust based on the model size and available memory
        )
        self.mcts = MCTS(client_args, mcts_args)

        # Reset the already expanded node for all self-play games
        for spg in self.self_play_games:
            spg.already_expanded_node = INVALID_NODE

    @timeit
    def search(self, boards: list[tuple[str, NodeId, int]]) -> MCTSResults:
        assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'

        try:
            return self.mcts.search(boards)
        except Exception as e:
            warn(f'Error during MCTS search: {e}')
            self.mcts.clear_node_pool()  # Clear the node pool to free memory
            return self.search([(board[0], INVALID_NODE, board[2]) for board in boards])

    def self_play(self) -> None:
        assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'

        boards: list[tuple[str, NodeId, int]] = []
        for spg in self.self_play_games:
            should_run_full_search = (
                (
                    not self.args.only_store_sampled_moves  # If all moves are stored, run full search
                    # or if the game is still in the early phase
                    or len(spg.played_moves) < self.args.num_moves_after_which_to_play_greedy
                    or len(spg.board.board.piece_map()) > 10  # or if there are many pieces on the board
                )
                # and the game has not been resigned
                and spg.resigned_at_move is None
                # and the game has been played for a while
                and len(spg.played_moves) > 10
            )

            num_moves_to_search = (
                self.args.mcts.num_searches_per_turn
                if should_run_full_search
                else self.args.mcts.num_parallel_searches // 10
            )

            boards.append((spg.board.board.fen(), spg.already_expanded_node, num_moves_to_search))

        mcts_results: MCTSResults = self.search(boards)

        stats = mcts_results.mctsStats
        # depth, entropy, kl_divergence
        log_scalar('dataset/average_search_depth', stats.averageDepth)
        log_scalar('dataset/average_search_entropy', mcts_results.mctsStats.averageEntropy)
        log_scalar('dataset/average_search_kl_divergence', stats.averageKLDivergence)
        log_scalar('mcts/node_pool_capacity', stats.nodePoolCapacity)
        log_scalar('mcts/live_node_count', stats.liveNodeCount)

        # inference_stats = self.mcts.get_inference_statistics()
        # log_scalar('dataset/inference/cache_hit_rate', inference_stats.cacheHitRate)
        # log_scalar('dataset/inference/unique_positions', inference_stats.uniquePositions)
        # log_scalar('dataset/inference/cache_size_mb', inference_stats.cacheSizeMB)
        # log_histogram(
        #     'dataset/inference/nn_output_value_distribution', np.array(inference_stats.nnOutputValueDistribution)
        # )
        # log_scalar(
        #     'dataset/inference/average_number_of_positions_in_inference_call',
        #     inference_stats.averageNumberOfPositionsInInferenceCall,
        # )

        for i, (spg, mcts_result) in enumerate(zip(self.self_play_games, mcts_results.results)):
            was_full_searched = boards[i][2] == self.args.mcts.num_searches_per_turn
            if was_full_searched:
                spg.memory.append(SelfPlayGameMemory(spg.board.copy(), mcts_result.visits, mcts_result.result))

            if mcts_result.result < self.args.resignation_threshold and spg.resigned_at_move is None:
                # Resignation if most of the mcts searches result in a loss
                self.dataset.stats += SelfPlayDatasetStats(resignations=1)

                if random.random() < 0.1:
                    # With 10% chance, play out the game to the end to see if it was winnable
                    spg.resigned_at_move = len(spg.played_moves)
                    spg.resignee = spg.board.current_player
                else:
                    self.self_play_games[i] = self._handle_end_of_game(spg, -1.0)
                    continue

            if CURRENT_GAME == 'chess':
                # TODO make this a parameter
                if len(spg.played_moves) >= 200:
                    # If the game is too long, end it and add it to the dataset
                    self.dataset.stats += SelfPlayDatasetStats(num_too_long_games=1)
                    self.self_play_games[i] = self._handle_end_of_game(spg, spg.approximate_result_score())
                    continue

                pieces = list(spg.board.board.piece_map().values())
                white_pieces = sum(1 for piece in pieces if piece.color == chess.WHITE)
                black_pieces = sum(1 for piece in pieces if piece.color == chess.BLACK)
                if (
                    False
                    and (white_pieces < 4 or black_pieces < 4)
                    and len(spg.played_moves) >= 80
                    and random.random() < 0.2
                ):
                    # If there are only a few pieces left, and the game has been going on for a while, have a chance to end the game early and add it to the dataset to avoid noisy long games
                    self.self_play_games[i] = self._handle_end_of_game(spg, spg.approximate_result_score())
                    continue

            spg_action_probabilities = visit_count_probabilities(mcts_result.visits, spg.board)

            while np.sum(spg_action_probabilities) > 0:
                new_spg, child_index, move = self._sample_self_play_game(
                    spg, spg_action_probabilities, mcts_result.children, mcts_result.visits
                )

                is_duplicate = any(hash(game) == hash(new_spg) for game in self.self_play_games)
                is_repetition = move in spg.played_moves[-6:]
                if is_duplicate or is_repetition:
                    # don't play the same move twice in a row
                    # Already exploring this state, so remove the probability of this move and try again
                    spg_action_probabilities[child_index] = 0
                else:
                    if new_spg.board.is_game_over():
                        # Game is over, add the game to the dataset
                        result = get_board_result_score(new_spg.board)
                        assert result is not None, 'Game should not be over if result is None'
                        self.self_play_games[i] = self._handle_end_of_game(new_spg, result)
                    else:
                        self.self_play_games[i] = new_spg
                    break

            else:
                # No valid moves left which are not already being explored
                # Therefore simply pick the most likely move, and expand to different states from the most likely next state in the next iteration
                new_spg, _, _ = self._sample_self_play_game(
                    spg,
                    visit_count_probabilities(mcts_result.visits, spg.board),
                    mcts_result.children,
                    mcts_result.visits,
                )
                if new_spg.board.is_game_over():
                    # Game is over, add the game to the dataset
                    result = get_board_result_score(new_spg.board)
                    assert result is not None, 'Game should not be over if result is None'
                    self.self_play_games[i] = self._handle_end_of_game(new_spg, result)
                else:
                    self.self_play_games[i] = new_spg

    def _handle_end_of_game(self, spg: SelfPlayGame, game_outcome: float) -> SelfPlayGame:
        # assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'
        # self.mcts.get_inference_statistics()
        self._add_training_data(spg, game_outcome)

        if spg.resigned_at_move is not None:
            self.dataset.stats += SelfPlayDatasetStats(
                num_resignations_evaluated_to_end=1,
                num_winnable_resignations=1 if game_outcome > 0.5 and spg.resignee == spg.board.current_player else 0,
                num_moves_after_resignation=len(spg.played_moves) - spg.resigned_at_move,
            )

        return new_game()

    @timeit
    def _sample_self_play_game(
        self,
        current: SelfPlayGame,
        visit_count_probabilities: np.ndarray,
        children: list[NodeId],
        visit_counts: list[tuple[int, int]],
    ) -> tuple[SelfPlayGame, int, CurrentGameMove]:
        # Sample a move from the action probabilities then create a new game state with that move
        # If the game is over, add the game to the dataset and return a new game state, thereby initializing a new game

        # discourage moves which were only recently played
        for i, move in enumerate(current.encoded_moves[-10:]):
            for j, (child_move, count) in enumerate(visit_counts):
                if move == child_move:
                    visit_count_probabilities[j] /= i + 1  # discourage moves which were played recently
        visit_count_probabilities /= np.sum(visit_count_probabilities)  # normalize probabilities

        # only use temperature for the first X moves, then simply use the most likely move
        # Keep exploration high for the first X moves, then play out as well as possible to reduce noise in the backpropagated final game results
        if len(current.played_moves) >= self.args.num_moves_after_which_to_play_greedy:
            child_index = np.argmax(visit_count_probabilities).item()
        else:
            child_index = _sample_from_probabilities(visit_count_probabilities, self.args.temperature)

        move = CurrentGame.decode_move(visit_counts[child_index][0], current.board)
        new_spg = current.expand(move)
        new_spg.already_expanded_node = children[child_index]

        return new_spg, child_index, move

    @timeit
    def _add_training_data(self, spg: SelfPlayGame, game_outcome: float) -> None:
        # result: 1 if current player won, -1 if current player lost, 0 if draw

        self._log_game(spg, game_outcome)

        self.dataset.add_generation_stats(
            game_length=len(spg.played_moves),
            generation_time=time.time() - spg.start_generation_time,
        )

        for mem in spg.memory[::-1]:
            turn_game_outcome = game_outcome if mem.board.current_player == spg.board.current_player else -game_outcome

            for board, visit_counts in CurrentGame.symmetric_variations(mem.board, mem.visit_counts):
                self.dataset.add_sample(
                    board.astype(np.int8).copy(),
                    self._preprocess_visit_counts(visit_counts),
                    lerp(
                        turn_game_outcome,
                        mem.result_score,
                        clamp(self.iteration * 10 / TRAINING_ARGS.num_iterations, 0.0, 1.0)
                        * self.args.result_score_weight,
                    ),
                )

            game_outcome *= 0.995  # discount the game outcome for each move

    def _preprocess_visit_counts(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        # Remove moves which were only visited exactly as many times as required, never more
        visit_counts = [
            (move, count - self.args.mcts.min_visit_count)
            for move, count in visit_counts
            if count > self.args.mcts.min_visit_count
        ]

        return visit_counts

    def _log_game(self, spg: SelfPlayGame, result: float) -> None:
        moves: list[str] = []
        board = CurrentGame.get_initial_board()
        for move in spg.played_moves:
            encoded_move = CurrentGame.encode_move(move, board)
            moves.append(str(encoded_move))
            board.make_move(move)

        log_text('starting_line', ','.join(moves[:7]))

        if random.random() < 0.01:
            # log a game every 1% of the time
            moves_str = ','.join(moves)
            log_text(f'moves/{self.iteration}/{hash(moves_str)}', f'{result}:{moves_str}')


def _sample_from_probabilities(action_probabilities: np.ndarray, temperature: float = 1.0) -> int:
    assert temperature > 0, 'Temperature must be greater than 0'

    temperature_action_probabilities = action_probabilities ** (1 / temperature)
    temperature_action_probabilities /= np.sum(temperature_action_probabilities)

    action_index = np.random.choice(len(action_probabilities), p=temperature_action_probabilities)

    return action_index


def new_game() -> SelfPlayGame:
    # Create a new game instance
    game = SelfPlayGame()

    # Play a random moves to start the game in different states
    random_moves_to_play = int(random.random() * 8)
    for _ in range(random_moves_to_play):
        game = game.expand(random.choice(game.board.get_valid_moves()))
        if game.board.is_game_over():
            # If the game is over, start a new game
            return new_game()

    return game
