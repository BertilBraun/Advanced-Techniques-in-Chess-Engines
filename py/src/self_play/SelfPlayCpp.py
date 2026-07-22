from __future__ import annotations
import gc
import random
import time

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from AlphaZeroCpp import MCTS, MCTSBoard, MCTSResults, MCTSRoot


import chess
import numpy as np
from dataclasses import dataclass

from src.games.Board import Player
from src.games.chess.repetition_history import REPETITION_HISTORY_PLIES, bounded_repetition_history
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.util import lerp
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.curriculum import curriculum_fade, curriculum_progress
from src.settings import CURRENT_GAME, CurrentBoard, CurrentGame, CurrentGameMove, log_text, TRAINING_ARGS
from src.Encoding import get_board_result_score
from src.train.TrainingArgs import TrainingArgs
from src.util.log import log
from src.util.save_paths import model_save_path
from src.util.tensorboard import is_tensorboard_writer_active, log_histogram, log_scalar
from src.util.timing import timeit


ENDGAME_PIECE_THRESHOLD = 8


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
        self.already_expanded_node: Optional[MCTSRoot] = None
        self.start_generation_time = time.time()

        # The move at which the player resigned, if any. None if the game is still ongoing.
        self.resigned_at_move: Optional[int] = None
        # The player who resigned, if any. None if the game is still ongoing.
        self.resignee: Optional[Player] = None
        self.low_material_termination_evaluated = False

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
        game.low_material_termination_evaluated = self.low_material_termination_evaluated
        return game

    def __hash__(self) -> int:
        return self.board.quick_hash()


def visit_count_probabilities(visit_counts: list[tuple[int, int]], board: CurrentBoard) -> np.ndarray:
    """Convert visit counts to probabilities."""
    assert len(visit_counts) > 0, f'No visit counts found for board: {board.board.fen()}'

    probabilities = np.zeros(len(visit_counts), dtype=np.float32)
    for i, (_, count) in enumerate(visit_counts):
        probabilities[i] = count

    assert np.sum(probabilities) > 0, f'No visits found for board: {board.board.fen()}'
    return probabilities / np.sum(probabilities)


class SelfPlayCpp:
    def __init__(self, device_id: int, args: TrainingArgs) -> None:
        self.device_id = device_id
        self.args = args.self_play
        self.save_path = args.save_path

        self.self_play_games: list[SelfPlayGame] = [new_game() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0

        self.mcts: MCTS | None = None  # MCTS instance for self-play, initialized in update_iteration
        self.num_searches_per_turn = 0
        self.endgame_shortcut_strength = 0.0
        self.completed_searches = 0

    def update_iteration(self, iteration: int) -> None:
        if len(self.dataset) > 0:
            log(f'Warning: Dataset should be empty when updating iteration. Discarding {len(self.dataset)} samples.')
        self.iteration = iteration
        self.dataset = SelfPlayDataset()

        if self.mcts is not None:
            # log inference statistics from the previous iteration
            inference_stats, time_info = self.mcts.get_inference_statistics()
            log_scalar('inference/cache_hit_rate', inference_stats.cacheHitRate, iteration - 1)
            log_scalar('inference/unique_positions', inference_stats.uniquePositions, iteration - 1)
            log_scalar('inference/cache_size_mb', inference_stats.cacheSizeMB, iteration - 1)
            log_scalar('inference/cache_capacity', inference_stats.cacheCapacity, iteration - 1)
            log_scalar('inference/cache_evictions', inference_stats.cacheEvictions, iteration - 1)
            log_scalar(
                'inference/cache_fingerprint_collisions',
                inference_stats.cacheFingerprintCollisions,
                iteration - 1,
            )
            log_histogram(
                'inference/nn_output_value_distribution',
                np.array(inference_stats.nnOutputValueDistribution),
                iteration - 1,
            )
            log_scalar(
                'inference/average_number_of_positions_in_inference_call',
                inference_stats.averageNumberOfPositionsInInferenceCall,
                iteration - 1,
            )

            if time_info.functionTimes:
                for element in time_info.functionTimes:
                    log_scalar(f'timing/{element.name}_percent_of_execution_time', element.percent)
                    log_scalar(f'timing/{element.name}_total_time', element.total)
                    log_scalar(f'timing/{element.name}_total_invocations', element.invocations)

                log_scalar('timing/total_traced_percent_cpp', time_info.percentRecorded)
                log_scalar('timing/total_time_cpp', time_info.totalTime)

            # Release tree handles before model-update allocations can overlap their storage.
            for self_play_game in self.self_play_games:
                self_play_game.already_expanded_node = None

            gc.collect()

        self._set_mcts(iteration)

    def _set_mcts(self, iteration: int) -> None:
        from AlphaZeroCpp import DirectSelfPlayInferenceParams, InferenceClientParams, MCTS, MCTSParams

        """Set the MCTS parameters for the current iteration."""
        # start with 10% of the searches, scale up to 100% over the first 5% of total iterations
        self.num_searches_per_turn = int(
            lerp(
                self.args.mcts.num_searches_per_turn / 2,
                self.args.mcts.num_searches_per_turn,
                curriculum_progress(
                    iteration,
                    TRAINING_ARGS.self_play_search_warmup_iterations,
                ),
            )
        )
        assert self.num_searches_per_turn > self.args.mcts.num_parallel_searches, (
            f'Number of searches per turn ({self.num_searches_per_turn}) must be greater than number of parallel searches ({self.args.mcts.num_parallel_searches}).'
        )

        log_scalar('training/num_searches_per_turn', self.num_searches_per_turn, iteration)
        self.endgame_shortcut_strength = curriculum_fade(
            iteration,
            TRAINING_ARGS.self_play_endgame_shortcut_fade_iterations,
        )
        log_scalar('training/endgame_shortcut_strength', self.endgame_shortcut_strength, iteration)

        mcts_args = MCTSParams(
            num_parallel_searches=self.args.mcts.num_parallel_searches,
            num_full_searches=self.num_searches_per_turn,
            num_fast_searches=int(
                self.num_searches_per_turn * self.args.mcts.fast_searches_proportion_of_full_searches
            ),
            dirichlet_alpha=self.args.mcts.dirichlet_alpha,
            dirichlet_epsilon=self.args.mcts.dirichlet_epsilon,
            c_param=self.args.mcts.c_param,
            min_visit_count=self.args.mcts.min_visit_count,
            num_threads=self.args.mcts.num_threads,
        )
        client_args = InferenceClientParams(
            self.device_id,
            currentModelPath=str(model_save_path(iteration, self.save_path).with_suffix('.jit.pt').absolute()),
            maxBatchSize=256,  # TODO: adjust based on the model size and available memory
            microsecondsTimeoutInferenceThread=500,  # TODO make this a parameter
            cacheCapacity=self.args.inference_cache_capacity,
        )
        if self.mcts is None:
            direct_inference_params = (
                DirectSelfPlayInferenceParams(
                    self.args.direct_inference.inference_workers,
                    self.args.direct_inference.inference_batch_size,
                    self.args.direct_inference.outstanding_batches_per_worker,
                )
                if self.args.direct_inference is not None
                else None
            )
            self.mcts = MCTS(
                client_args,
                mcts_args,
                use_inference_cache=self.args.use_inference_cache,
                direct_inference_params=direct_inference_params,
            )
        else:
            self.mcts.update(client_args.currentModelPath, mcts_args)

    @timeit
    def search(self, boards: list[MCTSBoard]) -> MCTSResults:
        assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'

        results = self.mcts.search(boards, collect_statistics=is_tensorboard_writer_active())
        self.completed_searches += results.searchesCompleted
        return results

    def self_play(self) -> None:
        from AlphaZeroCpp import MCTSBoard

        assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'
        boards: list[MCTSBoard] = []
        for spg in self.self_play_games:
            force_fast_endgame_playout = self._should_force_fast_endgame_playout(spg)
            should_run_full_search = (
                not force_fast_endgame_playout
                and spg.resigned_at_move is None
                and random.random() < self.args.mcts.playout_cap_randomization
            )

            self._prepare_search_root(spg)

            if should_run_full_search:
                # Do not reuse the node if it is a full search - Per KataGo's "RPC" (Randomized Playout Cap)
                # NOTE: Disabled for now - think about whether to re-enable this (probably not necessary for amateur games)
                # Instead - if we should run a full search, discount the visits
                spg.already_expanded_node.discount(self.args.mcts.percentage_of_node_visits_to_keep)

            boards.append(MCTSBoard(spg.already_expanded_node, should_run_full_search))

        mcts_results = self.search(boards)

        stats = mcts_results.mctsStats
        log_scalar('mcts/average_search_depth', stats.averageDepth)
        log_scalar('mcts/average_search_entropy', mcts_results.mctsStats.averageEntropy)
        log_scalar('mcts/average_search_kl_divergence', stats.averageKLDivergence)

        for i, (spg, mcts_result) in enumerate(zip(self.self_play_games, mcts_results.results)):
            if not mcts_result.visits:
                self.self_play_games[i] = self._finish_terminal_search_root(spg, mcts_result.root)
                continue

            was_full_searched = boards[i].should_run_full_search
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
                    self.self_play_games[i] = self._handle_end_of_game(spg, mcts_result.result)
                    continue

            if self._should_terminate_low_material_game(spg):
                approximate_result = spg.board.get_approximate_result_score() * spg.board.current_player
                self.dataset.stats += SelfPlayDatasetStats(
                    low_material_termination_evaluations=1,
                    low_material_terminations=1,
                    low_material_termination_material_scores=[approximate_result],
                )
                self.self_play_games[i] = self._handle_end_of_game(spg, approximate_result)
                continue

            self.self_play_games[i] = self._sample_self_play_game(
                spg,
                mcts_result.root,
                mcts_result.visits,
            )

    def _finish_terminal_search_root(self, game: SelfPlayGame, root: MCTSRoot) -> SelfPlayGame:
        if not root.is_terminal:
            raise RuntimeError('MCTS returned no visit counts for a non-terminal root.')
        finished_game = self._finish_game_after_move(game, native_game_over=True)
        assert finished_game is not None
        return finished_game

    def _prepare_search_root(self, game: SelfPlayGame) -> None:
        assert self.mcts is not None
        root = game.already_expanded_node
        if root is not None and (root.is_terminal or root.is_expanded):
            return
        history = bounded_repetition_history(game.board.board, REPETITION_HISTORY_PLIES)
        game.already_expanded_node = self.mcts.new_root_with_history(
            history.starting_fen,
            history.moves_uci,
        )

    def _should_force_fast_endgame_playout(self, game: SelfPlayGame) -> bool:
        if CURRENT_GAME != 'chess' or self.endgame_shortcut_strength <= 0.0:
            return False
        if len(game.played_moves) < self.args.num_moves_after_which_to_play_greedy:
            return False
        if len(game.board.board.piece_map()) > ENDGAME_PIECE_THRESHOLD:
            return False
        return random.random() < self.endgame_shortcut_strength

    def _should_terminate_low_material_game(self, game: SelfPlayGame) -> bool:
        if CURRENT_GAME != 'chess' or game.low_material_termination_evaluated:
            return False
        if len(game.played_moves) < self.args.low_material_termination_minimum_plies:
            return False
        if self.args.low_material_termination_probability <= 0.0:
            return False

        pieces = game.board.board.piece_map().values()
        white_piece_count = sum(piece.color == chess.WHITE for piece in pieces)
        black_piece_count = sum(piece.color == chess.BLACK for piece in pieces)
        if (
            white_piece_count >= self.args.low_material_termination_piece_threshold_per_player
            and black_piece_count >= self.args.low_material_termination_piece_threshold_per_player
        ):
            return False

        game.low_material_termination_evaluated = True
        if random.random() < self.args.low_material_termination_probability:
            return True
        self.dataset.stats += SelfPlayDatasetStats(
            low_material_termination_evaluations=1,
            low_material_termination_declines=1,
        )
        return False

    @timeit
    def _sample_self_play_game(
        self,
        current: SelfPlayGame,
        root: MCTSRoot,
        visit_counts: list[tuple[int, int]],
    ) -> SelfPlayGame:
        # Sample a move from the action probabilities then create a new game state with that move
        # If the game is over, add the game to the dataset and return a new game state, thereby initializing a new game

        children_probabilities = visit_count_probabilities(visit_counts, current.board)
        children_encoded_moves = [move for move, _ in visit_counts]

        # discourage moves which were only recently played
        for i, move in enumerate(current.encoded_moves[-10:]):
            for j, child_move in enumerate(children_encoded_moves):
                if move == child_move:
                    children_probabilities[j] /= i + 1  # discourage moves which were played recently
        children_probabilities /= np.sum(children_probabilities)  # normalize probabilities

        # only use temperature for the first X moves, then simply use the most likely move
        # Keep exploration high for the first X moves, then play out as well as possible to reduce noise in the backpropagated final game results
        if len(current.played_moves) >= self.args.num_moves_after_which_to_play_greedy:
            child_index = np.argmax(children_probabilities).item()
        else:
            # Scale down temperature from self.args.temperature to 0.1 as we approach num_moves_after_which_to_play_greedy
            game_progress = len(current.played_moves) / self.args.num_moves_after_which_to_play_greedy
            temperature = lerp(self.args.starting_temperature, self.args.final_temperature, game_progress)
            child_index = _sample_from_probabilities(children_probabilities, temperature)

        move = CurrentGame.decode_move(children_encoded_moves[child_index], current.board)
        new_spg = current.expand(move)
        new_spg.already_expanded_node = root.make_new_root(child_index)

        native_game_over = new_spg.already_expanded_node.is_terminal
        finished_game = self._finish_game_after_move(new_spg, native_game_over)
        if finished_game is None:
            return new_spg
        return finished_game

    def _finish_game_after_move(self, game: SelfPlayGame, native_game_over: bool) -> SelfPlayGame | None:
        if native_game_over or game.board.is_game_over():
            result = get_board_result_score(game.board)
            if result is None:
                assert native_game_over, 'Python reported a terminal game without a result.'
                result = 0.0
            return self._handle_end_of_game(game, result)

        maximum_game_plies = self._maximum_game_plies()
        if maximum_game_plies is not None and len(game.played_moves) >= maximum_game_plies:
            capped_game_outcome = game.board.get_approximate_result_score() * game.board.current_player
            self.dataset.stats += SelfPlayDatasetStats(
                num_too_long_games=1,
                capped_game_material_scores=[capped_game_outcome],
            )
            return self._handle_end_of_game(game, capped_game_outcome)
        return None

    def _maximum_game_plies(self) -> int | None:
        initial_maximum = self.args.maximum_game_plies
        if initial_maximum is None:
            return None
        schedule_end = self.args.maximum_game_plies_until_iteration
        final_maximum = self.args.final_maximum_game_plies
        if self.iteration >= schedule_end:
            return final_maximum
        if final_maximum is None:
            return initial_maximum
        increase = final_maximum - initial_maximum
        return initial_maximum + increase * self.iteration // schedule_end

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
                        # Scale up the proportion of the mcts result score based on the iteration number
                        # This is to ensure that the mcts result score is less significant in the early iterations where the value network is not yet well trained
                        curriculum_progress(
                            self.iteration,
                            TRAINING_ARGS.self_play_value_warmup_iterations,
                        )
                        * self.args.result_score_weight,
                    ),
                )

            # Discount the game outcome for each move in the game
            game_outcome *= 1 - self.args.game_outcome_discount_per_move

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

        if random.random() < 0.01:  # log about 1% of games
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
    random_moves_to_play = 2 + int(random.random() * 2)  # Play 2-4 random moves to start the game
    for _ in range(random_moves_to_play):
        game = game.expand(random.choice(game.board.get_valid_moves()))
        if game.board.is_game_over():
            # If the game is over, start a new game
            return new_game()

    return game
