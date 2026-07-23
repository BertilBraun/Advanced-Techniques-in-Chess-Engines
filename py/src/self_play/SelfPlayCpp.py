from __future__ import annotations

import random
import time
import uuid

from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AlphaZeroCpp import (
        InferenceStatistics,
        MCTS,
        MCTSBoard,
        MCTSParams,
        MCTSResult,
        MCTSResults,
        MCTSRoot,
        TimeInfo,
    )


import chess
import numpy as np

from src.games.Board import Player
from src.games.chess.repetition_history import REPETITION_HISTORY_PLIES, bounded_repetition_history
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.model_refresh import SearchScheduleState
from src.self_play.resignation import (
    CompletedResignationAudit,
    ResignationCalibrationState,
    ResignationManager,
    ResignationObservation,
    ResignationTerminationReason,
    best_child_value_from_root_perspective,
)
from src.util import lerp
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.value_target import (
    ReplayValueTarget,
    TerminationReason,
    outcome_from_sample_perspective,
)
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
    ply: int


@dataclass(frozen=True)
class SelfPlayStatisticsSnapshot:
    model_version: int | None
    inference: InferenceStatistics
    timing: TimeInfo
    completed_searches: int


class SelfPlayGame:
    def __init__(
        self,
        game_id: str | None = None,
        is_resignation_audit: bool = False,
        production_resignation_enabled: bool = False,
        resignation_threshold: float | None = None,
    ) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.played_moves: list[CurrentGameMove] = []
        self.encoded_moves: list[int] = []
        self.already_expanded_node: MCTSRoot | None = None
        self.start_generation_time = time.time()

        self.game_id = game_id if game_id is not None else uuid.uuid4().hex
        self.is_resignation_audit = is_resignation_audit
        self.production_resignation_enabled = production_resignation_enabled
        self.resignation_threshold = resignation_threshold
        self.resignation_observations: list[ResignationObservation] = []
        self.resignation_trigger_ply: int | None = None
        self.resignee: Player | None = None
        self.low_material_termination_evaluated = False
        self.oldest_model_version: int | None = None
        self.newest_model_version: int | None = None

    def acknowledge_model_version(self, model_version: int) -> None:
        if model_version < 0:
            raise ValueError('Model version must be nonnegative.')
        if self.newest_model_version is not None and model_version < self.newest_model_version:
            raise ValueError('Game model versions cannot move backwards.')
        if self.oldest_model_version is None:
            self.oldest_model_version = model_version
        self.newest_model_version = model_version

    def expand(self, move: CurrentGameMove) -> SelfPlayGame:
        new_game = self.copy()
        new_game.encoded_moves.append(CurrentGame.encode_move(move, new_game.board))
        new_game.board.make_move(move)
        new_game.played_moves.append(move)
        return new_game

    def copy(self) -> SelfPlayGame:
        game = SelfPlayGame(
            game_id=self.game_id,
            is_resignation_audit=self.is_resignation_audit,
            production_resignation_enabled=self.production_resignation_enabled,
            resignation_threshold=self.resignation_threshold,
        )
        game.board = self.board.copy()
        game.memory = self.memory.copy()
        game.played_moves = self.played_moves.copy()
        game.encoded_moves = self.encoded_moves.copy()
        game.already_expanded_node = self.already_expanded_node
        game.start_generation_time = self.start_generation_time
        game.resignation_observations = self.resignation_observations.copy()
        game.resignation_trigger_ply = self.resignation_trigger_ply
        game.resignee = self.resignee
        game.low_material_termination_evaluated = self.low_material_termination_evaluated
        game.oldest_model_version = self.oldest_model_version
        game.newest_model_version = self.newest_model_version
        return game

    def __hash__(self) -> int:
        return self.board.quick_hash()


def visit_count_probabilities(visit_counts: list[tuple[int, int]], board: CurrentBoard) -> np.ndarray:
    """Convert visit counts to probabilities."""
    if not has_positive_visit_counts(visit_counts):
        raise ValueError(f'Visit counts must contain a positive visit for board: {board.board.fen()}')

    probabilities = np.zeros(len(visit_counts), dtype=np.float32)
    for i, (_, count) in enumerate(visit_counts):
        probabilities[i] = count

    return probabilities / np.sum(probabilities)


def has_positive_visit_counts(visit_counts: list[tuple[int, int]]) -> bool:
    return (
        bool(visit_counts)
        and all(count >= 0 for _, count in visit_counts)
        and sum(count for _, count in visit_counts) > 0
    )


class SelfPlayCpp:
    def __init__(self, device_id: int, args: TrainingArgs) -> None:
        self.device_id = device_id
        self.args = args.self_play
        self.save_path = args.save_path
        self.resignation_manager = ResignationManager(self.save_path, self.args.resignation)

        self.iteration = 0
        self.model_version: int | None = None
        self.model_refresh_acknowledgements: list[int] = []
        self.search_schedule_state: SearchScheduleState | None = None
        self.dataset = SelfPlayDataset()
        self.self_play_games: list[SelfPlayGame] = [self._new_game() for _ in range(self.args.num_parallel_games)]

        self.mcts: MCTS | None = None  # MCTS instance for self-play, initialized in update_iteration
        self.num_searches_per_turn = 0
        self.endgame_shortcut_strength = 0.0
        self.completed_searches = 0

    def update_iteration(self, iteration: int) -> None:
        self.snapshot_statistics(iteration - 1)
        self.iteration = iteration
        self.update_search_schedule(self.search_schedule(iteration))
        self.refresh_model(
            iteration,
            model_save_path(iteration, self.save_path).with_suffix('.jit.pt'),
        )

    def snapshot_statistics(self, tensorboard_step: int) -> SelfPlayStatisticsSnapshot | None:
        if self.mcts is None:
            return None
        inference_stats, time_info = self.mcts.get_inference_statistics()
        log_scalar('inference/cache_hit_rate', inference_stats.cacheHitRate, tensorboard_step)
        log_scalar('inference/unique_positions', inference_stats.uniquePositions, tensorboard_step)
        log_scalar('inference/cache_size_mb', inference_stats.cacheSizeMB, tensorboard_step)
        log_scalar('inference/cache_capacity', inference_stats.cacheCapacity, tensorboard_step)
        log_scalar('inference/cache_evictions', inference_stats.cacheEvictions, tensorboard_step)
        log_scalar(
            'inference/cache_fingerprint_collisions',
            inference_stats.cacheFingerprintCollisions,
            tensorboard_step,
        )
        log_histogram(
            'inference/nn_output_value_distribution',
            np.array(inference_stats.nnOutputValueDistribution),
            tensorboard_step,
        )
        log_scalar(
            'inference/average_number_of_positions_in_inference_call',
            inference_stats.averageNumberOfPositionsInInferenceCall,
            tensorboard_step,
        )

        if time_info.functionTimes:
            for element in time_info.functionTimes:
                log_scalar(f'timing/{element.name}_percent_of_execution_time', element.percent)
                log_scalar(f'timing/{element.name}_total_time', element.total)
                log_scalar(f'timing/{element.name}_total_invocations', element.invocations)

            log_scalar('timing/total_traced_percent_cpp', time_info.percentRecorded)
            log_scalar('timing/total_time_cpp', time_info.totalTime)

        return SelfPlayStatisticsSnapshot(
            model_version=self.model_version,
            inference=inference_stats,
            timing=time_info,
            completed_searches=self.completed_searches,
        )

    def search_schedule(self, schedule_version: int) -> SearchScheduleState:
        if schedule_version < 0:
            raise ValueError('Search schedule version must be nonnegative.')
        num_full_searches = int(
            lerp(
                self.args.mcts.num_searches_per_turn / 2,
                self.args.mcts.num_searches_per_turn,
                curriculum_progress(
                    schedule_version,
                    TRAINING_ARGS.self_play_search_warmup_iterations,
                ),
            )
        )
        return SearchScheduleState(
            schedule_version=schedule_version,
            num_parallel_searches=self.args.mcts.num_parallel_searches,
            num_full_searches=num_full_searches,
            num_fast_searches=int(num_full_searches * self.args.mcts.fast_searches_proportion_of_full_searches),
            endgame_shortcut_strength=curriculum_fade(
                schedule_version,
                TRAINING_ARGS.self_play_endgame_shortcut_fade_iterations,
            ),
        )

    def update_search_schedule(self, schedule: SearchScheduleState) -> None:
        if schedule.num_full_searches <= self.args.mcts.num_parallel_searches:
            raise ValueError('Full-search budget must exceed the parallel-search count.')
        if schedule.num_fast_searches <= 0:
            raise ValueError('Fast-search budget must be positive.')

        previous_arena_capacity = self.mcts.arena_capacity if self.mcts is not None else None
        self.num_searches_per_turn = schedule.num_full_searches
        self.endgame_shortcut_strength = schedule.endgame_shortcut_strength
        self.iteration = schedule.schedule_version
        self.search_schedule_state = schedule

        log_scalar('training/num_searches_per_turn', self.num_searches_per_turn, schedule.schedule_version)
        log_scalar(
            'training/endgame_shortcut_strength',
            self.endgame_shortcut_strength,
            schedule.schedule_version,
        )
        if self.mcts is not None:
            arena_capacity_changed = self.mcts.update_search_schedule(self._native_mcts_params(schedule))
            if arena_capacity_changed:
                assert previous_arena_capacity != schedule.arena_capacity
                for game in self.self_play_games:
                    game.already_expanded_node = None

    def _native_mcts_params(self, schedule: SearchScheduleState) -> MCTSParams:
        from AlphaZeroCpp import MCTSParams

        return MCTSParams(
            num_parallel_searches=self.args.mcts.num_parallel_searches,
            num_full_searches=schedule.num_full_searches,
            num_fast_searches=schedule.num_fast_searches,
            dirichlet_alpha=self.args.mcts.dirichlet_alpha,
            dirichlet_epsilon=self.args.mcts.dirichlet_epsilon,
            c_param=self.args.mcts.c_param,
            min_visit_count=self.args.mcts.min_visit_count,
            num_threads=self.args.mcts.num_threads,
        )

    def refresh_model(
        self,
        model_version: int,
        model_path: str | PathLike[str],
        *,
        discard_roots: bool = False,
    ) -> None:
        if model_version < 0:
            raise ValueError('Model version must be nonnegative.')
        if self.model_version is not None and model_version <= self.model_version:
            raise ValueError('Model version must increase on every refresh.')
        schedule = self.search_schedule_state
        if schedule is None:
            raise RuntimeError('Search schedule must be initialized before loading a model.')
        if self.mcts is None:
            from AlphaZeroCpp import DirectSelfPlayInferenceParams, InferenceClientParams, MCTS

            client_args = InferenceClientParams(
                self.device_id,
                currentModelPath=str(model_path),
                maxBatchSize=256,  # TODO: adjust based on the model size and available memory
                microsecondsTimeoutInferenceThread=500,  # TODO make this a parameter
                cacheCapacity=self.args.inference_cache_capacity,
            )
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
                self._native_mcts_params(schedule),
                use_inference_cache=self.args.use_inference_cache,
                direct_inference_params=direct_inference_params,
                initial_model_version=model_version,
            )
        else:
            self.mcts.refresh_model(model_version, str(model_path))
        self.model_version = model_version
        self.model_refresh_acknowledgements.append(model_version)
        log_scalar('inference/acknowledged_model_version', model_version)
        if discard_roots:
            for game in self.self_play_games:
                game.already_expanded_node = None

    def _set_mcts(self, iteration: int) -> None:
        self.update_search_schedule(self.search_schedule(iteration))
        self.refresh_model(
            iteration,
            model_save_path(iteration, self.save_path).with_suffix('.jit.pt'),
        )

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
            if self.model_version is None:
                raise RuntimeError('A model must be refreshed before self-play starts.')
            spg.acknowledge_model_version(self.model_version)
            force_fast_endgame_playout = self._should_force_fast_endgame_playout(spg)
            should_run_full_search = self._should_run_full_search(
                spg,
                force_fast_endgame_playout,
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

            if not has_positive_visit_counts(mcts_result.visits):
                log(f'Discarding self-play game after zero-visit MCTS root: {spg.board.board.fen()}')
                self.self_play_games[i] = self._new_game()
                continue

            was_full_searched = boards[i].should_run_full_search
            if was_full_searched:
                spg.memory.append(
                    SelfPlayGameMemory(
                        spg.board.copy(),
                        mcts_result.visits,
                        mcts_result.result,
                        len(spg.played_moves),
                    )
                )
                if self._record_resignation_observation(spg, mcts_result):
                    if spg.is_resignation_audit:
                        self.dataset.stats += SelfPlayDatasetStats(hypothetical_resignations=1)
                    else:
                        self.dataset.stats += SelfPlayDatasetStats(actual_resignations=1)
                        self.self_play_games[i] = self._handle_resignation(spg)
                        continue

            if self._should_terminate_low_material_game(spg):
                approximate_result = spg.board.get_approximate_result_score() * spg.board.current_player
                self.dataset.stats += SelfPlayDatasetStats(
                    low_material_termination_evaluations=1,
                    low_material_terminations=1,
                    low_material_termination_material_scores=[approximate_result],
                )
                self.self_play_games[i] = self._handle_end_of_game(
                    spg,
                    approximate_result,
                    ResignationTerminationReason.LOW_MATERIAL,
                )
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

    def _record_resignation_observation(self, game: SelfPlayGame, result: MCTSResult) -> bool:
        if (
            not game.is_resignation_audit
            and not game.production_resignation_enabled
            or len(game.played_moves) < self.args.resignation.minimum_eligible_ply
            or game.resignation_trigger_ply is not None
        ):
            return False

        best_child_value = best_child_value_from_root_perspective(result.root.children)
        if best_child_value is None:
            return False
        observation = ResignationObservation(
            model_version=self.iteration,
            ply=len(game.played_moves),
            side_to_move=game.board.current_player,
            root_value=result.result,
            best_child_value=best_child_value,
        )
        game.resignation_observations.append(observation)
        threshold = game.resignation_threshold
        if threshold is None or observation.root_value >= threshold or observation.best_child_value >= threshold:
            return False

        game.resignation_trigger_ply = observation.ply
        game.resignee = game.board.current_player
        return True

    def _new_game(self) -> SelfPlayGame:
        assignment = self.resignation_manager.assignment(self.iteration)
        game = new_game(
            is_resignation_audit=assignment.is_audit_game,
            production_resignation_enabled=assignment.production_resignation_enabled,
            resignation_threshold=assignment.governing_threshold,
        )
        if game.is_resignation_audit:
            self.dataset.stats += SelfPlayDatasetStats(resignation_audit_games_started=1)
        return game

    def _handle_resignation(self, game: SelfPlayGame) -> SelfPlayGame:
        return self._handle_end_of_game(
            game,
            -1.0,
            ResignationTerminationReason.RESIGNATION,
        )

    def _log_resignation_state(self, state: ResignationCalibrationState) -> None:
        log_scalar(
            'resignation/calibrated_threshold',
            state.selected_threshold if state.selected_threshold is not None else -1.0,
        )
        log_scalar(
            'resignation/calibration_safe',
            float(state.selected_threshold_is_safe),
        )
        for statistics in state.threshold_statistics:
            threshold_name = f'{abs(statistics.threshold):.2f}'
            log_scalar(
                f'resignation/candidates/{threshold_name}/completed_triggers',
                statistics.completed_triggers,
            )
            log_scalar(
                f'resignation/candidates/{threshold_name}/false_non_loss_rate',
                statistics.false_non_loss_rate,
            )
            log_scalar(
                f'resignation/candidates/{threshold_name}/upper_confidence',
                statistics.false_non_loss_upper_confidence,
            )

    def _should_force_fast_endgame_playout(self, game: SelfPlayGame) -> bool:
        if CURRENT_GAME != 'chess' or self.endgame_shortcut_strength <= 0.0:
            return False
        if len(game.played_moves) < self.args.num_moves_after_which_to_play_greedy:
            return False
        if len(game.board.board.piece_map()) > ENDGAME_PIECE_THRESHOLD:
            return False
        return random.random() < self.endgame_shortcut_strength

    def _should_run_full_search(
        self,
        game: SelfPlayGame,
        force_fast_endgame_playout: bool,
    ) -> bool:
        return (
            not force_fast_endgame_playout
            and game.resignation_trigger_ply is None
            and random.random() < self.args.mcts.playout_cap_randomization
        )

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
            return self._handle_end_of_game(game, result, ResignationTerminationReason.NATURAL)

        maximum_game_plies = self._maximum_game_plies()
        if maximum_game_plies is not None and len(game.played_moves) >= maximum_game_plies:
            capped_game_outcome = game.board.get_approximate_result_score() * game.board.current_player
            self.dataset.stats += SelfPlayDatasetStats(
                num_too_long_games=1,
                capped_game_material_scores=[capped_game_outcome],
            )
            return self._handle_end_of_game(
                game,
                capped_game_outcome,
                ResignationTerminationReason.PLY_CAP,
            )
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

    def _handle_end_of_game(
        self,
        spg: SelfPlayGame,
        game_outcome: float,
        termination_reason: ResignationTerminationReason = ResignationTerminationReason.NATURAL,
    ) -> SelfPlayGame:
        # assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'
        # self.mcts.get_inference_statistics()
        self._add_training_data(spg, game_outcome, _replay_termination_reason(termination_reason))

        if spg.is_resignation_audit:
            audit = CompletedResignationAudit(
                game_id=spg.game_id,
                observations=tuple(spg.resignation_observations),
                audit_cutoff_threshold=spg.resignation_threshold,
                audit_cutoff_ply=spg.resignation_trigger_ply,
                final_current_player=spg.board.current_player,
                game_outcome=game_outcome,
                termination_reason=termination_reason,
            )
            state = self.resignation_manager.record_completed_audit(audit)
            self._log_resignation_state(state)
            audit_trigger = next(
                (
                    observation
                    for observation in spg.resignation_observations
                    if spg.resignation_threshold is not None
                    and observation.root_value < spg.resignation_threshold
                    and observation.best_child_value < spg.resignation_threshold
                ),
                None,
            )
            recovered_outcome = (
                game_outcome
                if audit_trigger is not None and audit_trigger.side_to_move == spg.board.current_player
                else -game_outcome
            )
            is_natural_trigger = (
                audit_trigger is not None and termination_reason is ResignationTerminationReason.NATURAL
            )
            is_false_non_loss = is_natural_trigger and recovered_outcome >= 0.0
            continuation_plies = len(spg.played_moves) - audit_trigger.ply if audit_trigger is not None else 0
            fast_searches = int(self.num_searches_per_turn * self.args.mcts.fast_searches_proportion_of_full_searches)
            self.dataset.stats += SelfPlayDatasetStats(
                resignation_audit_games_completed=1,
                resignation_audit_natural_triggers=int(is_natural_trigger),
                resignation_audit_capped_triggers=int(
                    audit_trigger is not None and termination_reason is not ResignationTerminationReason.NATURAL
                ),
                resignation_audit_recovered_wins=int(is_natural_trigger and recovered_outcome == 1.0),
                resignation_audit_recovered_draws=int(is_natural_trigger and recovered_outcome == 0.0),
                resignation_audit_recovered_losses=int(is_natural_trigger and recovered_outcome == -1.0),
                resignation_audit_white_triggers=int(is_natural_trigger and audit_trigger.side_to_move == 1),
                resignation_audit_black_triggers=int(is_natural_trigger and audit_trigger.side_to_move == -1),
                resignation_audit_white_false_non_losses=int(is_false_non_loss and audit_trigger.side_to_move == 1),
                resignation_audit_black_false_non_losses=int(is_false_non_loss and audit_trigger.side_to_move == -1),
                resignation_audit_root_value_abs_sum=sum(
                    abs(observation.root_value) for observation in spg.resignation_observations
                ),
                resignation_audit_root_value_count=len(spg.resignation_observations),
                resignation_audit_continuation_plies=continuation_plies,
                resignation_audit_estimated_searches_saved=continuation_plies * fast_searches,
            )

        return self._new_game()

    @timeit
    def _add_training_data(
        self,
        spg: SelfPlayGame,
        game_outcome: float,
        termination_reason: TerminationReason,
    ) -> None:
        # result: 1 if current player won, -1 if current player lost, 0 if draw

        self._log_game(spg, game_outcome)
        if spg.oldest_model_version is None or spg.newest_model_version is None:
            raise RuntimeError('Completed game has no acknowledged inference model.')
        self.dataset.stats += SelfPlayDatasetStats(
            game_model_version_ranges=[(spg.oldest_model_version, spg.newest_model_version)],
        )

        self.dataset.add_generation_stats(
            game_length=len(spg.played_moves),
            generation_time=time.time() - spg.start_generation_time,
        )

        for mem in spg.memory[::-1]:
            turn_game_outcome = outcome_from_sample_perspective(
                game_outcome,
                final_current_player=spg.board.current_player,
                sample_current_player=mem.board.current_player,
            )

            board = CurrentGame.get_canonical_board(mem.board).astype(np.int8).copy()
            current_piece_count, opponent_piece_count = CurrentGame.replay_piece_counts(board)
            self.dataset.add_sample(
                board,
                self._preprocess_visit_counts(mem.visit_counts),
                ReplayValueTarget.from_scores(
                    final_score=turn_game_outcome,
                    mcts_root_value=mem.result_score,
                    termination_reason=termination_reason,
                ),
                ReplaySampleMetadata(
                    ply=mem.ply,
                    current_player_piece_count=current_piece_count,
                    opponent_piece_count=opponent_piece_count,
                ),
            )

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


def _replay_termination_reason(reason: ResignationTerminationReason) -> TerminationReason:
    match reason:
        case ResignationTerminationReason.NATURAL:
            return TerminationReason.NATURAL
        case ResignationTerminationReason.RESIGNATION:
            return TerminationReason.RESIGNATION
        case ResignationTerminationReason.PLY_CAP:
            return TerminationReason.PLY_CAP
        case ResignationTerminationReason.LOW_MATERIAL:
            return TerminationReason.MATERIAL_ADJUDICATION


def new_game(
    is_resignation_audit: bool = False,
    production_resignation_enabled: bool = False,
    resignation_threshold: float | None = None,
) -> SelfPlayGame:
    # Create a new game instance
    game = SelfPlayGame(
        is_resignation_audit=is_resignation_audit,
        production_resignation_enabled=production_resignation_enabled,
        resignation_threshold=resignation_threshold,
    )

    # Play a random moves to start the game in different states
    random_moves_to_play = 2 + int(random.random() * 2)  # Play 2-4 random moves to start the game
    for _ in range(random_moves_to_play):
        game = game.expand(random.choice(game.board.get_valid_moves()))
        if game.board.is_game_over():
            # If the game is over, start a new game
            return new_game(
                is_resignation_audit=is_resignation_audit,
                production_resignation_enabled=production_resignation_enabled,
                resignation_threshold=resignation_threshold,
            )

    return game
