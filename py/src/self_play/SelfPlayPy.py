from __future__ import annotations
import copy
import random
import time

from os import PathLike

import numpy as np
from dataclasses import dataclass

from src.games.Board import Player
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.model_refresh import SearchScheduleState
from src.util import lerp
from src.mcts.MCTS import MCTS, action_probabilities
from src.mcts.MCTSNode import MCTSNode
from src.cluster.InferenceClient import InferenceClient
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.value_target import (
    ReplayValueTarget,
    TerminationReason,
    outcome_from_sample_perspective,
)
from src.self_play.curriculum import curriculum_progress
from src.settings import CURRENT_GAME, CurrentBoard, CurrentGame, CurrentGameMove, log_text, TRAINING_ARGS
from src.Encoding import get_board_result_score
from src.train.TrainingArgs import TrainingArgs
from src.util.log import log
from src.util.save_paths import model_save_path
from src.util.tensorboard import log_scalar
from src.util.timing import timeit


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
        self.start_generation_time = time.time()

        # The move at which the player resigned, if any. None if the game is still ongoing.
        self.resigned_at_move: int | None = None
        # The player who resigned, if any. None if the game is still ongoing.
        self.resignee: Player | None = None
        self.oldest_model_version: int | None = None
        self.newest_model_version: int | None = None

    def acknowledge_model_version(self, model_version: int) -> None:
        if self.oldest_model_version is None:
            self.oldest_model_version = model_version
        self.newest_model_version = model_version

    def expand(self, move: CurrentGameMove) -> SelfPlayGame:
        new_game = self.copy()
        new_game.board.make_move(move)
        new_game.played_moves.append(move)
        return new_game

    def copy(self) -> SelfPlayGame:
        game = SelfPlayGame()
        game.board = self.board.copy()
        game.memory = self.memory.copy()
        game.played_moves = self.played_moves.copy()
        game.start_generation_time = self.start_generation_time
        game.resigned_at_move = self.resigned_at_move
        game.resignee = self.resignee
        game.oldest_model_version = self.oldest_model_version
        game.newest_model_version = self.newest_model_version
        return game

    def __hash__(self) -> int:
        return self.board.quick_hash()


class SelfPlayPy:
    def __init__(self, device_id: int, args: TrainingArgs) -> None:
        self.client = InferenceClient(device_id=device_id, network_args=args.network, save_path=args.save_path)
        self.args = args.self_play
        if self.args.resignation.audit_enabled or self.args.resignation.production_enabled:
            raise ValueError('Calibrated resignation is supported only by the C++ self-play implementation.')

        self.self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0
        self.model_version: int | None = None
        self.model_refresh_acknowledgements: list[int] = []
        self.search_schedule_state: SearchScheduleState | None = None

        self.mcts: MCTS | None = None

    def update_iteration(self, iteration: int) -> None:
        self.update_search_schedule(self.search_schedule(iteration))
        self.refresh_model(iteration, model_save_path(iteration, self.client.save_path))

    def snapshot_statistics(self, _tensorboard_step: int) -> None:
        return

    def search_schedule(self, schedule_version: int) -> SearchScheduleState:
        num_full_searches = int(
            lerp(
                self.args.mcts.num_searches_per_turn / 5,
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
            num_fast_searches=max(1, num_full_searches // 4),
            endgame_shortcut_strength=0.0,
        )

    def update_search_schedule(self, schedule: SearchScheduleState) -> None:
        if schedule.num_full_searches <= self.args.mcts.num_parallel_searches:
            raise ValueError('Full-search budget must exceed the parallel-search count.')
        self.iteration = schedule.schedule_version
        self.search_schedule_state = schedule
        log_scalar(
            'dataset/num_searches_per_turn',
            schedule.num_full_searches,
            schedule.schedule_version,
        )
        mcts_args = copy.deepcopy(self.args.mcts)
        mcts_args.num_searches_per_turn = schedule.num_full_searches
        if self.mcts is None:
            self.mcts = MCTS(self.client, mcts_args)
        else:
            self.mcts.args = mcts_args

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
        if self.search_schedule_state is None:
            raise RuntimeError('Search schedule must be initialized before loading a model.')
        if discard_roots:
            log('Python self-play does not retain MCTS roots between searches.')
        self.client.refresh_model(model_version, model_path)
        self.model_version = model_version
        self.model_refresh_acknowledgements.append(model_version)

    def self_play(self) -> None:
        assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'
        if self.model_version is None:
            raise RuntimeError('A model must be refreshed before self-play starts.')
        for game in self.self_play_games:
            game.acknowledge_model_version(self.model_version)

        mcts_results = self.mcts.search(
            [spg.board for spg in self.self_play_games],
            should_run_full_search=[
                (
                    not self.args.only_store_sampled_moves  # If all moves are stored, run full search
                    or len(spg.played_moves)
                    < self.args.num_moves_after_which_to_play_greedy  # or if the game is still in the early phase
                    or len(spg.board.board.piece_map()) > 10  # or if there are many pieces on the board
                )
                and spg.resigned_at_move is None  # and the game has not been resigned
                for spg in self.self_play_games
            ],
        )

        for i, (spg, mcts_result) in enumerate(zip(self.self_play_games, mcts_results)):
            if mcts_result.is_full_search:
                spg.memory.append(
                    SelfPlayGameMemory(spg.board.copy(), mcts_result.visit_counts, mcts_result.result_score)
                )

            if CURRENT_GAME == 'chess':
                if len(spg.played_moves) >= 250:
                    # If the game is too long, end it and add it to the dataset
                    self.dataset.stats += SelfPlayDatasetStats(num_too_long_games=1)
                    self.self_play_games[i] = self._handle_end_of_game(
                        spg,
                        0.0,
                        TerminationReason.PLY_CAP,
                    )
                    continue

            spg_action_probabilities = action_probabilities(mcts_result.visit_counts)

            while np.sum(spg_action_probabilities) > 0:
                new_spg, move = self._sample_self_play_game(spg, spg_action_probabilities, mcts_result.children)

                is_duplicate = any(hash(game) == hash(new_spg) for game in self.self_play_games)
                is_repetition = move in spg.played_moves[-8:]
                if is_duplicate or is_repetition:
                    # don't play the same move twice in a row
                    # Already exploring this state, so remove the probability of this move and try again
                    spg_action_probabilities[CurrentGame.encode_move(move, spg.board)] = 0
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
                new_spg, _ = self._sample_self_play_game(
                    spg, action_probabilities(mcts_result.visit_counts), mcts_result.children
                )
                if new_spg.board.is_game_over():
                    # Game is over, add the game to the dataset
                    result = get_board_result_score(new_spg.board)
                    assert result is not None, 'Game should not be over if result is None'
                    self.self_play_games[i] = self._handle_end_of_game(new_spg, result)
                else:
                    self.self_play_games[i] = new_spg

    def _handle_end_of_game(
        self,
        spg: SelfPlayGame,
        game_outcome: float,
        termination_reason: TerminationReason = TerminationReason.NATURAL,
    ) -> SelfPlayGame:
        self._add_training_data(spg, game_outcome, termination_reason)

        if spg.resigned_at_move is not None:
            self.dataset.stats += SelfPlayDatasetStats(
                num_resignations_evaluated_to_end=1,
                num_winnable_resignations=1 if game_outcome > 0.5 and spg.resignee == spg.board.current_player else 0,
                num_moves_after_resignation=len(spg.played_moves) - spg.resigned_at_move,
            )

        return self._new_game()

    def _new_game(self) -> SelfPlayGame:
        # Create a new game instance
        new_game = SelfPlayGame()

        # Play a random moves to start the game in different states
        random_moves_to_play = int(random.random() * 8)
        for _ in range(random_moves_to_play):
            new_game = new_game.expand(random.choice(new_game.board.get_valid_moves()))
            if new_game.board.is_game_over():
                # If the game is over, start a new game
                return self._new_game()

        return new_game

    @timeit
    def _sample_self_play_game(
        self, current: SelfPlayGame, action_probabilities: np.ndarray, children: list[MCTSNode]
    ) -> tuple[SelfPlayGame, CurrentGameMove]:
        # Sample a move from the action probabilities then create a new game state with that move
        # If the game is over, add the game to the dataset and return a new game state, thereby initializing a new game

        # only use temperature for the first X moves, then simply use the most likely move
        # Keep exploration high for the first X moves, then play out as well as possible to reduce noise in the backpropagated final game results
        if len(current.played_moves) >= self.args.num_moves_after_which_to_play_greedy:
            move = CurrentGame.decode_move(np.argmax(action_probabilities).item(), current.board)
        else:
            move = self._sample_move(action_probabilities, current.board, self.args.starting_temperature)

        new_spg = current.expand(move)

        return new_spg, move

    def _sample_move(
        self, action_probabilities: np.ndarray, board: CurrentBoard, temperature: float = 1.0
    ) -> CurrentGameMove:
        assert temperature > 0, 'Temperature must be greater than 0'

        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)

        action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

        return CurrentGame.decode_move(action, board)

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

            for board, visit_counts in CurrentGame.symmetric_variations(mem.board, mem.visit_counts):
                self.dataset.add_sample(
                    board.astype(np.int8).copy(),
                    self._preprocess_visit_counts(visit_counts),
                    ReplayValueTarget.from_scores(
                        final_score=turn_game_outcome,
                        mcts_root_value=mem.result_score,
                        termination_reason=termination_reason,
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

        log_text('starting_line', ','.join(moves[:7]), self.iteration)

        if random.random() < 0.01:
            # log a game every 1% of the time
            moves_str = ','.join(moves)
            log_text(f'moves/{self.iteration}/{hash(moves_str)}', f'{result}:{moves_str}')
