from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from AlphaZeroCpp import (
    BatchedInferenceParameters,
    GoSelfPlaySearch7,
    GoSelfPlaySearch9,
    GoSelfPlaySearchRequest7,
    GoSelfPlaySearchRequest9,
    GoSelfPlaySearchResult7,
    GoSelfPlaySearchResult9,
    GoPlayer,
    GoRules,
    GoSearchRoot7,
    GoSearchRoot9,
    InferenceDevice,
    InferenceConfiguration,
    SelfPlaySearchParameters,
)

from src.games.go.configuration import GoExperimentConfiguration
from src.util.tensorboard import log_scalar
from src.self_play.completed_game import CompletedGamePublisher, SparseSearchVisit
from src.self_play.worker import GameSelfPlayPolicy
from src.games.go.completed_game import (
    GoCompletedGame,
    GoMoveSelectionMode,
    GoRepresentationMetadata,
    GoRulesMetadata,
    GoSearchObservation,
    GoTerminationReason,
)


NativeGoSearch = GoSelfPlaySearch7 | GoSelfPlaySearch9
NativeGoSearchRoot = GoSearchRoot7 | GoSearchRoot9
NativeGoSearchRequest = GoSelfPlaySearchRequest7 | GoSelfPlaySearchRequest9
NativeGoSearchResult = GoSelfPlaySearchResult7 | GoSelfPlaySearchResult9


@dataclass
class GoSelfPlayGame:
    root: NativeGoSearchRoot
    started_at_seconds: float
    actions: list[int] = field(default_factory=list)
    observations: list[GoSearchObservation] = field(default_factory=list)


class GoSelfPlayPolicy(
    GameSelfPlayPolicy[
        GoSelfPlayGame,
        NativeGoSearchRequest,
        NativeGoSearchResult,
        None,
    ]
):
    def __init__(
        self,
        configuration: GoExperimentConfiguration,
        publisher: CompletedGamePublisher,
        device_id: int,
    ) -> None:
        self.configuration = configuration
        self.publisher = publisher
        self.model_generation: int | None = None
        self.rules = GoRules(
            configuration.go.rules.komi_half_points,
            configuration.go.rules.maximum_moves,
        )
        seven_by_seven = configuration.go.representation.board_size == 7
        self.search_type = GoSelfPlaySearch7 if seven_by_seven else GoSelfPlaySearch9
        self.search_request_type = GoSelfPlaySearchRequest7 if seven_by_seven else GoSelfPlaySearchRequest9
        self.device = (
            InferenceDevice.CPU
            if configuration.training.topology.trainer.device_type == 'cpu'
            else InferenceDevice.CUDA
        )
        search = configuration.go.self_play.search
        inference = configuration.go.self_play.inference
        dimensions = self.search_type.inference_dimensions()
        representation = configuration.go.representation
        expected_dimensions = (
            representation.channel_count,
            representation.board_size,
            representation.board_size,
            representation.action_count,
            3,
        )
        actual_dimensions = (
            dimensions.channels,
            dimensions.rows,
            dimensions.columns,
            dimensions.actions,
            dimensions.outcomes,
        )
        if actual_dimensions != expected_dimensions:
            raise ValueError('Resolved Go representation disagrees with the native template dimensions.')
        self.device_id = device_id
        self.search_parameters = SelfPlaySearchParameters(
            search.num_parallel_searches,
            search.num_searches_per_turn,
            search.num_searches_per_turn,
            search.c_param,
            search.dirichlet_alpha,
            search.dirichlet_epsilon,
            search.min_visit_count,
        )
        self.inference_parameters = BatchedInferenceParameters(
            inference.inference_workers,
            inference.inference_batch_size,
            inference.outstanding_batches_per_worker,
        )
        self.search: NativeGoSearch | None = None
        self.random = np.random.default_rng(configuration.training.random_seed + publisher.worker_id)

    def refresh_model(
        self,
        model_generation: int,
        model_path: Path,
        active_games: tuple[GoSelfPlayGame, ...],
    ) -> None:
        if self.search is None:
            self.search = self.search_type(
                InferenceConfiguration(self.device_id, str(model_path), self.device),
                self.search_parameters,
                self.inference_parameters,
                model_generation,
            )
        else:
            self.search.refresh_model(model_generation, str(model_path))
        self.model_generation = model_generation

    def snapshot_statistics(self, tensorboard_step: int) -> None:
        if self.search is None:
            return
        inference, timing = self.search.inference_statistics()
        log_scalar(
            'inference/average_number_of_positions_in_inference_call',
            inference.averageNumberOfPositionsInInferenceCall,
            tensorboard_step,
        )
        log_scalar('timing/total_time_cpp', timing.totalTime, tensorboard_step)

    def new_game(self) -> GoSelfPlayGame:
        if self.search is None:
            raise RuntimeError('A model must be loaded before creating a Go game.')
        return GoSelfPlayGame(self.search.new_root(self.rules), time.time())

    def build_search_request(self, game: GoSelfPlayGame) -> NativeGoSearchRequest:
        return self.search_request_type(game.root, True)

    def search_active_games(self, requests: tuple[NativeGoSearchRequest, ...]) -> tuple[NativeGoSearchResult, ...]:
        if self.search is None:
            raise RuntimeError('A model must be loaded before Go search starts.')
        return tuple(self.search.search(list(requests)).results)

    def advance_game(
        self,
        game: GoSelfPlayGame,
        request: NativeGoSearchRequest,
        result: NativeGoSearchResult,
    ) -> GoSelfPlayGame:
        self._play_move(game, result)
        if not game.root.is_terminal:
            return game
        self._publish(game)
        return self.new_game()

    def _play_move(self, game: GoSelfPlayGame, result: NativeGoSearchResult) -> None:
        if self.model_generation is None:
            raise RuntimeError('A model must be loaded before recording a Go move.')
        positive_visits = tuple(
            (visit.action_id, visit.visit_count) for visit in result.visits if visit.visit_count > 0
        )
        if not positive_visits:
            raise RuntimeError('Native Go search returned no visited action.')
        ply = len(game.actions)
        greedy_after = self.configuration.go.self_play.num_moves_after_which_to_play_greedy
        if ply >= greedy_after:
            selected_action = max(positive_visits, key=lambda visit: (visit[1], -visit[0]))[0]
            selection_mode = GoMoveSelectionMode.GREEDY
        else:
            counts = np.asarray([visit_count for _, visit_count in positive_visits], dtype=np.float64)
            temperature = self.configuration.go.self_play.starting_temperature
            probabilities = np.power(counts, 1.0 / temperature)
            probabilities /= probabilities.sum()
            selected_action = positive_visits[int(self.random.choice(len(positive_visits), p=probabilities))][0]
            selection_mode = GoMoveSelectionMode.TEMPERATURE
        game.observations.append(
            GoSearchObservation(
                ply=ply,
                model_generation=self.model_generation,
                legal_action_ids=tuple(sorted(game.root.position.legal_actions())),
                visits=tuple(
                    SparseSearchVisit(action_id=action_id, visit_count=visit_count)
                    for action_id, visit_count in positive_visits
                ),
                root_value=result.root_value,
                selected_action_id=selected_action,
                move_selection_mode=selection_mode,
                search_budget=self.configuration.go.self_play.search.num_searches_per_turn,
                minimum_visit_count=self.configuration.go.self_play.search.min_visit_count,
            )
        )
        game.actions.append(selected_action)
        game.root.play(selected_action)

    def _publish(self, game: GoSelfPlayGame) -> Path:
        if self.model_generation is None:
            raise RuntimeError('A model must be loaded before publishing a Go game.')
        terminal = game.root.position.terminal_result()
        safety_cap = terminal.reason.name == 'MAXIMUM_MOVES'
        observations = tuple(game.observations)
        final_score = game.root.position.terminal_value()
        completed = GoCompletedGame(
            identity=self.publisher.reserve_identity(),
            rules=GoRulesMetadata(
                komi_half_points=self.configuration.go.rules.komi_half_points,
                maximum_moves=self.configuration.go.rules.maximum_moves,
            ),
            representation=GoRepresentationMetadata(board_size=self.configuration.go.representation.board_size),
            model_generation=self.model_generation,
            minimum_model_generation=min(observation.model_generation for observation in observations),
            created_at_seconds=time.time(),
            generation_seconds=time.time() - game.started_at_seconds,
            actions=tuple(game.actions),
            final_current_player=1 if game.root.position.player == GoPlayer.BLACK else -1,
            final_score=final_score,
            termination_reason=(GoTerminationReason.MAXIMUM_MOVES if safety_cap else GoTerminationReason.TWO_PASSES),
            observations=observations,
        )
        return self.publisher.publish(completed)
