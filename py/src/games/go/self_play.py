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
    GoPosition7,
    GoPosition9,
    GoRules,
    GoSearchRoot7,
    GoSearchRoot9,
    InferenceDevice,
    InferenceConfiguration,
    SelfPlaySearchParameters,
)

from src.games.go.configuration import GoExperimentConfiguration
from src.util.tensorboard import log_scalar
from src.games.contracts import WdlTarget
from src.self_play.active_game import CompletedGame, ContinuingGame
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SparseSearchVisit,
    TerminationReason,
)
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.self_play.worker import GameSelfPlayPolicy


NativeGoSearch = GoSelfPlaySearch7 | GoSelfPlaySearch9
NativeGoSearchRoot = GoSearchRoot7 | GoSearchRoot9
NativeGoSearchRequest = GoSelfPlaySearchRequest7 | GoSelfPlaySearchRequest9
NativeGoSearchResult = GoSelfPlaySearchResult7 | GoSelfPlaySearchResult9


@dataclass
class GoSelfPlayGame:
    identity: GameIdentity
    root: NativeGoSearchRoot
    started_at_seconds: float
    actions: list[int] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)


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
        worker_id: int,
        device_id: int,
    ) -> None:
        self.configuration = configuration
        self.model_generation: int | None = None
        self.rules = GoRules(
            configuration.go.rules.komi_half_points,
            configuration.go.rules.maximum_moves,
        )
        seven_by_seven = configuration.go.representation.board_size == 7
        self.search_type = GoSelfPlaySearch7 if seven_by_seven else GoSelfPlaySearch9
        self.search_request_type = GoSelfPlaySearchRequest7 if seven_by_seven else GoSelfPlaySearchRequest9
        self.position_type = GoPosition7 if seven_by_seven else GoPosition9
        self.device = (
            InferenceDevice.CPU
            if configuration.training.topology.trainer.device_type == 'cpu'
            else InferenceDevice.CUDA
        )
        inference = configuration.go.self_play.inference
        dimensions = self.search_type.inference_dimensions()
        representation = configuration.network_dimensions
        expected_dimensions = (
            representation.channels,
            representation.rows,
            representation.columns,
            representation.actions,
            representation.outcomes,
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
        self.resolved_parameters = self._resolve_parameters(0)
        self.search_parameters = self._native_search_parameters()
        self.inference_parameters = BatchedInferenceParameters(
            inference.inference_workers,
            inference.inference_batch_size,
            inference.outstanding_batches_per_worker,
        )
        self.search: NativeGoSearch | None = None
        self.random = np.random.default_rng(configuration.training.random_seed + worker_id)

    def refresh_model(
        self,
        model_generation: int,
        model_path: Path,
        active_games: tuple[GoSelfPlayGame, ...],
    ) -> None:
        self.resolved_parameters = self._resolve_parameters(model_generation)
        self.search_parameters = self._native_search_parameters()
        if self.search is None:
            self.search = self.search_type(
                InferenceConfiguration(self.device_id, str(model_path), self.device),
                self.search_parameters,
                self.inference_parameters,
                model_generation,
            )
        else:
            self.search.refresh_model(model_generation, str(model_path))
            self.search.update_search_schedule(self.search_parameters)
        self.model_generation = model_generation
        for game in active_games:
            game.root.reset()

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

    def new_game(self, identity: GameIdentity) -> GoSelfPlayGame:
        if self.search is None:
            raise RuntimeError('A model must be loaded before creating a Go game.')
        return GoSelfPlayGame(identity, self.search.new_root(self.position_type(self.rules)), time.time())

    def build_search_request(self, game: GoSelfPlayGame) -> NativeGoSearchRequest:
        full_search = self.random.random() < self.resolved_parameters.full_search_probability
        if full_search:
            game.root.discount(self.resolved_parameters.retained_root_visit_fraction)
        return self.search_request_type(game.root, full_search)

    def search_active_games(self, requests: tuple[NativeGoSearchRequest, ...]) -> tuple[NativeGoSearchResult, ...]:
        if self.search is None:
            raise RuntimeError('A model must be loaded before Go search starts.')
        return tuple(self.search.search(list(requests)).results)

    def advance_game(
        self,
        game: GoSelfPlayGame,
        request: NativeGoSearchRequest,
        result: NativeGoSearchResult,
    ) -> ContinuingGame[GoSelfPlayGame] | CompletedGame:
        self._play_move(game, request, result)
        if not game.root.is_terminal:
            return ContinuingGame(game)
        return CompletedGame(self._complete(game))

    def _play_move(
        self,
        game: GoSelfPlayGame,
        request: NativeGoSearchRequest,
        result: NativeGoSearchResult,
    ) -> None:
        if self.model_generation is None:
            raise RuntimeError('A model must be loaded before recording a Go move.')
        positive_visits = tuple(
            (visit.action_id, visit.visit_count) for visit in result.visits if visit.visit_count > 0
        )
        if not positive_visits:
            raise RuntimeError('Native Go search returned no visited action.')
        ply = len(game.actions)
        greedy_after = self.resolved_parameters.greedy_after_ply
        if ply >= greedy_after:
            selected_action = max(positive_visits, key=lambda visit: (visit[1], -visit[0]))[0]
        else:
            counts = np.asarray([visit_count for _, visit_count in positive_visits], dtype=np.float64)
            game_progress = ply / greedy_after
            temperature = (
                self.resolved_parameters.starting_temperature
                + (self.resolved_parameters.final_temperature - self.resolved_parameters.starting_temperature)
                * game_progress
            )
            probabilities = np.power(counts, 1.0 / temperature)
            probabilities /= probabilities.sum()
            selected_action = positive_visits[int(self.random.choice(len(positive_visits), p=probabilities))][0]
        game.observations.append(
            SearchObservation(
                ply=ply,
                model_generation=self.model_generation,
                visits=tuple(
                    SparseSearchVisit(action_id=action_id, visit_count=visit_count)
                    for action_id, visit_count in positive_visits
                ),
                root_value=result.root_value,
                selected_action_id=selected_action,
                full_search=request.full_search,
                search_budget=(
                    self.resolved_parameters.full_searches
                    if request.full_search
                    else self.resolved_parameters.fast_searches
                ),
                minimum_root_visits=self.resolved_parameters.minimum_root_visits,
                sample_weight=self.resolved_parameters.primary_sample_weight,
            )
        )
        game.actions.append(selected_action)
        game.root.play(selected_action)

    def _complete(self, game: GoSelfPlayGame) -> CompletedSelfPlayGame:
        if self.model_generation is None:
            raise RuntimeError('A model must be loaded before publishing a Go game.')
        terminal = game.root.position.terminal_result()
        safety_cap = terminal.reason.name == 'MAXIMUM_MOVES'
        final_score = game.root.position.terminal_value()
        return CompletedSelfPlayGame(
            identity=game.identity,
            created_at_seconds=time.time(),
            generation_seconds=time.time() - game.started_at_seconds,
            action_ids=tuple(game.actions),
            observations=tuple(game.observations),
            final_wdl=WdlTarget.from_scalar(final_score),
            termination_reason=(TerminationReason.MAXIMUM_PLIES if safety_cap else TerminationReason.NATURAL),
        )

    def _resolve_parameters(self, model_generation: int) -> ResolvedSelfPlayParameters:
        return self.configuration.go.self_play.resolve(
            model_generation,
            self.configuration.go.rules.maximum_moves,
        )

    def _native_search_parameters(self) -> SelfPlaySearchParameters:
        parameters = self.resolved_parameters
        return SelfPlaySearchParameters(
            parameters.parallel_searches,
            parameters.full_searches,
            parameters.fast_searches,
            parameters.exploration_constant,
            parameters.dirichlet_alpha,
            parameters.dirichlet_epsilon,
            parameters.minimum_root_visits,
        )
