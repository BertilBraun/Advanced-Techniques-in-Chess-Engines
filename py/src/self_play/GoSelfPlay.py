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

from src.experiment.configuration import GoExperimentConfiguration
from src.util.tensorboard import log_scalar
from src.self_play.active_game import ActiveGamePolicy, ActiveGamePool
from src.self_play.completed_game import CompletedGamePublisher, SparseSearchVisit
from src.self_play.go_completed_game import (
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
class _ActiveGoGame:
    root: NativeGoSearchRoot
    started_at_seconds: float
    actions: list[int] = field(default_factory=list)
    observations: list[GoSearchObservation] = field(default_factory=list)


class GoSelfPlay(ActiveGamePolicy[_ActiveGoGame, NativeGoSearchRequest, NativeGoSearchResult]):
    def __init__(
        self,
        configuration: GoExperimentConfiguration,
        model_path: Path,
        model_generation: int,
        publisher: CompletedGamePublisher,
        device_id: int,
    ) -> None:
        self.configuration = configuration
        self.publisher = publisher
        self.model_generation = model_generation
        self.rules = GoRules(
            configuration.go.rules.komi_half_points,
            configuration.go.rules.maximum_moves,
        )
        seven_by_seven = configuration.go.representation.board_size == 7
        search_type = GoSelfPlaySearch7 if seven_by_seven else GoSelfPlaySearch9
        self.search_request_type = GoSelfPlaySearchRequest7 if seven_by_seven else GoSelfPlaySearchRequest9
        device = (
            InferenceDevice.CPU
            if configuration.training.topology.trainer.device_type == 'cpu'
            else InferenceDevice.CUDA
        )
        search = configuration.training.self_play.search
        inference = configuration.training.self_play.inference
        dimensions = search_type.inference_dimensions()
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
        self.search: NativeGoSearch = search_type(
            InferenceConfiguration(device_id, str(model_path), device),
            SelfPlaySearchParameters(
                search.num_parallel_searches,
                search.num_searches_per_turn,
                search.num_searches_per_turn,
                search.c_param,
                search.dirichlet_alpha,
                search.dirichlet_epsilon,
                search.min_visit_count,
            ),
            BatchedInferenceParameters(
                inference.inference_workers,
                inference.inference_batch_size,
                inference.outstanding_batches_per_worker,
            ),
            model_generation,
        )
        self.random = np.random.default_rng(configuration.training.random_seed + publisher.worker_id)
        pool_size = min(
            configuration.training.topology.self_play.parallel_games_per_process,
            configuration.training.self_play.inference.inference_batch_size,
        )
        self.active_games = ActiveGamePool(self, pool_size)
        self._published: list[Path] = []

    def generate(self, game_count: int) -> tuple[Path, ...]:
        if game_count <= 0:
            raise ValueError('Go self-play game count must be positive.')
        start = len(self._published)
        while len(self._published) - start < game_count:
            remaining_games = game_count - (len(self._published) - start)
            self.active_games.run_turn(remaining_games)
        return tuple(self._published[start:])

    def refresh_model(self, model_generation: int, model_path: Path) -> None:
        self.search.refresh_model(model_generation, str(model_path))
        self.model_generation = model_generation

    def run_batch(self) -> None:
        self.generate(self.configuration.training.topology.self_play.parallel_games_per_process)

    def refresh_published_model(self, model_generation: int, model_path: Path) -> None:
        self.refresh_model(model_generation, model_path)

    def snapshot_statistics(self, tensorboard_step: int) -> None:
        inference, timing = self.search.inference_statistics()
        log_scalar(
            'inference/average_number_of_positions_in_inference_call',
            inference.averageNumberOfPositionsInInferenceCall,
            tensorboard_step,
        )
        log_scalar('timing/total_time_cpp', timing.totalTime, tensorboard_step)

    def new_game(self) -> _ActiveGoGame:
        return _ActiveGoGame(self.search.new_root(self.rules), time.time())

    def build_search_request(self, game: _ActiveGoGame) -> NativeGoSearchRequest:
        return self.search_request_type(game.root, True)

    def search_active_games(self, requests: tuple[NativeGoSearchRequest, ...]) -> tuple[NativeGoSearchResult, ...]:
        return tuple(self.search.search(list(requests)).results)

    def advance_game(
        self,
        game: _ActiveGoGame,
        request: NativeGoSearchRequest,
        result: NativeGoSearchResult,
    ) -> _ActiveGoGame:
        self._play_move(game, result)
        if not game.root.is_terminal:
            return game
        self._published.append(self._publish(game))
        return self.new_game()

    def _play_move(self, game: _ActiveGoGame, result: NativeGoSearchResult) -> None:
        positive_visits = tuple(
            (visit.action_id, visit.visit_count) for visit in result.visits if visit.visit_count > 0
        )
        if not positive_visits:
            raise RuntimeError('Native Go search returned no visited action.')
        ply = len(game.actions)
        greedy_after = self.configuration.training.self_play.num_moves_after_which_to_play_greedy
        if ply >= greedy_after:
            selected_action = max(positive_visits, key=lambda visit: (visit[1], -visit[0]))[0]
            selection_mode = GoMoveSelectionMode.GREEDY
        else:
            counts = np.asarray([visit_count for _, visit_count in positive_visits], dtype=np.float64)
            temperature = self.configuration.training.self_play.starting_temperature
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
                search_budget=self.configuration.training.self_play.search.num_searches_per_turn,
                minimum_visit_count=self.configuration.training.self_play.search.min_visit_count,
            )
        )
        game.actions.append(selected_action)
        game.root.play(selected_action)

    def _publish(self, game: _ActiveGoGame) -> Path:
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
