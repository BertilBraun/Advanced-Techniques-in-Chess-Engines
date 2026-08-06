from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from AlphaZeroCpp import (
    GameSearchResult,
    GoBatchedSearch7,
    GoBatchedSearch9,
    GoPlayer,
    GoRules,
    GoSearchRoot7,
    GoSearchRoot9,
    InferenceDevice,
)

from src.experiment.chess_experiment import GoExperimentConfiguration
from src.self_play.chess_completed_game import SparseSearchVisit
from src.self_play.go_completed_game import (
    GoCompletedGame,
    GoCompletedGamePublisher,
    GoMoveSelectionMode,
    GoRepresentationMetadata,
    GoRulesMetadata,
    GoSearchObservation,
    GoTerminationReason,
)


NativeGoSearch = GoBatchedSearch7 | GoBatchedSearch9
NativeGoSearchRoot = GoSearchRoot7 | GoSearchRoot9


@dataclass
class _ActiveGoGame:
    root: NativeGoSearchRoot
    started_at_seconds: float
    actions: list[int] = field(default_factory=list)
    observations: list[GoSearchObservation] = field(default_factory=list)


class GoSelfPlay:
    def __init__(
        self,
        configuration: GoExperimentConfiguration,
        model_path: Path,
        model_generation: int,
        publisher: GoCompletedGamePublisher,
        device_id: int,
    ) -> None:
        self.configuration = configuration
        self.publisher = publisher
        self.model_generation = model_generation
        self.rules = GoRules(
            configuration.go.rules.komi_half_points,
            configuration.go.rules.maximum_moves,
        )
        search_type = GoBatchedSearch7 if configuration.go.representation.board_size == 7 else GoBatchedSearch9
        device = (
            InferenceDevice.CPU
            if configuration.training.topology.trainer.device_type == 'cpu'
            else InferenceDevice.CUDA
        )
        search = configuration.training.self_play.search
        inference = configuration.training.self_play.inference
        self.search: NativeGoSearch = search_type(
            str(model_path),
            device,
            device_id,
            inference.inference_batch_size,
            search_type.inference_dimensions(),
            search.c_param,
            search.num_searches_per_turn + search.num_parallel_searches + 2,
            model_generation,
        )
        self.random = np.random.default_rng(configuration.training.random_seed + publisher.worker_id)

    def generate(self, game_count: int) -> tuple[Path, ...]:
        if game_count <= 0:
            raise ValueError('Go self-play game count must be positive.')
        parallel_games = min(
            game_count,
            self.configuration.training.topology.self_play.parallel_games_per_process,
            self.configuration.training.self_play.inference.inference_batch_size,
        )
        active = [self._new_game() for _ in range(parallel_games)]
        published: list[Path] = []
        while len(published) < game_count:
            results = self.search.search(
                [game.root for game in active],
                self.configuration.training.self_play.search.num_searches_per_turn,
            )
            completed_indices: list[int] = []
            for index, (game, result) in enumerate(zip(active, results, strict=True)):
                self._play_move(game, result)
                if game.root.is_terminal:
                    published.append(self._publish(game))
                    completed_indices.append(index)
            for index in reversed(completed_indices):
                if len(published) + len(active) - 1 < game_count:
                    active[index] = self._new_game()
                else:
                    del active[index]
        return tuple(published)

    def refresh_model(self, model_generation: int, model_path: Path) -> None:
        self.search.refresh_model(model_generation, str(model_path))
        self.model_generation = model_generation

    def _new_game(self) -> _ActiveGoGame:
        return _ActiveGoGame(self.search.new_root(self.rules), time.time())

    def _play_move(self, game: _ActiveGoGame, result: GameSearchResult) -> None:
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
        observations = (
            tuple(observation.validated_copy(update={'sample_eligible': False}) for observation in game.observations)
            if safety_cap
            else tuple(game.observations)
        )
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
