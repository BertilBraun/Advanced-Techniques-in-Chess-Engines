from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING, Generic
from uuid import uuid4

import numpy as np

from src.games.contracts import WdlTarget
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SparseSearchVisit,
    TerminationReason,
    publish_completed_self_play_game,
)
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.self_play.native_search import NativeRequestT, NativeResultT, NativeRootT, NativeSearchT, PositionT
from src.util.tensorboard import log_scalar


if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceStatistics, TimeInfo

    from src.games.implementation import GameImplementation
    from src.training.checkpoint import CheckpointReference


@dataclass
class ActiveSelfPlayGame(Generic[NativeRootT]):
    identity: GameIdentity
    root: NativeRootT
    started_at_seconds: float
    action_ids: list[int] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)


@dataclass(frozen=True)
class SelfPlayStatisticsSnapshot:
    model_generation: int
    completed_searches: int
    inference: InferenceStatistics
    timing: TimeInfo


class SelfPlayWorker(Generic[PositionT, NativeRootT, NativeRequestT, NativeResultT, NativeSearchT]):
    def __init__(
        self,
        game: GameImplementation[PositionT, NativeSearchT],
        parallel_game_count: int,
        worker_id: int,
        device_id: int,
        inbox_path: Path,
    ) -> None:
        if parallel_game_count <= 0:
            raise ValueError('Self-play requires at least one parallel game.')
        self.game = game
        self.parallel_game_count = parallel_game_count
        self.worker_id = worker_id
        self.device_id = device_id
        self.inbox_path = inbox_path
        self.random = np.random.default_rng(game.training.random_seed + worker_id)
        self.process_instance_id = uuid4()
        self.next_game_number = 0
        self.model_generation: int | None = None
        self.parameters: ResolvedSelfPlayParameters | None = None
        self.search: NativeSearchT | None = None
        self.active_games: list[ActiveSelfPlayGame[NativeRootT]] = []
        self.completed_searches = 0

    def run_batch(self) -> None:
        search, parameters = self._loaded_runtime()
        requests: list[NativeRequestT] = []
        for active_game in self.active_games:
            full_search = self.random.random() < parameters.full_search_probability
            if full_search:
                active_game.root.discount(parameters.retained_root_visit_fraction)
            requests.append(search.request(active_game.root, full_search))
        batch = search.search(requests, collect_statistics=False)
        if len(batch.results) != len(self.active_games):
            raise RuntimeError('Batched self-play search returned the wrong result count.')
        self.completed_searches += batch.simulations_completed
        next_games: list[ActiveSelfPlayGame[NativeRootT]] = []
        for active_game, request, result in zip(self.active_games, requests, batch.results, strict=True):
            completed = self._advance_game(active_game, request, result, parameters)
            if completed is None:
                next_games.append(active_game)
            else:
                publish_completed_self_play_game(self.inbox_path, completed)
                next_games.append(self._new_game(search, parameters))
        self.active_games = next_games

    def refresh_published_model(self, checkpoint: CheckpointReference) -> None:
        parameters = self.game.self_play_parameters_at(checkpoint.generation)
        if self.search is None:
            self.search = self.game.create_native_search(self.device_id, checkpoint, parameters)
        else:
            self.search.refresh_model(checkpoint.generation, str(checkpoint.inference_model_path))
            self.search.update_search_schedule(self.game.native_search_parameters(parameters))
        self.parameters = parameters
        self.model_generation = checkpoint.generation
        if not self.active_games:
            self.active_games = [self._new_game(self.search, parameters) for _ in range(self.parallel_game_count)]
        else:
            for active_game in self.active_games:
                active_game.root.reset()

    def snapshot_statistics(self) -> SelfPlayStatisticsSnapshot:
        search, _ = self._loaded_runtime()
        assert self.model_generation is not None
        inference, timing = search.inference_statistics()
        log_scalar(
            'inference/average_number_of_positions_in_inference_call',
            inference.averageNumberOfPositionsInInferenceCall,
            self.model_generation,
        )
        log_scalar('timing/total_time_cpp', timing.totalTime, self.model_generation)
        return SelfPlayStatisticsSnapshot(
            model_generation=self.model_generation,
            completed_searches=self.completed_searches,
            inference=inference,
            timing=timing,
        )

    def _new_game(
        self,
        search: NativeSearchT,
        parameters: ResolvedSelfPlayParameters,
    ) -> ActiveSelfPlayGame[NativeRootT]:
        while True:
            position = self.game.state.initial_position()
            action_ids: list[int] = []
            for _ in range(parameters.random_opening_plies):
                legal_actions = self.game.state.legal_action_ids(position)
                action_id = int(self.random.choice(legal_actions))
                action_ids.append(action_id)
                position = self.game.state.child_position(position, action_id)
                if self.game.state.natural_terminal_wdl(position) is not None:
                    break
            if self.game.state.natural_terminal_wdl(position) is None:
                return ActiveSelfPlayGame(
                    identity=self._next_identity(),
                    root=search.new_root(position),
                    started_at_seconds=time.time(),
                    action_ids=action_ids,
                )

    def _advance_game(
        self,
        active_game: ActiveSelfPlayGame[NativeRootT],
        request: NativeRequestT,
        result: NativeResultT,
        parameters: ResolvedSelfPlayParameters,
    ) -> CompletedSelfPlayGame | None:
        assert self.model_generation is not None
        visits = tuple(
            SparseSearchVisit(action_id=visit.action_id, visit_count=visit.visit_count)
            for visit in result.visits
            if visit.visit_count > 0
        )
        if not visits:
            raise RuntimeError('Native search returned no visited action for a nonterminal root.')
        ply = len(active_game.action_ids)
        selected_action_id = self._select_action(visits, ply, parameters)
        active_game.observations.append(
            SearchObservation(
                ply=ply,
                model_generation=self.model_generation,
                visits=visits,
                root_value=result.root_value,
                selected_action_id=selected_action_id,
                full_search=request.full_search,
                sample_weight=parameters.primary_sample_weight,
                search_budget=parameters.full_searches if request.full_search else parameters.fast_searches,
                minimum_root_visits=parameters.minimum_root_visits,
            )
        )
        active_game.action_ids.append(selected_action_id)
        active_game.root = result.root
        active_game.root.play(selected_action_id)
        natural_wdl = self.game.state.natural_terminal_wdl(active_game.root.position)
        if natural_wdl is not None:
            return self._complete(active_game, natural_wdl, TerminationReason.NATURAL)
        if parameters.maximum_game_plies is not None and len(active_game.action_ids) >= parameters.maximum_game_plies:
            final_wdl = self.game.state.adjudicated_wdl(active_game.root.position, TerminationReason.MAXIMUM_PLIES)
            return self._complete(active_game, final_wdl, TerminationReason.MAXIMUM_PLIES)
        return None

    def _select_action(
        self,
        visits: tuple[SparseSearchVisit, ...],
        ply: int,
        parameters: ResolvedSelfPlayParameters,
    ) -> int:
        if ply >= parameters.greedy_after_ply:
            return min(visits, key=lambda visit: (-visit.visit_count, visit.action_id)).action_id
        progress = ply / parameters.greedy_after_ply
        temperature = (
            parameters.starting_temperature
            + (parameters.final_temperature - parameters.starting_temperature) * progress
        )
        counts = np.asarray([visit.visit_count for visit in visits], dtype=np.float64)
        probabilities = np.power(counts, 1.0 / temperature)
        probabilities /= probabilities.sum()
        return visits[int(self.random.choice(len(visits), p=probabilities))].action_id

    @staticmethod
    def _complete(
        active_game: ActiveSelfPlayGame[NativeRootT],
        final_wdl: WdlTarget,
        reason: TerminationReason,
    ) -> CompletedSelfPlayGame:
        return CompletedSelfPlayGame(
            identity=active_game.identity,
            created_at_seconds=time.time(),
            generation_seconds=time.time() - active_game.started_at_seconds,
            action_ids=tuple(active_game.action_ids),
            observations=tuple(active_game.observations),
            final_wdl=final_wdl,
            termination_reason=reason,
        )

    def _loaded_runtime(self) -> tuple[NativeSearchT, ResolvedSelfPlayParameters]:
        if self.search is None or self.parameters is None:
            raise RuntimeError('A model must be loaded before self-play starts.')
        return self.search, self.parameters

    def _next_identity(self) -> GameIdentity:
        identity = GameIdentity(
            worker_id=self.worker_id,
            process_instance_id=self.process_instance_id,
            game_number=self.next_game_number,
        )
        self.next_game_number += 1
        return identity
