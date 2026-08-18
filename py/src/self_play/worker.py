from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic
from uuid import uuid4

import numpy as np
from src.games.contracts import WdlTarget
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchCheckpointObservation,
    SearchObservation,
    SearchStopReason,
    TerminationReason,
    publish_completed_self_play_game,
)
from src.self_play.native_search import NativeRequestT, NativeResultT, NativeRootT, NativeSearchT, PositionT
from src.self_play.parameters import (
    RandomOpeningStartParameters,
    ResolvedSelfPlayParameters,
    RestartStateStartParameters,
)
from src.self_play.resignation import (
    CalibratedResignationConfiguration,
    PublishedResignationPolicy,
)
from src.self_play.restart_archive import RestartStateArchive, worker_restart_archive_path
from src.util.tensorboard import log_scalar

if TYPE_CHECKING:
    from AlphaZeroCpp import GameSearchVisit, InferenceStatistics
    from src.games.implementation import GameImplementation
    from src.training.checkpoint import CheckpointReference


@dataclass
class ActiveSelfPlayGame(Generic[NativeRootT]):
    identity: GameIdentity
    root: NativeRootT
    started_at_seconds: float
    action_ids: list[int] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)
    reserved_restart_action_id: int | None = None
    is_resignation_continuation: bool = False
    resignation_threshold: float | None = None


@dataclass(frozen=True)
class SelfPlayStatisticsSnapshot:
    model_generation: int
    completed_searches: int
    inference: InferenceStatistics


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
        self.restart_archive_path = worker_restart_archive_path(inbox_path.parent, worker_id)
        self.random = np.random.default_rng(game.training.random_seed + worker_id)
        self.process_instance_id = uuid4()
        self.next_game_number = 0
        self.model_generation: int | None = None
        self.parameters: ResolvedSelfPlayParameters | None = None
        self.search: NativeSearchT | None = None
        self.active_games: list[ActiveSelfPlayGame[NativeRootT]] = []
        self.completed_searches = 0
        self.restart_archive: RestartStateArchive | None = None
        self.true_starts = 0
        self.restart_starts = 0
        self.empty_restart_fallbacks = 0
        self.resignation_policy = PublishedResignationPolicy()

    def run_batch(self) -> None:
        search, parameters = self._loaded_runtime()
        requests: list[NativeRequestT] = []
        for active_game in self.active_games:
            continuation_forces_fast_search = (
                parameters.force_fast_search_after_ply is not None
                and len(active_game.action_ids) >= parameters.force_fast_search_after_ply
            )
            full_search = active_game.reserved_restart_action_id is not None or (
                not continuation_forces_fast_search and self.random.random() < parameters.full_search_probability
            )
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
                self._archive_completed_game(completed, parameters)
                publish_completed_self_play_game(self.inbox_path, completed)
                next_games.append(self._new_game(search, parameters))
        self.active_games = next_games

    def refresh_published_model(self, checkpoint: CheckpointReference) -> None:
        parameters = self.game.self_play_parameters_at(checkpoint.generation)
        if self.search is None:
            self.search = self.game.create_native_search(self.device_id, checkpoint, parameters)
            capacity_changed = False
        else:
            self.search.refresh_model(checkpoint.generation, str(checkpoint.inference_model_path))
            capacity_changed = self.search.update_search_schedule(self.game.native_search_parameters(parameters))
        self.parameters = parameters
        self.model_generation = checkpoint.generation
        self._prepare_restart_archive(parameters)
        if not self.active_games:
            self.active_games = [self._new_game(self.search, parameters) for _ in range(self.parallel_game_count)]
        elif capacity_changed:
            for active_game in self.active_games:
                active_game.root = self.search.new_root(active_game.root.position)
        else:
            for active_game in self.active_games:
                active_game.root.reset()

    def update_resignation_policy(self, policy: PublishedResignationPolicy) -> None:
        self.resignation_policy = policy

    def snapshot_statistics(self) -> SelfPlayStatisticsSnapshot:
        search, _ = self._loaded_runtime()
        assert self.model_generation is not None
        inference = search.inference_statistics()
        log_scalar(
            'inference/average_number_of_positions_in_inference_call',
            inference.averageNumberOfPositionsInInferenceCall,
            self.model_generation,
        )
        self._log_restart_statistics()
        return SelfPlayStatisticsSnapshot(
            model_generation=self.model_generation,
            completed_searches=self.completed_searches,
            inference=inference,
        )

    def _new_game(
        self,
        search: NativeSearchT,
        parameters: ResolvedSelfPlayParameters,
    ) -> ActiveSelfPlayGame[NativeRootT]:
        match parameters.start_position:
            case RandomOpeningStartParameters(maximum_plies=maximum_plies):
                return self._new_random_opening_game(search, maximum_plies)
            case RestartStateStartParameters() as restart_parameters:
                return self._new_restart_or_true_game(search, restart_parameters)

    def _new_random_opening_game(
        self,
        search: NativeSearchT,
        maximum_opening_plies: int,
    ) -> ActiveSelfPlayGame[NativeRootT]:
        while True:
            position = self.game.state.initial_position()
            action_ids: list[int] = []
            opening_plies = int(self.random.integers(0, maximum_opening_plies + 1))
            for _ in range(opening_plies):
                legal_actions = self.game.state.legal_action_ids(position)
                action_id = int(self.random.choice(legal_actions))
                action_ids.append(action_id)
                position = self.game.state.child_position(position, action_id)
                if self.game.state.natural_terminal_wdl(position) is not None:
                    break
            if self.game.state.natural_terminal_wdl(position) is None:
                return self._active_game(
                    identity=self._next_identity(),
                    root=search.new_root(position),
                    started_at_seconds=time.time(),
                    action_ids=action_ids,
                )

    def _new_restart_or_true_game(
        self,
        search: NativeSearchT,
        parameters: RestartStateStartParameters,
    ) -> ActiveSelfPlayGame[NativeRootT]:
        assert self.model_generation is not None
        archive = self._required_restart_archive()
        if self.random.random() < parameters.true_start_probability:
            self.true_starts += 1
            return self._new_true_start_game(search)
        reserved = archive.reserve(self.model_generation, parameters)
        if reserved is None:
            self.empty_restart_fallbacks += 1
            self.true_starts += 1
            return self._new_true_start_game(search)
        position = self.game.state.initial_position()
        for action_id in reserved.action_prefix:
            if action_id not in self.game.state.legal_action_ids(position):
                raise RuntimeError('Restart archive contains an illegal action prefix.')
            position = self.game.state.child_position(position, action_id)
        if reserved.action_id not in self.game.state.legal_action_ids(position):
            raise RuntimeError('Restart archive contains an illegal reserved action.')
        self.restart_starts += 1
        return self._active_game(
            identity=self._next_identity(),
            root=search.new_root(position),
            started_at_seconds=time.time(),
            action_ids=list(reserved.action_prefix),
            reserved_restart_action_id=reserved.action_id,
        )

    def _new_true_start_game(self, search: NativeSearchT) -> ActiveSelfPlayGame[NativeRootT]:
        return self._active_game(
            identity=self._next_identity(),
            root=search.new_root(self.game.state.initial_position()),
            started_at_seconds=time.time(),
        )

    def _advance_game(
        self,
        active_game: ActiveSelfPlayGame[NativeRootT],
        request: NativeRequestT,
        result: NativeResultT,
        parameters: ResolvedSelfPlayParameters,
    ) -> CompletedSelfPlayGame | None:
        assert self.model_generation is not None
        search_visits = tuple(result.search_visits)
        if not search_visits:
            raise RuntimeError('Native search returned no visited action for a nonterminal root.')
        ply = len(active_game.action_ids)
        if active_game.reserved_restart_action_id is not None and not request.full_search:
            raise RuntimeError('Restart roots require a full search.')
        policy_target_visits = tuple(result.policy_target_visits)
        if not policy_target_visits:
            raise RuntimeError('Native search returned an empty policy target for a nonterminal root.')
        resignation_triggered = (
            not active_game.is_resignation_continuation
            and active_game.resignation_threshold is not None
            and result.root_value <= active_game.resignation_threshold
            and result.highest_visited_child_q <= active_game.resignation_threshold
        )
        selected_action_id = active_game.reserved_restart_action_id
        if selected_action_id is None and not resignation_triggered:
            selected_action_id = self._select_action(search_visits, ply, parameters)
        observation = SearchObservation(
            ply=ply,
            model_generation=self.model_generation,
            policy_target_visits=policy_target_visits,
            root_value=result.root_value,
            highest_visited_child_action_id=result.highest_visited_child_action_id,
            highest_visited_child_visit_count=result.highest_visited_child_visit_count,
            highest_visited_child_q=result.highest_visited_child_q,
            selected_action_id=None if resignation_triggered else selected_action_id,
            full_search=request.full_search,
            sample_weight=parameters.primary_sample_weight,
            search_budget=parameters.maximum_full_searches if request.full_search else parameters.fast_searches,
            network_root_value=result.network_root_value,
            policy_correction=result.policy_correction,
            value_correction=result.value_correction,
            search_correction_target=result.search_correction_target,
            predicted_search_correction=result.predicted_search_correction,
            starting_visits=result.starting_visits,
            final_visits=result.final_visits,
            stop_reason=SearchStopReason(result.stop_reason.name.lower()),
            learned_gate_evaluated=result.learned_gate_evaluated,
            checkpoints=tuple(
                SearchCheckpointObservation(
                    visits=checkpoint.visits,
                    leader_action_id=checkpoint.leader_action_id,
                    most_visited_action_id=checkpoint.most_visited_action_id,
                    top_visit_share=checkpoint.top_visit_share,
                    top_two_margin=checkpoint.top_two_margin,
                    root_value=checkpoint.root_value,
                    root_value_delta=checkpoint.root_value_delta,
                    leader_stable=checkpoint.leader_stable,
                )
                for checkpoint in result.checkpoints
            ),
        )
        active_game.observations.append(observation)
        if resignation_triggered:
            return self._complete(
                active_game,
                WdlTarget(win=0.0, draw=0.0, loss=1.0),
                TerminationReason.RESIGNATION,
            )
        assert selected_action_id is not None
        active_game.action_ids.append(selected_action_id)
        active_game.reserved_restart_action_id = None
        active_game.root = result.root
        active_game.root.play(selected_action_id)
        natural_wdl = self.game.state.natural_terminal_wdl(active_game.root.position)
        if natural_wdl is not None:
            return self._complete(active_game, natural_wdl, TerminationReason.NATURAL)
        if parameters.maximum_game_plies is not None and len(active_game.action_ids) >= parameters.maximum_game_plies:
            oracle = self.game.terminal_oracle
            final_wdl = None if oracle is None else oracle.probe_wdl(active_game.root.position)
            if final_wdl is None:
                final_wdl = self.game.state.adjudicated_wdl(
                    active_game.root.position,
                    TerminationReason.MAXIMUM_PLIES,
                )
            return self._complete(active_game, final_wdl, TerminationReason.MAXIMUM_PLIES)
        return None

    def _select_action(
        self,
        visits: tuple[GameSearchVisit, ...],
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
            is_resignation_continuation=active_game.is_resignation_continuation,
            resignation_threshold=active_game.resignation_threshold,
        )

    def _active_game(
        self,
        identity: GameIdentity,
        root: NativeRootT,
        started_at_seconds: float,
        action_ids: list[int] | None = None,
        reserved_restart_action_id: int | None = None,
    ) -> ActiveSelfPlayGame[NativeRootT]:
        configuration = self.game.resignation_configuration
        is_continuation = (
            isinstance(configuration, CalibratedResignationConfiguration)
            and self.random.random() < configuration.continuation_game_probability
        )
        return ActiveSelfPlayGame(
            identity=identity,
            root=root,
            started_at_seconds=started_at_seconds,
            action_ids=[] if action_ids is None else action_ids,
            reserved_restart_action_id=reserved_restart_action_id,
            is_resignation_continuation=is_continuation,
            resignation_threshold=self.resignation_policy.threshold,
        )

    def _loaded_runtime(self) -> tuple[NativeSearchT, ResolvedSelfPlayParameters]:
        if self.search is None or self.parameters is None:
            raise RuntimeError('A model must be loaded before self-play starts.')
        return self.search, self.parameters

    def close(self) -> None:
        if self.restart_archive is not None:
            self.restart_archive.close()
            self.restart_archive = None
        self.game.close()

    def _prepare_restart_archive(self, parameters: ResolvedSelfPlayParameters) -> None:
        match parameters.start_position:
            case RandomOpeningStartParameters():
                if self.restart_archive is not None:
                    self.restart_archive.close()
                    self.restart_archive = None
            case RestartStateStartParameters():
                if self.restart_archive is None:
                    self.restart_archive = RestartStateArchive(self.restart_archive_path)

    def _archive_completed_game(
        self,
        game: CompletedSelfPlayGame,
        parameters: ResolvedSelfPlayParameters,
    ) -> None:
        match parameters.start_position:
            case RandomOpeningStartParameters():
                return
            case RestartStateStartParameters() as restart_parameters:
                self._required_restart_archive().archive_completed_game(game, restart_parameters)

    def _required_restart_archive(self) -> RestartStateArchive:
        if self.restart_archive is None:
            raise RuntimeError('Restart-state self-play requires an initialized archive.')
        return self.restart_archive

    def _log_restart_statistics(self) -> None:
        if self.restart_archive is None or self.model_generation is None:
            return
        snapshot = self.restart_archive.snapshot()
        for name, value in (
            ('true_starts', self.true_starts),
            ('restart_starts', self.restart_starts),
            ('empty_fallbacks', self.empty_restart_fallbacks),
            ('archived_positions', snapshot.archived_positions),
            ('archived_candidates', snapshot.archived_candidates),
            ('exhausted_evictions', snapshot.exhausted_evictions),
            ('expired_evictions', snapshot.expired_evictions),
            ('capacity_evictions', snapshot.capacity_evictions),
            ('positions', snapshot.positions),
            ('candidates', snapshot.candidates),
        ):
            log_scalar(f'restart_states/{name}', value, self.model_generation)

    def _next_identity(self) -> GameIdentity:
        identity = GameIdentity(
            worker_id=self.worker_id,
            process_instance_id=self.process_instance_id,
            game_number=self.next_game_number,
        )
        self.next_game_number += 1
        return identity
