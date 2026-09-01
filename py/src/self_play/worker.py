from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Generic
from uuid import uuid4

import numpy as np
from src.games.contracts import WdlTarget
from src.search_stopping.policy import SearchStopPolicy, cap_visit_count, checkpoint_visit_counts
from src.search_stopping.records import (
    ANCHOR_RECORD_DTYPE,
    PAIRED_FLOOR_RECORD_DTYPE,
    anchor_log_path,
    append_records,
    audit_log_path,
    audit_record_dtype,
    paired_floor_log_path,
)
from src.search_stopping.sampling import AuditPositionIdentity, is_audit_position
from src.search_stopping.targets import PolicyDistribution, policy_kl
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
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
from src.util.atomic_file import fsync_directory
from src.util.log import error as log_error
from src.util.tensorboard import log_scalar


def _game_group_key(game_identity: str) -> int:
    return int.from_bytes(hashlib.sha256(game_identity.encode('utf-8')).digest()[:8], 'big')


@dataclass(frozen=True)
class _AuditPlan:
    identity: AuditPositionIdentity
    checkpoint_visits: list[int]
    paired: bool
    anchored: bool


if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceStatistics
    from src.games.implementation import GameImplementation
    from src.training.checkpoint import CheckpointReference


def _stop_reason_from_native(native_stop_reason: object) -> SearchStopReason:
    from AlphaZeroCpp import SearchStopReason as NativeSearchStopReason

    match native_stop_reason:
        case NativeSearchStopReason.FIXED_LIMIT:
            return SearchStopReason.FIXED_LIMIT
        case NativeSearchStopReason.ADDITIONAL_VISITS:
            return SearchStopReason.ADDITIONAL_VISITS
        case NativeSearchStopReason.CAP_REACHED:
            return SearchStopReason.CAP_REACHED
        case NativeSearchStopReason.LEARNED_EARLY_STOP:
            return SearchStopReason.LEARNED_EARLY_STOP
        case unknown:
            raise ValueError(f'Unknown native search stop reason: {unknown!r}')


def _dense_distributions(
    sparse_policies: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> tuple[PolicyDistribution, ...]:
    action_ids = sorted({action for actions, _ in sparse_policies for action in actions})
    index_of = {action: index for index, action in enumerate(action_ids)}
    distributions = []
    for actions, counts in sparse_policies:
        total = float(sum(counts))
        probabilities = [0.0] * len(action_ids)
        for action, count in zip(actions, counts, strict=True):
            probabilities[index_of[action]] = count / total
        distributions.append(PolicyDistribution(probabilities=tuple(probabilities)))
    return tuple(distributions)


def _sparse_policy(visits: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(visit.action_id for visit in visits),
        tuple(visit.visit_count for visit in visits),
    )


@dataclass
class ActiveSelfPlayGame(Generic[NativeRootT]):
    identity: GameIdentity
    root: NativeRootT
    started_at_seconds: float
    action_ids: list[int] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)
    reserved_restart_action_id: int | None = None
    awaiting_cut_evaluation: bool = False
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
        self.stopping_configuration = game.training.lifecycle.search_stopping
        self.stopping_path = inbox_path.parents[1] / 'search-stopping'
        self.audit_fraction = Fraction(self.stopping_configuration.audit_sample_fraction)
        self.paired_audit_fraction = Fraction(self.stopping_configuration.paired_audit_fraction)
        self.anchor_fraction = Fraction(self.stopping_configuration.anchor_fraction)

    def run_batch(self) -> None:
        search, parameters = self._loaded_runtime()
        requests: list[NativeRequestT] = []
        audit_plans: dict[int, _AuditPlan] = {}
        for game_index, active_game in enumerate(self.active_games):
            active_game.root.discount(parameters.retained_root_visit_fraction)
            plan = self._audit_plan(active_game, parameters)
            if plan is None:
                requests.append(search.request(active_game.root, root_ply=len(active_game.action_ids)))
                continue
            audit_plans[game_index] = plan
            requests.append(
                search.request(
                    active_game.root,
                    policy_checkpoint_visits=plan.checkpoint_visits,
                    checkpoint_detail=self._policies_detail(),
                    root_ply=len(active_game.action_ids),
                    audit=True,
                )
            )
        extra_requests, extra_owners = self._extra_audit_requests(search, parameters, audit_plans)
        requests.extend(extra_requests)
        batch = search.search(requests, collect_statistics=False)
        if len(batch.results) != len(self.active_games) + len(extra_requests):
            raise RuntimeError('Batched self-play search returned the wrong result count.')
        results = batch.results
        self._record_audits(parameters, audit_plans, results, extra_owners)
        game_results = results[: len(self.active_games)]
        self.completed_searches += batch.simulations_completed
        next_games: list[ActiveSelfPlayGame[NativeRootT]] = []
        published = False
        for active_game, result in zip(self.active_games, game_results, strict=True):
            completed = self._advance_game(active_game, result, parameters)
            if completed is None:
                next_games.append(active_game)
            else:
                self._archive_completed_game(completed, parameters)
                publish_completed_self_play_game(self.inbox_path, completed, sync_directory=False)
                published = True
                next_games.append(self._new_game(search, parameters))
        # A directory fsync costs as much as the file fsync, so the games finishing in one step
        # share a single one instead of each paying its own.
        if published:
            fsync_directory(self.inbox_path)
        self.active_games = next_games

    def refresh_published_model(self, checkpoint: CheckpointReference, search_stop_policy: SearchStopPolicy) -> None:
        parameters = self.game.self_play_parameters_at(checkpoint.generation, search_stop_policy)
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
        self._log_search_thread_split(inference)
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
        result: NativeResultT,
        parameters: ResolvedSelfPlayParameters,
    ) -> CompletedSelfPlayGame | None:
        assert self.model_generation is not None
        search_visits = result.search_visit_columns
        if not search_visits[0]:
            raise RuntimeError('Native search returned no visited action for a nonterminal root.')
        ply = len(active_game.action_ids)
        policy_target_columns = result.policy_target_columns
        if not policy_target_columns[0]:
            raise RuntimeError('Native search returned an empty policy target for a nonterminal root.')
        resignation_triggered = (
            not active_game.awaiting_cut_evaluation
            and not active_game.is_resignation_continuation
            and active_game.resignation_threshold is not None
            and result.root_value <= active_game.resignation_threshold
            and result.highest_visited_child_q <= active_game.resignation_threshold
        )
        selected_action_id = None if active_game.awaiting_cut_evaluation else active_game.reserved_restart_action_id
        if selected_action_id is None and not resignation_triggered and not active_game.awaiting_cut_evaluation:
            selected_action_id = self._select_action(search_visits, ply, parameters)
        observation = SearchObservation(
            ply=ply,
            model_generation=self.model_generation,
            policy_target_visits=SearchVisitCounts.from_columns(policy_target_columns),
            root_value=result.root_value,
            highest_visited_child_action_id=result.highest_visited_child_action_id,
            highest_visited_child_visit_count=result.highest_visited_child_visit_count,
            highest_visited_child_q=result.highest_visited_child_q,
            selected_action_id=(
                None if resignation_triggered or active_game.awaiting_cut_evaluation else selected_action_id
            ),
            sample_weight=parameters.primary_sample_weight,
            baseline_visits=parameters.baseline_visits,
            network_root_value=result.network_root_value,
            policy_correction=result.policy_correction,
            value_correction=result.value_correction,
            stop_checkpoint_index=result.stop_checkpoint_index,
            parallel_searches=result.parallel_searches,
            starting_visits=result.starting_visits,
            final_visits=result.final_visits,
            stop_reason=_stop_reason_from_native(result.stop_reason),
        )
        active_game.observations.append(observation)
        if active_game.awaiting_cut_evaluation:
            # The search ran at the cut position itself, so its root value already belongs to the player
            # to move there and needs no sign flip.
            return self._complete(
                active_game,
                WdlTarget.from_scalar(result.root_value),
                TerminationReason.MAXIMUM_PLIES,
            )
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
            if final_wdl is None and parameters.bootstrap_cut_game_value:
                # Evaluate the cut position itself rather than stamping the preceding move's value.
                active_game.awaiting_cut_evaluation = True
                return None
            if final_wdl is None:
                final_wdl = self.game.state.adjudicated_wdl(
                    active_game.root.position,
                    TerminationReason.MAXIMUM_PLIES,
                )
            return self._complete(active_game, final_wdl, TerminationReason.MAXIMUM_PLIES)
        return None

    def _select_action(
        self,
        visits: tuple[list[int], list[int]],
        ply: int,
        parameters: ResolvedSelfPlayParameters,
    ) -> int:
        action_ids, visit_counts = visits
        if ply >= parameters.greedy_after_ply:
            return min(zip(visit_counts, action_ids, strict=True), key=lambda visit: (-visit[0], visit[1]))[1]
        progress = ply / parameters.greedy_after_ply
        temperature = (
            parameters.starting_temperature
            + (parameters.final_temperature - parameters.starting_temperature) * progress
        )
        counts = np.asarray(visit_counts, dtype=np.float64)
        probabilities = np.power(counts, 1.0 / temperature)
        probabilities /= probabilities.sum()
        return action_ids[int(self.random.choice(len(action_ids), p=probabilities))]

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

    def _audit_plan(
        self,
        active_game: ActiveSelfPlayGame[NativeRootT],
        parameters: ResolvedSelfPlayParameters,
    ) -> _AuditPlan | None:
        policy = parameters.search_stop_policy
        if not policy.checkpoint_multiples or self.model_generation is None:
            return None
        identity = AuditPositionIdentity(
            source_generation=self.model_generation,
            game_identity=active_game.identity.archive_key,
            ply=len(active_game.action_ids),
        )
        run_seed = self.game.training.random_seed
        if not is_audit_position(identity, run_seed, self.audit_fraction):
            return None
        starting_visits = active_game.root.visits
        relative = checkpoint_visit_counts(tuple(policy.checkpoint_multiples), parameters.baseline_visits)
        return _AuditPlan(
            identity=identity,
            checkpoint_visits=[starting_visits + visits for visits in relative],
            paired=is_audit_position(identity, run_seed + 1, self.paired_audit_fraction),
            anchored=is_audit_position(identity, run_seed + 2, self.anchor_fraction),
        )

    @staticmethod
    def _policies_detail() -> object:
        from AlphaZeroCpp import SearchCheckpointDetail

        return SearchCheckpointDetail.POLICIES

    def _extra_audit_requests(
        self,
        search: NativeSearchT,
        parameters: ResolvedSelfPlayParameters,
        audit_plans: dict[int, _AuditPlan],
    ) -> tuple[list[NativeRequestT], list[tuple[int, str]]]:
        cap_visits = cap_visit_count(parameters.search_stop_policy.cap_multiple, parameters.baseline_visits)
        anchor_visits = int(round(self.stopping_configuration.anchor_visit_multiple * parameters.baseline_visits))
        extra_requests: list[NativeRequestT] = []
        extra_owners: list[tuple[int, str]] = []
        for game_index, plan in audit_plans.items():
            position = self.active_games[game_index].root.position
            if plan.paired:
                for _ in range(2):
                    extra_requests.append(
                        search.request(
                            search.new_root(position, maximum_capacity=cap_visits + 64),
                            assigned_additional_visits=cap_visits,
                        )
                    )
                    extra_owners.append((game_index, 'paired'))
            if plan.anchored:
                extra_requests.append(
                    search.request(
                        search.new_root(position, maximum_capacity=anchor_visits + 64),
                        assigned_additional_visits=anchor_visits,
                        add_root_noise=False,
                    )
                )
                extra_owners.append((game_index, 'anchor'))
        return extra_requests, extra_owners

    def _record_audits(
        self,
        parameters: ResolvedSelfPlayParameters,
        audit_plans: dict[int, _AuditPlan],
        results: list[NativeResultT],
        extra_owners: list[tuple[int, str]],
    ) -> None:
        if not audit_plans:
            return
        try:
            extras: dict[int, dict[str, list[NativeResultT]]] = {}
            for (game_index, kind), result in zip(extra_owners, results[len(self.active_games) :], strict=True):
                extras.setdefault(game_index, {}).setdefault(kind, []).append(result)
            for game_index, plan in audit_plans.items():
                self._append_audit_records(
                    plan,
                    parameters,
                    results[game_index],
                    extras.get(game_index, {}),
                )
        except Exception:
            log_error('Audit recording failed; self-play continues without this batch of audits.')

    def _append_audit_records(
        self,
        plan: _AuditPlan,
        parameters: ResolvedSelfPlayParameters,
        result: NativeResultT,
        extras: dict[str, list[NativeResultT]],
    ) -> None:
        checkpoint_count = len(plan.checkpoint_visits)
        checkpoints = list(result.checkpoints)
        if len(checkpoints) != checkpoint_count:
            raise RuntimeError('Audit search returned an unexpected checkpoint count.')
        sparse = tuple(_sparse_policy(checkpoint.policy_target_visits) for checkpoint in checkpoints) + (
            _sparse_policy(result.policy_target_visits),
        )
        distributions = _dense_distributions(sparse)
        final_distribution = distributions[-1]
        final_argmax = max(
            range(len(final_distribution.probabilities)), key=final_distribution.probabilities.__getitem__
        )
        dtype = audit_record_dtype(checkpoint_count)
        record = np.zeros(1, dtype=dtype)
        row = record[0]
        row['source_generation'] = plan.identity.source_generation
        row['model_generation'] = plan.identity.source_generation
        row['game_key'] = _game_group_key(plan.identity.game_identity)
        row['ply'] = plan.identity.ply
        row['baseline_visits'] = parameters.baseline_visits
        row['starting_visits'] = result.starting_visits
        row['final_visits'] = result.final_visits
        row['final_root_value'] = result.root_value
        for index, checkpoint in enumerate(checkpoints):
            distribution = distributions[index]
            row['kl_to_final'][index] = policy_kl(final_distribution, distribution)
            row['value_gap'][index] = abs(checkpoint.root_value - result.root_value)
            checkpoint_argmax = max(range(len(distribution.probabilities)), key=distribution.probabilities.__getitem__)
            row['argmax_swap'][index] = checkpoint_argmax != final_argmax
            row['guard_movement'][index] = result.guard_movements[index]
            row['stop_probability'][index] = result.stop_probabilities[index]
            row['would_stop'][index] = result.stop_verdicts[index]
            row['features'][index] = result.stop_features[index]
        append_records(
            audit_log_path(self.stopping_path, plan.identity.source_generation, self.worker_id),
            record,
            dtype,
        )
        paired = extras.get('paired', [])
        if len(paired) == 2:
            first, second = _dense_distributions(
                (_sparse_policy(paired[0].policy_target_visits), _sparse_policy(paired[1].policy_target_visits))
            )
            floor_record = np.zeros(1, dtype=PAIRED_FLOOR_RECORD_DTYPE)
            floor_record[0]['source_generation'] = plan.identity.source_generation
            floor_record[0]['ply'] = plan.identity.ply
            floor_record[0]['baseline_visits'] = parameters.baseline_visits
            floor_record[0]['kl_symmetric'] = 0.5 * (policy_kl(first, second) + policy_kl(second, first))
            floor_record[0]['value_gap'] = abs(paired[0].root_value - paired[1].root_value)
            append_records(
                paired_floor_log_path(self.stopping_path, plan.identity.source_generation, self.worker_id),
                floor_record,
                PAIRED_FLOOR_RECORD_DTYPE,
            )
        anchors = extras.get('anchor', [])
        if anchors:
            anchor_distribution, capped_distribution = _dense_distributions(
                (_sparse_policy(anchors[0].policy_target_visits), _sparse_policy(result.policy_target_visits))
            )
            anchor_record = np.zeros(1, dtype=ANCHOR_RECORD_DTYPE)
            anchor_record[0]['source_generation'] = plan.identity.source_generation
            anchor_record[0]['ply'] = plan.identity.ply
            anchor_record[0]['baseline_visits'] = parameters.baseline_visits
            anchor_record[0]['kl_anchor_to_capped'] = policy_kl(anchor_distribution, capped_distribution)
            append_records(
                anchor_log_path(self.stopping_path, plan.identity.source_generation, self.worker_id),
                anchor_record,
                ANCHOR_RECORD_DTYPE,
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

    def _log_search_thread_split(self, inference: InferenceStatistics) -> None:
        assert self.model_generation is not None
        log_scalar('inference/worker_utilization', inference.workerUtilization, self.model_generation)

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
